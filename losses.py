"""
S²M²cinematic Loss Functions
==============================

Vier Loss-Komponenten für bidirektionales Fine-Tuning:

1. PhotometricLoss:  Self-Supervised auf Stereopaar (SSIM + L1)
                     → Lernt positive UND negative Disparitäten
2. GTAnchorLoss:     Supervised auf S²M²-positivity-GT, trust-gewichtet
                     → Verhindert Catastrophic Forgetting bei positiver Disparität
3. SignGuideLoss:    Bidirektionale NCC-basierte Richtungskorrektur
                     → Aktiver Anreiz für negative Disparitäten
4. SmoothnessLoss:   Edge-aware Glattheit der Disparitätskarte
                     → Regularisiert negative Bereiche wo kein GT existiert
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhotometricLoss(nn.Module):
    """
    Self-Supervised Photometric Loss.

    Warpt das rechte Auge mit der vorhergesagten Disparität zum linken
    und misst den Rekonstruktionsfehler. Funktioniert für positive UND
    negative Disparitäten — das Stereopaar enthält die volle Information.

    L = alpha * SSIM + (1 - alpha) * L1
    """

    def __init__(self, alpha=0.85):
        super().__init__()
        self.alpha = alpha

    def _ssim(self, x, y):
        """Structural Similarity (Wang et al. 2004), gepatcht."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        # clamp(min=0) für numerische Stabilität: E[X²] - E[X]² kann durch
        # Floating-Point-Fehler minimal negativ werden
        sigma_x = (F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2).clamp(min=0)
        sigma_y = (F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2).clamp(min=0)
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
        ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        return torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1)

    def forward(self, disp, left_img, right_img):
        """
        disp:      [B, 1, H, W]  vorhergesagte Disparität in Pixel (positiv UND negativ)
        left_img:  [B, 3, H, W]  linkes Auge [0,1]
        right_img: [B, 3, H, W]  rechtes Auge [0,1]

        Warping: x_right = x_left - disp
          disp > 0: Pixel liegt links im rechten Bild (hinter Konvergenz)
          disp < 0: Pixel liegt rechts im rechten Bild (vor Konvergenz)
        """
        B, _, H, W = disp.shape

        # Sampling Grid — einmalig berechnen, kein expand nötig dank Broadcasting
        x_base = torch.linspace(0, W - 1, W, device=disp.device, dtype=disp.dtype)
        x_base = x_base.view(1, 1, 1, W)  # [1,1,1,W] — broadcastet über B,1,H
        y_base = torch.linspace(0, H - 1, H, device=disp.device, dtype=disp.dtype)
        y_base = y_base.view(1, 1, H, 1)  # [1,1,H,1]

        # Rechtes Auge Sampling: x_right = x_left - disp
        x_sample = x_base - disp  # [B,1,H,W]

        # Normalisieren auf [-1, 1] für grid_sample
        W_f = float(W - 1) if W > 1 else 1.0
        H_f = float(H - 1) if H > 1 else 1.0
        x_norm = 2.0 * x_sample / W_f - 1.0
        y_norm = (2.0 * y_base / H_f - 1.0).expand(B, 1, H, W)
        grid = torch.cat([x_norm, y_norm], dim=1).permute(0, 2, 3, 1)  # [B,H,W,2]

        # Warpen
        right_warped = F.grid_sample(right_img, grid,
                                     mode='bilinear', padding_mode='border',
                                     align_corners=True)

        # Gültigkeitsmaske: Warp innerhalb des Bildes
        valid = ((x_sample > 0) & (x_sample < W - 1)).float()

        # Photometric Error
        l1_err = (left_img - right_warped).abs().mean(dim=1, keepdim=True)
        ssim_err = self._ssim(left_img, right_warped).mean(dim=1, keepdim=True)
        photo_err = self.alpha * ssim_err + (1 - self.alpha) * l1_err

        n_valid = valid.sum().clamp(min=1.0)
        return (photo_err * valid).sum() / n_valid


class GTAnchorLoss(nn.Module):
    """
    Trust-gewichteter GT-Anker-Loss.

    Nutzt die S²M²-positivity-Disparitäten als Anker für den positiven
    Disparitätsbereich. Nur Pixel mit GT > threshold werden verankert.
    Near-Zero-Bereiche (wo negative Parallaxe sein könnte) bleiben frei.

    L = mean(trust * |pred_disp - gt_disp|) für Pixel mit gt_disp > threshold
    """

    def __init__(self, threshold=2.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred_disp, gt_disp, gt_conf, gt_occ):
        """
        pred_disp: [B, 1, H, W]  vorhergesagte Disparität (Pixel)
        gt_disp:   [B, 1, H, W]  S²M² positivity GT (Pixel, >= 0)
        gt_conf:   [B, 1, H, W]  S²M² Konfidenz [0,1]
        gt_occ:    [B, 1, H, W]  S²M² Okklusion [0,1]
        """
        # Nur Pixel mit klar positiver GT ankern (>2px)
        pos_mask = (gt_disp > self.threshold).float()

        # Trust nur auf positiven Pixeln, KEIN Minimum
        trust = gt_conf * gt_occ * pos_mask

        # Gewichteter L1
        diff = (pred_disp - gt_disp).abs()
        weighted = trust * diff
        n_valid = trust.sum().clamp(min=1.0)

        return weighted.sum() / n_valid


class SignMagnitudeLoss(nn.Module):
    """
    Bidirektionale Disparitäts-Regression via Block-Matching auf dem Stereopaar.

    Berechnet für jeden Pixel die beste horizontale Verschiebung zwischen
    linkem und rechtem Auge mittels Normalized Cross-Correlation (NCC).
    Sucht in BEIDE Richtungen (-max_disp bis +max_disp).

    ZWEI Loss-Komponenten:

    1. Sign-Hinge: Bestraft falsches Vorzeichen (wie bisher)
       L_sign = max(0, -pred_disp * sign(ncc_disp))

    2. Magnitude-Regression: Zieht pred_disp zum NCC-Offset (skaliert auf
       Original-Auflösung), gewichtet durch NCC-Konfidenz.
       L_mag = conf * |pred_disp - ncc_disp_scaled|

       Das gibt dem Modell ein konkretes Ziel: "Dieses Pixel sollte -8px
       sein, nicht -0.3px."

    Performance: Berechnung auf 1/2 Auflösung (Downscale 2) für bessere
    Disparitäts-Genauigkeit als bei 1/4.
    """

    def __init__(self, block_size=11, max_disp=48, conf_threshold=0.3,
                 downscale=2, sign_weight=0.3, magnitude_weight=0.7):
        """
        block_size:       NCC-Blockgröße (ungerade)
        max_disp:         Maximale Suchweite in Pixel (auf Originalauflösung)
        conf_threshold:   Minimale NCC-Korrelation für gültige Zuordnung
        downscale:        Faktor für Auflösungsreduktion
        sign_weight:      Gewicht für Vorzeichen-Hinge-Loss
        magnitude_weight: Gewicht für Magnitude-Regression
        """
        super().__init__()
        self.block_size = block_size
        self.max_disp = max_disp
        self.conf_threshold = conf_threshold
        self.downscale = downscale
        self.sign_weight = sign_weight
        self.magnitude_weight = magnitude_weight

    @torch.no_grad()
    def _compute_ncc_disp(self, left_img, right_img):
        """
        Berechnet NCC-Disparität und Konfidenz via blockweisem NCC.

        Returns:
            ncc_disp [B, 1, H, W] float32 — Disparität in Pixel (Originalauflösung)
            ncc_conf [B, 1, H, W] float32 — Konfidenz [0, 1]
        """
        B, C, H_orig, W_orig = left_img.shape
        ds = self.downscale

        # Downscale
        if ds > 1:
            left_ds = F.avg_pool2d(left_img, ds, ds)
            right_ds = F.avg_pool2d(right_img, ds, ds)
        else:
            left_ds = left_img
            right_ds = right_img

        _, _, H, W = left_ds.shape
        max_disp_ds = self.max_disp // ds

        # Grayscale
        left_gray = left_ds.mean(dim=1, keepdim=True)
        right_gray = right_ds.mean(dim=1, keepdim=True)

        pad = self.block_size // 2
        best_corr = torch.full((B, 1, H, W), -1.0,
                               device=left_img.device, dtype=torch.float32)
        best_disp = torch.zeros((B, 1, H, W),
                                device=left_img.device, dtype=torch.float32)

        # Lokale Statistiken linkes Bild (einmalig)
        left_mean = F.avg_pool2d(left_gray, self.block_size, stride=1, padding=pad)
        left_var = (F.avg_pool2d(left_gray ** 2, self.block_size,
                                  stride=1, padding=pad) - left_mean ** 2).clamp(min=1e-8)
        left_std = left_var.sqrt()

        disp_tensors = {}

        for d in range(-max_disp_ds, max_disp_ds + 1):
            if d == 0:
                continue

            # Shift
            if d > 0:
                right_shifted = torch.zeros_like(right_gray)
                if d < W:
                    right_shifted[:, :, :, :W - d] = right_gray[:, :, :, d:]
            else:
                ad = -d
                right_shifted = torch.zeros_like(right_gray)
                if ad < W:
                    right_shifted[:, :, :, ad:] = right_gray[:, :, :, :W - ad]

            # NCC
            right_mean = F.avg_pool2d(right_shifted, self.block_size,
                                       stride=1, padding=pad)
            right_var = (F.avg_pool2d(right_shifted ** 2, self.block_size,
                                       stride=1, padding=pad) - right_mean ** 2).clamp(min=1e-8)
            right_std = right_var.sqrt()

            cross = F.avg_pool2d(left_gray * right_shifted, self.block_size,
                                  stride=1, padding=pad)
            ncc = (cross - left_mean * right_mean) / (left_std * right_std + 1e-8)

            better = ncc > best_corr
            if better.any():
                if d not in disp_tensors:
                    disp_tensors[d] = torch.tensor(float(d), device=best_disp.device,
                                                   dtype=best_disp.dtype)
                best_corr = torch.where(better, ncc, best_corr)
                best_disp = torch.where(better, disp_tensors[d], best_disp)

        # Skalierung auf Originalauflösung: Disparität * downscale
        ncc_disp = best_disp * float(ds)
        ncc_conf = best_corr.clamp(min=0)

        # Upscale
        if ds > 1:
            ncc_disp = F.interpolate(ncc_disp, size=(H_orig, W_orig), mode='nearest')
            ncc_conf = F.interpolate(ncc_conf, size=(H_orig, W_orig), mode='nearest')

        # Konfidenz maskieren
        valid = (ncc_conf > self.conf_threshold).float()
        ncc_conf = ncc_conf * valid

        return ncc_disp, ncc_conf

    def forward(self, pred_disp, left_img, right_img):
        """
        pred_disp: [B, 1, H, W]  vorhergesagte Disparität
        left_img:  [B, 3, H, W]  linkes Auge [0,1]
        right_img: [B, 3, H, W]  rechtes Auge [0,1]
        """
        ncc_disp, ncc_conf = self._compute_ncc_disp(left_img, right_img)

        active = (ncc_conf > 0).float()
        n_active = active.sum().clamp(min=1.0)

        # 1. Sign-Hinge: Bestrafe falsches Vorzeichen
        sign_map = torch.sign(ncc_disp) * active
        sign_violation = torch.clamp(-pred_disp * sign_map, min=0.0)
        l_sign = (sign_violation * active).sum() / n_active

        # 2. Magnitude-Regression: Ziehe zum NCC-Offset, gewichtet durch Konfidenz
        mag_diff = (pred_disp - ncc_disp).abs()
        l_mag = (ncc_conf * mag_diff * active).sum() / n_active

        return self.sign_weight * l_sign + self.magnitude_weight * l_mag


class SmoothnessLoss(nn.Module):
    """
    Edge-aware Smoothness Loss.

    Regularisiert die Disparitätskarte — besonders wichtig für die
    negativen Bereiche, wo kein GT existiert. Kanten im Bild erlauben
    Diskontinuitäten in der Disparität (Tiefensprünge an Objektgrenzen).

    L = mean(|∂d/∂x| * exp(-|∂I/∂x|)) + mean(|∂d/∂y| * exp(-|∂I/∂y|))
    """

    def forward(self, disp, img):
        """
        disp: [B, 1, H, W]  Disparität
        img:  [B, 3, H, W]  Bild [0,1]
        """
        disp_dx = (disp[:, :, :, 1:] - disp[:, :, :, :-1]).abs()
        disp_dy = (disp[:, :, 1:, :] - disp[:, :, :-1, :]).abs()

        img_dx = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        img_dy = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)

        smooth_x = disp_dx * torch.exp(-img_dx)
        smooth_y = disp_dy * torch.exp(-img_dy)

        return smooth_x.mean() + smooth_y.mean()


class S2M2CinematicLoss(nn.Module):
    """
    Kombinierter Loss für S²M²cinematic Fine-Tuning.

    L_total = w_photo * L_photometric
            + w_anchor * L_gt_anchor
            + w_sign  * L_sign_magnitude
            + w_smooth * L_smoothness

    Phase 1 (Decoder only): w_photo=1.0, w_anchor=0.3, w_sign=1.0, w_smooth=0.1
    Phase 2 (Full model):   w_photo=1.0, w_anchor=0.2, w_sign=1.0, w_smooth=0.05
    """

    def __init__(self, w_photo=1.0, w_anchor=0.3, w_sign=1.0, w_smooth=0.1):
        super().__init__()
        self.photo_loss = PhotometricLoss(alpha=0.85)
        self.anchor_loss = GTAnchorLoss(threshold=2.0)
        self.sign_loss = SignMagnitudeLoss(
            block_size=11, max_disp=48, conf_threshold=0.3, downscale=2,
            sign_weight=0.3, magnitude_weight=0.7)
        self.smooth_loss = SmoothnessLoss()
        self.w_photo = w_photo
        self.w_anchor = w_anchor
        self.w_sign = w_sign
        self.w_smooth = w_smooth

    def forward(self, pred_disp, left_img, right_img, gt_disp, gt_conf, gt_occ):
        """
        Gibt (total_loss, loss_dict) zurück.
        loss_dict enthält die Einzelkomponenten für Logging.
        """
        l_photo = self.photo_loss(pred_disp, left_img, right_img)
        l_anchor = self.anchor_loss(pred_disp, gt_disp, gt_conf, gt_occ)
        l_sign = self.sign_loss(pred_disp, left_img, right_img)
        l_smooth = self.smooth_loss(pred_disp, left_img)

        total = (self.w_photo * l_photo +
                 self.w_anchor * l_anchor +
                 self.w_sign * l_sign +
                 self.w_smooth * l_smooth)

        return total, {
            'photo': l_photo.item(),
            'anchor': l_anchor.item(),
            'sign': l_sign.item(),
            'smooth': l_smooth.item(),
            'total': total.item(),
        }
