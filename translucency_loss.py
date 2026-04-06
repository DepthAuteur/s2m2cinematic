"""
S²M²translucent — Transparenz-Awareness Loss
===============================================

Zusätzlicher Loss-Term für S²M²cinematic der transparente/semi-transparente
Bereiche im Stereopaar erkennt und das Modell trainiert, durch diese Bereiche
hindurchzuschauen.

Kernidee:
  Transparente Materialien erzeugen im Stereopaar ein charakteristisches
  Signal: Die Confidence ist niedrig (weil das Material den Stereo-Match
  erschwert), aber die Disparität ist trotzdem physikalisch plausibel
  (weil das SBS-Paar die Parallaxe enthält). Diese Low-Confidence-
  High-Validity Bereiche sind genau die transparenten Regionen.

  Zusätzlich zeigen transparente Bereiche einen Photometric-Mismatch
  zwischen linkem und rechtem Auge — das Material verzerrt/bricht das
  Licht unterschiedlich in beiden Augen. Dieser binokulare Unterschied
  ist ein starkes Signal das kein synthetisches Dataset braucht.

Datenquelle:
  Echte Filme! Titanic (Unterwasser/Glas), Avatar (Hologramme/Biolumineszenz),
  Sea Rex (Aquariumglas), Hugo (Uhrglas/Fenster), etc.
  Keine zusätzlichen Daten, kein synthetisches Dataset, keine Roboter.

Integration:
  Wird als 5. Term in S2M2CinematicLoss eingefügt.
  Training: 3-5 Epochen Nachtraining auf den besten S²M²cinematic Gewichten.

Verwendung:
  criterion = S2M2TranslucentLoss(
      w_photo=1.0, w_anchor=0.2, w_sign=1.0,
      w_smooth=0.05, w_translucent=0.5
  )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TranslucencyLoss(nn.Module):
    """
    Transparenz-Awareness Loss für S²M²cinematic.

    Erkennt transparente Bereiche aus drei Signalen:
      1. Low-Confidence Mask:  Confidence < threshold → potentiell transparent
      2. High-Disparity-Grad:  Starke Gradienten → Tiefensprünge an Glasgrenzen
      3. Binocular Mismatch:   Photometrischer Unterschied links/rechts
                               (Lichtbrechung durch Material)

    Loss: Gewichtetes L1 das transparente Bereiche stärker bestraft.
    Ziel: Das Modell lernt in diesen Bereichen trotz niedriger Confidence
    eine korrekte Disparität vorherzusagen (durch das Material hindurch).

    Args:
        conf_threshold:      Confidence-Schwelle für "potentiell transparent" (0.3–0.5)
        mismatch_threshold:  Binokularer Mismatch-Schwelle (0.05–0.15)
        grad_threshold:      Disparitäts-Gradient-Schwelle für Glasgrenzen (2.0 px)
        boost_factor:        Verstärkungsfaktor für transparente Pixel (2.0–5.0)
    """

    def __init__(
        self,
        conf_threshold: float = 0.4,
        mismatch_threshold: float = 0.08,
        grad_threshold: float = 2.0,
        boost_factor: float = 3.0,
    ):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.mismatch_threshold = mismatch_threshold
        self.grad_threshold = grad_threshold
        self.boost_factor = boost_factor

    def _detect_translucent_regions(self, pred_disp, left_img, right_img,
                                      gt_conf, gt_occ):
        """
        Erkennt transparente Bereiche aus dem Stereopaar.

        Returns:
            translucent_mask: [B, 1, H, W] float in [0, 1]
                Höhere Werte = höhere Wahrscheinlichkeit für Transparenz.
                0.0 = sicher opak, 1.0 = sicher transparent.
        """
        B, _, H, W = pred_disp.shape

        # ── Signal 1: Low-Confidence Bereiche ────────────────────────────────
        # Confidence < threshold → Stereo-Match war unsicher → potentiell
        # verursacht durch transparentes Material das den Match erschwert
        low_conf = (gt_conf < self.conf_threshold).float()  # [B, 1, H, W]

        # Aber: Nicht ALLE Low-Confidence Bereiche sind transparent.
        # Ausschluss: Okkludierte Bereiche (dort ist Low-Confidence normal)
        not_occluded = (gt_occ > 0.5).float()  # occ=1 → sichtbar in beiden Augen
        low_conf_visible = low_conf * not_occluded

        # ── Signal 2: Binokularer Mismatch ───────────────────────────────────
        # Transparente Materialien brechen Licht unterschiedlich in beiden Augen.
        # Warpe das rechte Bild mit der GT-Disparität zum linken Bild und
        # messe den verbleibenden Rekonstruktionsfehler.
        # Hoher Fehler trotz korrekter Disparität → Material-Effekt.
        with torch.no_grad():
            # Warp right → left mit pred_disp
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=pred_disp.device),
                torch.linspace(-1, 1, W, device=pred_disp.device),
                indexing='ij',
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

            # Disparität → Pixel-Shift → normalisierter Grid-Offset
            shift_norm = pred_disp.squeeze(1).unsqueeze(-1) * 2.0 / W
            grid_warped = grid.clone()
            grid_warped[..., 0] = grid[..., 0] + shift_norm.squeeze(-1)

            warped_right = F.grid_sample(
                right_img, grid_warped,
                mode='bilinear', padding_mode='border', align_corners=True
            )

            # Rekonstruktionsfehler (sollte bei perfektem Match ~0 sein)
            recon_error = (left_img - warped_right).abs().mean(dim=1, keepdim=True)

            # Hoher Fehler → binokularer Mismatch → Transparenz-Signal
            high_mismatch = (recon_error > self.mismatch_threshold).float()

        # ── Signal 3: Starke Disparitäts-Gradienten ──────────────────────────
        # Glasgrenzen erzeugen scharfe Tiefensprünge die nicht mit Bildkanten
        # korrelieren (im Gegensatz zu Objektgrenzen)
        with torch.no_grad():
            disp_dx = (pred_disp[:, :, :, 1:] - pred_disp[:, :, :, :-1]).abs()
            disp_dy = (pred_disp[:, :, 1:, :] - pred_disp[:, :, :-1, :]).abs()

            img_dx = (left_img[:, :, :, 1:] - left_img[:, :, :, :-1]).abs().mean(1, keepdim=True)
            img_dy = (left_img[:, :, 1:, :] - left_img[:, :, :-1, :]).abs().mean(1, keepdim=True)

            # Tiefensprung ohne Bildkante → Glasgrenze
            disp_grad = F.pad(disp_dx, (0, 1)) + F.pad(disp_dy, (0, 0, 0, 1))
            img_grad = F.pad(img_dx, (0, 1)) + F.pad(img_dy, (0, 0, 0, 1))
            glass_edge = ((disp_grad > self.grad_threshold) &
                          (img_grad < 0.1)).float()

        # ── Kombination: Soft-Mask ────────────────────────────────────────────
        # Jedes Signal ist ein unabhängiger Hinweis auf Transparenz.
        # Multiplikation wäre zu restriktiv — Addition mit Clamp ist robuster.
        translucent_score = (
            0.4 * low_conf_visible +
            0.4 * high_mismatch +
            0.2 * glass_edge
        ).clamp(0.0, 1.0)

        # Leichtes Smoothing (5×5 Average) um Einzelpixel-Rauschen zu entfernen
        translucent_mask = F.avg_pool2d(
            translucent_score, kernel_size=5, stride=1, padding=2
        ).clamp(0.0, 1.0)

        return translucent_mask

    def forward(self, pred_disp, left_img, right_img, gt_disp,
                gt_conf, gt_occ):
        """
        pred_disp: [B, 1, H, W]  vorhergesagte Disparität
        left_img:  [B, 3, H, W]  linkes Auge [0,1]
        right_img: [B, 3, H, W]  rechtes Auge [0,1]
        gt_disp:   [B, 1, H, W]  GT-Disparität (aus S²M²cinematic)
        gt_conf:   [B, 1, H, W]  GT-Confidence [0,1]
        gt_occ:    [B, 1, H, W]  GT-Occlusion-Map [0=okkludiert, 1=sichtbar]

        Returns:
            loss:    Skalar
            stats:   Dict mit Diagnose-Werten
        """
        # Transparente Bereiche erkennen
        trans_mask = self._detect_translucent_regions(
            pred_disp, left_img, right_img, gt_conf, gt_occ
        )

        n_trans = trans_mask.sum().clamp(min=1.0)
        trans_pct = trans_mask.mean().item() * 100

        # Gewichtetes L1: transparente Pixel × boost_factor
        # Opake Pixel behalten Gewicht 1.0, transparente bekommen boost_factor
        weight_map = 1.0 + (self.boost_factor - 1.0) * trans_mask

        # Nur dort wo GT gültig ist (conf > 0.05 ODER translucent erkannt)
        # In transparenten Bereichen akzeptieren wir auch niedrige Confidence
        # weil wir wissen dass die GT trotzdem korrekt sein kann
        gt_valid = ((gt_conf > 0.05) | (trans_mask > 0.3)).float()

        pixel_error = (pred_disp - gt_disp).abs() * weight_map * gt_valid
        loss = pixel_error.sum() / gt_valid.sum().clamp(min=1.0)

        return loss, {
            'translucent_pct': trans_pct,
            'translucent_loss': loss.item(),
            'n_translucent_px': int(n_trans.item()),
        }
