"""
S²M²translucent — Erweiterte Loss-Klasse
==========================================

Fügt TranslucencyLoss als 5. Komponente zu S2M2CinematicLoss hinzu.
Alle bisherigen Loss-Terme bleiben erhalten — der Transparenz-Term kommt dazu.

Verwendung in finetune.py (Phase 3 / Nachtraining):
    from s2m2translucent_loss import S2M2TranslucentLoss
    criterion = S2M2TranslucentLoss(
        w_photo=1.0, w_anchor=0.2, w_sign=0.5, w_smooth=0.05,
        w_translucent=0.5
    )

Trainings-Empfehlung:
  - Starte von den besten S²M²cinematic Gewichten
  - LR: 2e-5 (niedrig, nur Fine-Tuning)
  - Epochen: 3-5
  - Phase: Voller Encoder+Decoder (nicht eingefroren)
  - Auflösung: wie Phase 2 (960×512)
"""

import torch
import torch.nn as nn

# Importiere die bestehenden Loss-Komponenten
from losses import (
    PhotometricLoss,
    GTAnchorLoss,
    SignMagnitudeLoss,
    SmoothnessLoss,
)
from translucency_loss import TranslucencyLoss


class S2M2TranslucentLoss(nn.Module):
    """
    Kombinierter Loss für S²M²translucent Fine-Tuning.

    L_total = w_photo  * L_photometric
            + w_anchor * L_gt_anchor
            + w_sign   * L_sign_magnitude
            + w_smooth * L_smoothness
            + w_translucent * L_translucency    ← NEU

    Empfohlene Gewichte für das Nachtraining:
        w_photo=1.0       Photometric bleibt Hauptkomponente
        w_anchor=0.2      Anchor leicht (Modell kennt positive Disp bereits)
        w_sign=0.5        Sign reduziert (Vorzeichen sind bereits gelernt)
        w_smooth=0.05     Smoothness minimal
        w_translucent=0.5 Transparenz-Awareness — neuer Fokus
    """

    def __init__(
        self,
        w_photo=1.0,
        w_anchor=0.2,
        w_sign=0.5,
        w_smooth=0.05,
        w_translucent=0.5,
        # TranslucencyLoss Parameter
        conf_threshold=0.4,
        mismatch_threshold=0.08,
        grad_threshold=2.0,
        boost_factor=3.0,
    ):
        super().__init__()
        self.photo_loss = PhotometricLoss(alpha=0.85)
        self.anchor_loss = GTAnchorLoss(threshold=2.0)
        self.sign_loss = SignMagnitudeLoss(
            block_size=11, max_disp=48, conf_threshold=0.3, downscale=2,
            sign_weight=0.3, magnitude_weight=0.7,
        )
        self.smooth_loss = SmoothnessLoss()
        self.translucent_loss = TranslucencyLoss(
            conf_threshold=conf_threshold,
            mismatch_threshold=mismatch_threshold,
            grad_threshold=grad_threshold,
            boost_factor=boost_factor,
        )

        self.w_photo = w_photo
        self.w_anchor = w_anchor
        self.w_sign = w_sign
        self.w_smooth = w_smooth
        self.w_translucent = w_translucent

    def forward(self, pred_disp, left_img, right_img, gt_disp, gt_conf, gt_occ):
        """
        Gibt (total_loss, loss_dict) zurück.
        loss_dict enthält alle 5+1 Einzelkomponenten für Logging.
        """
        l_photo = self.photo_loss(pred_disp, left_img, right_img)
        l_anchor = self.anchor_loss(pred_disp, gt_disp, gt_conf, gt_occ)
        l_sign = self.sign_loss(pred_disp, left_img, right_img)
        l_smooth = self.smooth_loss(pred_disp, left_img)
        l_trans, trans_stats = self.translucent_loss(
            pred_disp, left_img, right_img, gt_disp, gt_conf, gt_occ
        )

        total = (self.w_photo * l_photo +
                 self.w_anchor * l_anchor +
                 self.w_sign * l_sign +
                 self.w_smooth * l_smooth +
                 self.w_translucent * l_trans)

        return total, {
            'photo': l_photo.item(),
            'anchor': l_anchor.item(),
            'sign': l_sign.item(),
            'smooth': l_smooth.item(),
            'translucent': l_trans.item(),
            'translucent_pct': trans_stats['translucent_pct'],
            'total': total.item(),
        }
