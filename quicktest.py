#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S²M²cinematic Quick-Test
==========================

Testet den SignMagnitudeLoss auf wenigen ausgewählten Frames.
Lädt S²M² L, führt 100 Trainingsschritte durch, und validiert
ob negative Disparitäten in die richtige Richtung (und Stärke)
getrieben werden.

Nutzung:
  python quicktest.py --root G:\\CCS \\
      --disparity_dir F:\\Tools\\...\\s2m2cinematic\\gt_anchor \\
      --frames frame_001000 frame_005000

Dauert ~5 Minuten statt 17 Stunden.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast, GradScaler

_S2M2_SRC = Path(__file__).parent.parent / 's2m2' / 'src'
if _S2M2_SRC.exists():
    sys.path.insert(0, str(_S2M2_SRC))

from finetune import load_s2m2_model, modify_forward_for_finetuning
from losses import SignMagnitudeLoss, S2M2CinematicLoss
from dataset import StereoFilmDataset


def main():
    parser = argparse.ArgumentParser(description='S²M²cinematic Quick-Test')
    parser.add_argument('--root', type=str, default='G:\\CCS')
    parser.add_argument('--disparity_dir', type=str, required=True)
    parser.add_argument('--frames', nargs='+', required=True,
                        help='Frame-Stems zum Testen (z.B. frame_001000)')
    parser.add_argument('--steps', type=int, default=100,
                        help='Trainingsschritte')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--model_type', type=str, default='L')
    args = parser.parse_args()

    ROOT = Path(args.root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sbs_dir = ROOT / 'frames' / 'sbs'
    if not sbs_dir.exists():
        sbs_dir = ROOT / 'frames' / 'fSBS_3D'
    disp_dir = Path(args.disparity_dir)

    # Gewichte finden
    weights_map = {"L": "CH256NTR3", "XL": "CH384NTR3"}
    weights_stem = weights_map[args.model_type]
    weights_path = (
        Path(args.root) / 's2m2' / 'weights' / 'pretrain_weights' / f'{weights_stem}.pth'
    )
    if not weights_path.exists():
        print(f"  S²M² weights not found at {weights_path}")
        print(f"  Please place {weights_stem}.pth at: <root>/s2m2/weights/pretrain_weights/")
        print(f"  Or specify via --s2m2_weights /path/to/weights.pth")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  S²M²cinematic Quick-Test")
    print(f"  Frames: {args.frames}")
    print(f"  Steps:  {args.steps}")
    print(f"{'=' * 60}\n")

    # ── Modell laden ──────────────────────────────────────────────────────────
    model = load_s2m2_model(weights_path, model_type=args.model_type, device=device)
    model = modify_forward_for_finetuning(model)
    model.train()

    # ── Nur Decoder trainieren (wie Phase 1) ──────────────────────────────────
    for name, param in model.named_parameters():
        if any(enc in name for enc in ['cnn_backbone', 'feat_pyramid', 'transformer']):
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainierbar: {trainable / 1e6:.1f}M Parameter")

    # ── Dataset: nur die gewünschten Frames ───────────────────────────────────
    ds = StereoFilmDataset(
        sbs_dir=sbs_dir, disparity_dir=disp_dir,
        film_structure_path=None, sample_per_film=None,
        image_height=1056, image_width=1920, training=False,
    )
    # Filtern auf gewünschte Frames
    frame_set = set(args.frames)
    ds.frames = [f for f in ds.frames if f.stem in frame_set]
    if not ds.frames:
        print(f"  FEHLER: Keine Frames gefunden für {args.frames}")
        sys.exit(1)
    print(f"  Gefunden: {len(ds.frames)} Frames")

    # ── Daten laden ───────────────────────────────────────────────────────────
    batches = []
    for i in range(len(ds)):
        batch = ds[i]
        batches.append({k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()})
    print(f"  Geladen auf GPU\n")

    # ── NCC-Disparität anzeigen (was der Loss als Ziel sieht) ─────────────────
    sign_loss_fn = SignMagnitudeLoss(block_size=11, max_disp=48,
                                     conf_threshold=0.3, downscale=2)

    print(f"  {'─' * 55}")
    print(f"  NCC Block-Matching Analyse (das Regressionsziel)")
    print(f"  {'─' * 55}")
    for batch in batches:
        stem = batch['stem']
        with torch.no_grad():
            ncc_disp, ncc_conf = sign_loss_fn._compute_ncc_disp(
                batch['left'], batch['right'])
        ncc_np = ncc_disp.squeeze().cpu().numpy()
        conf_np = ncc_conf.squeeze().cpu().numpy()
        valid = conf_np > 0.3
        neg_px = (ncc_np[valid] < -1).sum()
        neg_pct = neg_px / max(valid.sum(), 1) * 100
        print(f"  {stem}:")
        print(f"    NCC Disp: Min={ncc_np[valid].min():.1f}px  "
              f"Max={ncc_np[valid].max():.1f}px  "
              f"Median={np.median(ncc_np[valid]):.1f}px")
        print(f"    NCC Neg:  {neg_px:,} Pixel ({neg_pct:.1f}% der validen)")
        print(f"    NCC Conf: Ø={conf_np[valid].mean():.3f}  "
              f"Valid={valid.sum()/conf_np.size*100:.1f}%")
    print()

    # ── Baseline: Disparität VOR Training ─────────────────────────────────────
    print(f"  {'─' * 55}")
    print(f"  Baseline (vor Training)")
    print(f"  {'─' * 55}")
    model.eval()
    for batch in batches:
        stem = batch['stem']
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            pred_disp, _, _ = model(batch['left_u8'], batch['right_u8'])
        d = pred_disp.squeeze().cpu().numpy()
        neg_pct = (d < -0.5).sum() / d.size * 100
        print(f"  {stem}: P98={np.percentile(d, 98):.1f}px  "
              f"Min={d.min():.1f}px  NegPix={neg_pct:.1f}%")
    print()

    # ── Mini-Training ─────────────────────────────────────────────────────────
    print(f"  {'─' * 55}")
    print(f"  Mini-Training: {args.steps} Schritte")
    print(f"  {'─' * 55}")

    criterion = S2M2CinematicLoss(w_photo=1.0, w_anchor=0.3, w_sign=1.0, w_smooth=0.1)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()
    model.train()

    for step in range(1, args.steps + 1):
        batch = batches[step % len(batches)]

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=torch.float16):
            pred_disp, _, _ = model(batch['left_u8'], batch['right_u8'])
            loss, parts = criterion(
                pred_disp, batch['left'], batch['right'],
                batch['gt_disp'], batch['gt_conf'], batch['gt_occ'])

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 10 == 0 or step == 1:
            print(f"    Step {step:3d}: loss={loss.item():.4f}  "
                  f"pho={parts['photo']:.4f}  anc={parts['anchor']:.4f}  "
                  f"sgn={parts['sign']:.4f}  smo={parts['smooth']:.4f}")

    # ── Ergebnis: Disparität NACH Training ────────────────────────────────────
    print(f"\n  {'─' * 55}")
    print(f"  Ergebnis (nach {args.steps} Schritten)")
    print(f"  {'─' * 55}")
    model.eval()
    for batch in batches:
        stem = batch['stem']
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            pred_disp, _, _ = model(batch['left_u8'], batch['right_u8'])
        d = pred_disp.squeeze().cpu().numpy()
        neg_pct = (d < -0.5).sum() / d.size * 100
        print(f"  {stem}: P98={np.percentile(d, 98):.1f}px  "
              f"Min={d.min():.1f}px  NegPix={neg_pct:.1f}%")

        # Vergleich NCC vs. Predicted in negativen Bereichen
        with torch.no_grad():
            ncc_disp, ncc_conf = sign_loss_fn._compute_ncc_disp(
                batch['left'], batch['right'])
        ncc_np = ncc_disp.squeeze().cpu().numpy()
        conf_np = ncc_conf.squeeze().cpu().numpy()
        neg_mask = (ncc_np < -1) & (conf_np > 0.3)
        if neg_mask.any():
            print(f"    In NCC-negativen Bereichen:")
            print(f"      NCC-Ziel:  Ø={ncc_np[neg_mask].mean():.1f}px")
            print(f"      Predicted: Ø={d[neg_mask].mean():.1f}px")
            print(f"      Lücke:     {d[neg_mask].mean() - ncc_np[neg_mask].mean():.1f}px")

    print(f"\n{'=' * 60}")
    print(f"  Quick-Test abgeschlossen")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
