#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S²M²cinematic Validierung
===========================

Vergleicht S²M²cinematic (bidirektional) mit S²M² positivity (original)
auf Referenz-Frames. Erzeugt Vergleichs-Visualisierungen.

Nutzung:
  python validate.py --root G:\\CCS --weights checkpoints/s2m2cinematic_best.pth \\
      --frames frame_043420 frame_050768 frame_068500
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_S2M2_SRC = Path(__file__).parent.parent / 's2m2' / 'src'
if _S2M2_SRC.exists():
    sys.path.insert(0, str(_S2M2_SRC))
from s2m2.core.model.s2m2 import S2M2


def model_params(model_type: str) -> dict:
    """
    Gibt S²M²-Konstruktor-Parameter fuer L oder XL zurueck.
      L:  feature_channels=256, ~180.7M Parameter (CH256NTR3.pth)
      XL: feature_channels=384, ~406M Parameter  (CH384NTR3.pth)
    """
    if model_type.upper() == 'L':
        return dict(feature_channels=256, dim_expansion=1,
                    num_transformer=3, refine_iter=3)
    elif model_type.upper() == 'XL':
        return dict(feature_channels=384, dim_expansion=1,
                    num_transformer=3, refine_iter=3)
    else:
        raise ValueError(f"Unbekannter model_type '{model_type}' — verwende 'L' oder 'XL'")


def load_cinematic_model(weights_path, device, model_type='L'):
    """S²M²cinematic Modell laden (use_positivity=False)."""
    params = model_params(model_type)
    model = S2M2(**params, use_positivity=False)
    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=True)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.my_load_state_dict(sd)
    # forward patchen (wie in finetune.py)
    from finetune import modify_forward_for_finetuning
    model = modify_forward_for_finetuning(model)
    return model.to(device).eval()


def load_original_model(weights_path, device, model_type='XL'):
    """S²M² Original laden (use_positivity=True)."""
    params = model_params(model_type)
    model = S2M2(**params, use_positivity=True)
    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=True)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.my_load_state_dict(sd)
    return model.to(device).eval()


def load_sbs_frame(path, H=1056, W=1920):
    sbs = cv2.imread(str(path))
    if sbs is None:
        return None, None
    sbs = cv2.cvtColor(sbs, cv2.COLOR_BGR2RGB)
    h, w = sbs.shape[:2]
    half_w = w // 2
    left = cv2.resize(sbs[:, :half_w], (W, H))
    right = cv2.resize(sbs[:, half_w:], (W, H))
    return left, right


def to_input(img):
    """RGB uint8 [H,W,3] → Tensor [1,3,H,W] uint8."""
    return torch.from_numpy(img.transpose(2, 0, 1).copy()).unsqueeze(0).contiguous()


def colormap(disp, signed=False, green_width=2):
    """Disparität → Farbbild.

    Args:
        disp: Disparitäts-Array [H, W]
        signed: True für rot/grün/blau Signed-Darstellung
        green_width: Schärfe der Grün-Zone in der Signed-Darstellung.
            Höherer Wert = schmalerer Grün-Bereich um Null.
            k=2 (Original): Grün reicht bis ±mx/2 → subtile Disparitäten unsichtbar
            k=4: Grün reicht bis ±mx/4 → guter Kompromiss
            k=6: Grün reicht bis ±mx/6 → maximale Sensitivität
    """
    if signed:
        mx = max(np.abs(disp).max(), 1e-6)
        r = np.clip(disp / mx, 0, 1)
        b = np.clip(-disp / mx, 0, 1)
        g = np.clip(1.0 - np.abs(disp / mx) * green_width, 0, 1)
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    else:
        valid = np.abs(disp)[np.abs(disp) > 0.001]
        if len(valid) < 100:
            return np.zeros((*disp.shape, 3), dtype=np.uint8)
        p2, p98 = np.percentile(valid, [2, 98])
        rng = max(p98 - p2, 1e-6)
        norm = np.clip((np.abs(disp) - p2) / rng, 0, 1)
        return cv2.applyColorMap((norm * 255).astype(np.uint8),
                                 cv2.COLORMAP_TURBO)[:, :, ::-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--original_weights', type=str, default=None)
    parser.add_argument('--frames', nargs='+', required=True)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='L',
                        choices=['L', 'XL'],
                        help='S²M²cinematic Modell-Groesse: L (256ch, default) '
                             'oder XL (384ch)')
    parser.add_argument('--original_model_type', type=str, default='XL',
                        choices=['L', 'XL'],
                        help='Original S²M² Modell-Groesse (default: XL)')
    parser.add_argument('--green-width', type=int, default=4,
                        choices=[2, 3, 4, 5, 6],
                        help='Schaerfe der Gruen-Zone in Signed-Darstellung. '
                             'k=2: breit (Original), k=4: Kompromiss (default), '
                             'k=6: maximale Sensitivitaet')
    args = parser.parse_args()

    ROOT = Path(args.root)
    OUT = Path(args.outdir) if args.outdir else Path(__file__).parent / 'validation'
    OUT.mkdir(parents=True, exist_ok=True)

    sbs_dir = ROOT / 'frames' / 'sbs'
    if not sbs_dir.exists():
        sbs_dir = ROOT / 'frames' / 'fSBS_3D'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Modelle laden
    print(f"Lade S²M²cinematic ({args.model_type})...")
    model_cin = load_cinematic_model(args.weights, device,
                                     model_type=args.model_type)

    orig_weights = args.original_weights
    if orig_weights is None:
        orig_weights = Path('F:/Tools/Creative_Cinematic_Stereographing/s2m2/'
                            'weights/pretrain_weights/CH384NTR3.pth')
    if Path(orig_weights).exists():
        print(f"Lade S²M² Original (positivity, {args.original_model_type})...")
        model_orig = load_original_model(orig_weights, device,
                                         model_type=args.original_model_type)
    else:
        print("Original-Gewichte nicht gefunden — nur cinematic-Ergebnisse")
        model_orig = None

    # Frames verarbeiten
    sbs_files = sorted(sbs_dir.glob('frame_*.jpg'))
    name_set = set(args.frames)
    samples = [f for f in sbs_files if f.stem in name_set]
    print(f"\n  {len(samples)} Frames zum Vergleich\n")

    for path in samples:
        stem = path.stem
        print(f"  ── {stem} ──")
        left, right = load_sbs_frame(path)
        if left is None:
            continue

        left_t = to_input(left).to(device)
        right_t = to_input(right).to(device)

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # Cinematic (bidirektional)
            t0 = time.time()
            disp_cin, occ_cin, conf_cin = model_cin(left_t, right_t)
            t_cin = time.time() - t0
            disp_cin = disp_cin.squeeze().cpu().numpy().astype(np.float32)

            # Original (positivity)
            if model_orig is not None:
                t0 = time.time()
                disp_orig, _, _ = model_orig(left_t, right_t)
                t_orig = time.time() - t0
                disp_orig = disp_orig.squeeze().cpu().numpy().astype(np.float32)
            else:
                disp_orig = None
                t_orig = 0

        # Statistik
        neg_pct = float((disp_cin < -0.5).sum() / disp_cin.size * 100)
        print(f"    Cinematic: P98={np.percentile(disp_cin, 98):.1f}px  "
              f"Min={disp_cin.min():.1f}px  NegPix={neg_pct:.1f}%  ({t_cin:.2f}s)")
        if disp_orig is not None:
            print(f"    Original:  P98={np.percentile(disp_orig, 98):.1f}px  "
                  f"Min={disp_orig.min():.1f}px  ({t_orig:.2f}s)")

        # Asymmetrie-Test
        W = disp_cin.shape[1]
        half = W // 2
        cin_L = np.median(np.abs(disp_cin[:, :half][np.abs(disp_cin[:, :half]) > 0.5]))
        cin_R = np.median(np.abs(disp_cin[:, half:][np.abs(disp_cin[:, half:]) > 0.5]))
        print(f"    Asymmetrie Cinematic: L={cin_L:.2f}  R={cin_R:.2f}  "
              f"Ratio={cin_L / max(cin_R, 0.01):.2f}")
        if disp_orig is not None:
            orig_L = np.median(disp_orig[:, :half][disp_orig[:, :half] > 0.5]) \
                if (disp_orig[:, :half] > 0.5).any() else 0
            orig_R = np.median(disp_orig[:, half:][disp_orig[:, half:] > 0.5]) \
                if (disp_orig[:, half:] > 0.5).any() else 0
            print(f"    Asymmetrie Original: L={orig_L:.2f}  R={orig_R:.2f}  "
                  f"Ratio={orig_L / max(orig_R, 0.01):.2f}")

        # Visualisierungen
        vis_cin = colormap(disp_cin)
        vis_cin_signed = colormap(disp_cin, signed=True,
                                  green_width=args.green_width)

        if disp_orig is not None:
            vis_orig = colormap(disp_orig)
            # 2×2 Vergleich
            H_v = vis_cin.shape[0]
            W_v = vis_cin.shape[1]
            panel = np.zeros((H_v * 2, W_v * 2, 3), dtype=np.uint8)
            panel[:H_v, :W_v] = vis_orig
            panel[:H_v, W_v:] = vis_cin
            panel[H_v:, :W_v] = vis_cin_signed
            # Overlay
            left_small = cv2.resize(left, (W_v, H_v))
            overlay = (left_small.astype(float) * 0.55 + vis_cin.astype(float) * 0.45)
            panel[H_v:, W_v:] = np.clip(overlay, 0, 255).astype(np.uint8)
            for txt, pos in [('Original (positivity)', (20, 40)),
                             ('Cinematic (bidirektional)', (W_v + 20, 40)),
                             (f'Cinematic SIGNED k={args.green_width} (rot=pos, blau=neg)', (20, H_v + 40)),
                             ('Cinematic Overlay', (W_v + 20, H_v + 40))]:
                cv2.putText(panel, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 2)
            Image.fromarray(panel).save(str(OUT / f'{stem}_comparison.jpg'), quality=95)
        else:
            Image.fromarray(vis_cin).save(str(OUT / f'{stem}_cinematic.jpg'), quality=95)
            Image.fromarray(vis_cin_signed).save(
                str(OUT / f'{stem}_signed.jpg'), quality=95)

        # NPZ speichern
        np.savez(str(OUT / f'{stem}_validation.npz'),
                 disp_cinematic=disp_cin,
                 disp_original=disp_orig if disp_orig is not None else np.array([]))

        print(f"    → {OUT / stem}_*")

    print(f"\n  Ergebnisse in: {OUT}")


if __name__ == '__main__':
    main()
