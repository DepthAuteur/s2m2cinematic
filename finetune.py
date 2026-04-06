#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S²M²cinematic Fine-Tuning
===========================

Fine-Tuning von S²M² für bidirektionale Disparitätsschätzung
in professioneller stereoskopischer Kinematografie.

Nutzung:
  python finetune.py --root G:\\CCS --model_type L --full_films SeaRex

Phase 1 (Epochen 1-5):   Encoder eingefroren, hohe Auflösung (1920×1056)
Phase 2 (Epochen 6-50):  Voller Encoder+Decoder, niedrigere Auflösung (960×512)
Phase 3 (--translucent): Nachtraining mit TranslucencyLoss (3-5 Epochen)

Voraussetzungen:
  - S²M² Quellcode in ../s2m2/src/
  - S²M² Gewichte (CH256NTR3.pth für L, CH384NTR3.pth für XL)
  - fSBS-Frames + S²M² positivity-Disparitäten (GT-Anker)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# S²M² Quellcode einbinden
_S2M2_SRC = Path(__file__).parent.parent / 's2m2' / 'src'
if _S2M2_SRC.exists():
    sys.path.insert(0, str(_S2M2_SRC))
from s2m2.core.model.s2m2 import S2M2

from dataset import StereoFilmDataset
from losses import S2M2CinematicLoss
from s2m2translucent_loss import S2M2TranslucentLoss


def load_s2m2_model(weights_path, model_type='L', device='cpu'):
    """
    S²M² laden und für bidirektionale Disparität modifizieren.

    Änderungen gegenüber Original:
      1. use_positivity=False (Triu-Maske im Optimal Transport entfernt)
      2. disp.clamp(min=0) wird in der forward()-Methode übersprungen

    model_type: S (128ch/1tr), M (192ch/2tr), L (256ch/3tr), XL (384ch/3tr)
    """
    model_config = {
        "S":  {"feature_channels": 128, "num_transformer": 1},
        "M":  {"feature_channels": 192, "num_transformer": 2},
        "L":  {"feature_channels": 256, "num_transformer": 3},
        "XL": {"feature_channels": 384, "num_transformer": 3},
    }
    if model_type not in model_config:
        raise ValueError(f"model_type muss S/M/L/XL sein, nicht '{model_type}'")

    cfg = model_config[model_type]
    model = S2M2(
        feature_channels=cfg['feature_channels'],
        dim_expansion=1,
        num_transformer=cfg['num_transformer'],
        use_positivity=False,      # ← KERNÄNDERUNG: bidirektional
        refine_iter=3,
    )

    # Gewichte laden
    checkpoint = torch.load(str(weights_path), map_location='cpu', weights_only=True)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.my_load_state_dict(state_dict)
    print(f"  S²M² {model_type} Gewichte geladen: {Path(weights_path).name}")
    print(f"  use_positivity=False (bidirektional)")

    # Modell-Statistik
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameter: {total / 1e6:.1f}M")

    return model.to(device)


def modify_forward_for_finetuning(model):
    """
    Monkey-patch der forward()-Methode um clamp(min=0) zu entfernen
    und die Okklusionsmaske bidirektional zu machen.

    Das ist nötig weil use_positivity=False nur die Triu-Maske in DispInit
    entfernt, aber die clamp()-Aufrufe in S2M2.forward() bleiben.
    """
    # Import einmalig, nicht bei jedem Forward-Call
    from s2m2.core.model.submodules import CostVolume

    def patched_forward(img0, img1):
        img0_nor, img1_nor = model.normalize_img(img0, img1)

        # CNN feature extraction
        feature_4x, feature_2x = model.extract_feature(img0_nor, img1_nor)
        feature0_2x, _ = feature_2x.chunk(2, dim=0)

        # Feature pyramid
        feature_py_4x, feature_py_8x, feature_py_16x, feature_py_32x = \
            model.feat_pyramid(feature_4x)

        # Multi-res Transformer
        feature_tr_4x = model.transformer(
            feature_py_4x, feature_py_8x, feature_py_16x, feature_py_32x)

        # Initial disparity/confidence/occlusion
        disp, conf, occ, cv = model.disp_init(feature_tr_4x)

        feature0_tr_4x, feature1_tr_4x = feature_tr_4x.chunk(2, dim=0)
        feature0_py_4x, feature1_py_4x = feature_py_4x.chunk(2, dim=0)

        # Global refinement — OHNE clamp(min=0)
        disp = model.global_refiner(
            feature0_tr_4x.contiguous(), disp.detach(), conf.detach())

        # Iterative local refinement
        feature0_fusion_4x = model.feat_fusion_layer(feature0_tr_4x, feature0_py_4x)
        ctx0 = model.ctx_feat(feature0_fusion_4x)
        hidden = torch.tanh(ctx0)

        b, c, h, w = feature0_fusion_4x.shape
        coords_4x = torch.arange(
            w, device=feature0_fusion_4x.device,
            dtype=feature0_fusion_4x.dtype)

        cv_fn = CostVolume(
            cv, coords_4x.reshape(1, 1, w, 1).repeat(b, h, 1, 1), radius=4)

        for itr in range(model.refine_iter):
            hidden, disp, conf, occ = model.refiner(
                hidden, ctx0, disp, conf, occ, cv_fn)

            # Bidirektionale Okklusionsmaske:
            # Original: occ_mask = (coords - disp) >= 0
            #   → nur für positive disp korrekt
            # Bidirektional: 0 <= (coords - disp) < w
            corr_x = coords_4x.reshape(1, 1, 1, -1) - disp
            occ_mask = (corr_x >= 0) & (corr_x < w)
            occ = occ * occ_mask

        # 4x upsampling
        upsample_mask = model.upsample_mask_4x_refine(hidden, feature0_2x)
        disp_up = model.upsample4x(disp * 4, upsample_mask)
        occ_up = model.upsample4x(occ, upsample_mask)
        conf_up = model.upsample4x(conf, upsample_mask)

        # Edge guided sharpen
        filter_weights = model.upsample_mask_1x(disp_up, img0_nor, feature0_2x)
        disp_up = model.upsample1x(disp_up, filter_weights)
        occ_up = model.upsample1x(occ_up, filter_weights)
        conf_up = model.upsample1x(conf_up, filter_weights)

        if model.output_upsample:
            disp_up = 2 * disp_up

        return disp_up, occ_up, conf_up

    model.forward = patched_forward
    print("  forward() gepatcht: clamp(min=0) entfernt, Okklusion bidirektional")
    return model


def freeze_encoder(model, freeze=True):
    """Encoder einfrieren/auftauen."""
    for name, param in model.named_parameters():
        if any(enc in name for enc in ['cnn_backbone', 'feat_pyramid', 'transformer']):
            param.requires_grad = not freeze
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    status = "eingefroren" if freeze else "aufgetaut"
    print(f"  Encoder {status} — {trainable / 1e6:.1f}M trainierbar")


def main():
    parser = argparse.ArgumentParser(description='S²M²cinematic Fine-Tuning')
    parser.add_argument('--root', type=str, default='G:\\CCS',
                        help='CCS Root-Verzeichnis')
    parser.add_argument('--model_type', type=str, default='L',
                        choices=['S', 'M', 'L', 'XL'],
                        help='S²M² Modellgröße (default: L, 181M Parameter)')
    parser.add_argument('--s2m2_weights', type=str, default=None,
                        help='Pfad zu Gewichtedatei (auto-detect wenn nicht angegeben)')
    parser.add_argument('--disparity_dir', type=str, default=None,
                        help='Verzeichnis mit S²M²-positivity NPZ')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximale Epochen (Early Stopping kann frueher abbrechen)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Epochen mit eingefrorenem Encoder')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early Stopping Patience (nur Phase 2)')
    parser.add_argument('--min_delta', type=float, default=0.0005,
                        help='Minimale Verbesserung fuer Early Stopping')
    parser.add_argument('--sample_per_film', type=int, default=2500)
    parser.add_argument('--full_films', type=str, nargs='*', default=[],
                        help='Filme die NICHT gesampelt werden (alle Frames). '
                             'Z.B. --full_films SeaRex')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_height', type=int, default=512,
                        help='Trainings-Bildhoehe Phase 2 / Encoder aufgetaut (muss durch 32 teilbar sein)')
    parser.add_argument('--image_width', type=int, default=960,
                        help='Trainings-Bildbreite Phase 2 / Encoder aufgetaut')
    parser.add_argument('--p1_image_height', type=int, default=1056,
                        help='Trainings-Bildhoehe Phase 1 / Encoder eingefroren (volle Aufloesung)')
    parser.add_argument('--p1_image_width', type=int, default=1920,
                        help='Trainings-Bildbreite Phase 1 / Encoder eingefroren (volle Aufloesung)')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_encoder', type=float, default=1e-6)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=10000,
                        help='Mid-Epoch-Checkpoint alle N Batches (0 = deaktiviert)')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Validierung alle N Epochen')
    parser.add_argument('--restart', action='store_true',
                        help='Training von vorne starten (Checkpoint ignorieren)')
    parser.add_argument('--translucent', action='store_true',
                        help='Phase 3: Nachtraining mit TranslucencyLoss '
                             '(startet vom letzten Checkpoint)')
    parser.add_argument('--lr_translucent', type=float, default=2.5e-6,
                        help='Decoder-LR fuer Phase 3 Translucent (default: 2.5e-6)')
    parser.add_argument('--lr_translucent_encoder', type=float, default=5e-7,
                        help='Encoder-LR fuer Phase 3 Translucent (nicht verwendet bei frozen encoder)')
    args = parser.parse_args()

    ROOT = Path(args.root)
    SAVE_DIR = Path(args.save_dir) if args.save_dir else Path(__file__).parent / 'checkpoints'
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Pfade
    sbs_dir = ROOT / 'frames' / 'sbs'
    if not sbs_dir.exists():
        sbs_dir = ROOT / 'frames' / 'fSBS_3D'
    disp_dir = Path(args.disparity_dir) if args.disparity_dir else \
        Path('F:/Tools/Creative_Cinematic_Stereographing/disparities')
    film_json = ROOT / 'film_structure.json'

    # S²M² Gewichte (Dateiname hängt vom model_type ab)
    _model_config = {"S": "CH128NTR1", "M": "CH192NTR2", "L": "CH256NTR3", "XL": "CH384NTR3"}
    _weights_stem = _model_config[args.model_type]
    weights_candidates = []
    if args.s2m2_weights:
        weights_candidates.append(Path(args.s2m2_weights))
    weights_candidates.extend([
        Path(f'F:/Tools/Creative_Cinematic_Stereographing/s2m2/weights/pretrain_weights/{_weights_stem}.pth'),
        ROOT / 's2m2' / 'weights' / 'pretrain_weights' / f'{_weights_stem}.pth',
    ])
    weights_path = None
    for c in weights_candidates:
        if c.exists():
            weights_path = c
            break
    if weights_path is None:
        print("FEHLER: CH384NTR3.pth nicht gefunden")
        sys.exit(1)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 60}")
    print(f"  S²M²cinematic Fine-Tuning")
    print(f"{'=' * 60}")
    print(f"  Device:     {device}")
    if device.type == 'cuda':
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  SBS:        {sbs_dir}")
    print(f"  Disparität: {disp_dir}")
    print(f"  Gewichte:   {weights_path} ({args.model_type})")
    print(f"  Epochen:    {args.epochs} ({args.warmup_epochs} Warmup)")
    print(f"  Phase 1:    {args.p1_image_width}×{args.p1_image_height} (Encoder eingefroren)")
    print(f"  Phase 2:    {args.image_width}×{args.image_height} (Encoder aufgetaut)")
    print(f"  Samples:    {args.sample_per_film}/Film"
          + (f" + full: {', '.join(args.full_films)}" if args.full_films else ""))
    print()

    # ── Modell laden und modifizieren ─────────────────────────────────────────
    model = load_s2m2_model(weights_path, model_type=args.model_type, device=device)
    model = modify_forward_for_finetuning(model)

    # Gradient Checkpointing: spart ~40% VRAM auf Kosten von ~20% Trainingszeit
    if hasattr(model, 'transformer'):
        for module in model.transformer.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
    torch.cuda.empty_cache()
    print(f"  VRAM nach Modell-Load: "
          f"{torch.cuda.memory_allocated() / 1e9:.1f} GB allocated, "
          f"{torch.cuda.memory_reserved() / 1e9:.1f} GB reserved")

    model.train()

    # ── Dataset mit Train/Val Split ───────────────────────────────────────────
    # Indices einmalig bestimmen, Datasets werden pro Phase mit passender
    # Auflösung neu erstellt (Phase 1: hohe Aufl., Phase 2: niedrigere)
    full_dataset = StereoFilmDataset(
        sbs_dir=sbs_dir,
        disparity_dir=disp_dir,
        film_structure_path=film_json,
        sample_per_film=args.sample_per_film,
        full_films=args.full_films,
        image_height=args.p1_image_height,
        image_width=args.p1_image_width,
        training=True,
    )

    # 90/10 Split — deterministisch, Val enthält Frames aus allen Filmen
    n_total = len(full_dataset)
    indices = list(range(n_total))
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    # Frame-Listen für Wiederverwendung bei Auflösungswechsel
    _train_frame_paths = [full_dataset.frames[i] for i in train_indices]
    _val_frame_paths = [full_dataset.frames[i] for i in val_indices]
    del full_dataset  # VRAM/RAM freigeben

    print(f"  Split: {n_train:,} Train / {n_val:,} Val")

    def _create_loaders(img_h, img_w):
        """Erstellt Train/Val DataLoader mit gegebener Auflösung."""
        t_ds = StereoFilmDataset(
            sbs_dir=sbs_dir, disparity_dir=disp_dir,
            film_structure_path=film_json,
            sample_per_film=None,
            image_height=img_h, image_width=img_w,
            training=True,
        )
        t_ds.frames = list(_train_frame_paths)

        v_ds = StereoFilmDataset(
            sbs_dir=sbs_dir, disparity_dir=disp_dir,
            film_structure_path=film_json,
            sample_per_film=None,
            image_height=img_h, image_width=img_w,
            training=False,
        )
        v_ds.frames = list(_val_frame_paths)

        t_loader = DataLoader(
            t_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True,
        )
        v_loader = DataLoader(
            v_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        print(f"  DataLoader: {len(t_loader)} Train / {len(v_loader)} Val "
              f"@ {img_w}×{img_h}")
        return t_loader, v_loader

    # Phase 1 Loader
    train_loader, val_loader = _create_loaders(
        args.p1_image_height, args.p1_image_width)

    # ── Loss + Optimizer ──────────────────────────────────────────────────────
    # SignMagnitude: aktiver Anreiz für korrekte Vorzeichen UND Magnitude via NCC.
    criterion_phase1 = S2M2CinematicLoss(w_photo=1.0, w_anchor=0.3, w_sign=1.0, w_smooth=0.1)
    criterion_phase2 = S2M2CinematicLoss(w_photo=1.0, w_anchor=0.2, w_sign=1.0, w_smooth=0.05)
    criterion_phase3 = S2M2TranslucentLoss(
        w_photo=1.0, w_anchor=0.2, w_sign=1.0, w_smooth=0.05, w_translucent=0.05,
        conf_threshold=0.4, mismatch_threshold=0.08,
        grad_threshold=2.0, boost_factor=1.5,
    )

    # Phase 1: nur Decoder
    freeze_encoder(model, freeze=True)
    decoder_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(decoder_params, lr=args.lr, weight_decay=1e-4)

    scaler = GradScaler()
    best_val_loss = float('inf')
    no_improve = 0

    # QS-Monitor für S²M²cinematic
    _qs_dir = Path(__file__).parent
    if str(_qs_dir) not in sys.path:
        sys.path.insert(0, str(_qs_dir))
    try:
        from qs_monitor import QSMonitor, qs_s2m2cinematic_epoch
        qs = QSMonitor(output_dir=SAVE_DIR, stage='s2m2cinematic')
    except ImportError:
        qs = None

    # ── Checkpoint Resume ─────────────────────────────────────────────────────
    start_epoch = 1
    ckpt_latest = SAVE_DIR / 'latest.pth'

    # Phase 3 mit --restart: lade s2m2cinematic_best.pth (nicht latest.pth!)
    if args.translucent and args.restart:
        ckpt_cinematic = SAVE_DIR / 's2m2cinematic_best.pth'
        if ckpt_cinematic.exists():
            print(f"\n  Phase 3 Neustart von {ckpt_cinematic}")
            ckpt = torch.load(str(ckpt_cinematic), map_location=device, weights_only=False)
            model.load_state_dict(ckpt['state_dict'])
            start_epoch = ckpt['epoch'] + 1
            print(f"  S²M²cinematic Gewichte geladen (Ep {ckpt['epoch']}), "
                  f"starte Phase 3 ab Epoche {start_epoch}")
        else:
            print(f"\n  FEHLER: {ckpt_cinematic} nicht gefunden!")
            print(f"  Bitte zuerst S²M²cinematic trainieren.")
            sys.exit(1)
    elif ckpt_latest.exists() and not args.restart:
        print(f"\n  Resume von {ckpt_latest}")
        ckpt = torch.load(str(ckpt_latest), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        if ckpt.get('mid_epoch', False):
            start_epoch = ckpt['epoch']
            print(f"  Mid-Epoch Resume: Ep {start_epoch} Batch {ckpt.get('batch', '?')} "
                  f"→ Epoche {start_epoch} wird wiederholt (Gewichte beibehalten)")
        else:
            start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        no_improve = ckpt.get('no_improve', 0)
        print(f"  Fortgesetzt ab Epoche {start_epoch}, Best Val: {best_val_loss:.4f}")

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    if args.translucent:
        print(f"  Phase 3: S²M²translucent Fine-Tuning (ab Epoche {start_epoch})")
        print(f"  TranslucencyLoss aktiv | Encoder EINGEFROREN | LR Decoder={args.lr_translucent}")
        # Reset Early Stopping für Phase 3
        best_val_loss = float('inf')
        no_improve = 0
        # Encoder einfrieren — nur Decoder trainieren
        # Schützt die globale Tiefenskala (P98) und Feature-Repräsentation
        freeze_encoder(model, freeze=True)
        decoder_params = [p for p in model.parameters() if p.requires_grad]
        trainable = sum(p.numel() for p in decoder_params)
        print(f"  Encoder eingefroren — {trainable/1e6:.1f}M Decoder-Parameter trainierbar")
        optimizer = torch.optim.AdamW(
            decoder_params, lr=args.lr_translucent, weight_decay=1e-4
        )
        # Scaler frisch (keine alten Scale-States von Phase 2)
        scaler = GradScaler()
        # DataLoader mit Phase-2-Auflösung (960×512 bleibt)
        if start_epoch > 1:
            del train_loader, val_loader
            torch.cuda.empty_cache()
            train_loader, val_loader = _create_loaders(
                args.image_height, args.image_width)
    elif start_epoch <= args.warmup_epochs:
        print(f"  Phase 1: Decoder Fine-Tuning (Epochen {start_epoch}-{args.warmup_epochs})")
    else:
        print(f"  Phase 2: Full Model Fine-Tuning (ab Epoche {start_epoch})")
    print(f"  Early Stopping: Patience={args.patience}, Min-Delta={args.min_delta}")
    print(f"{'─' * 60}")

    for epoch in range(start_epoch, args.epochs + 1):
        # Phase-Wechsel
        if epoch == args.warmup_epochs + 1:
            print(f"\n{'─' * 60}")
            print(f"  Phase 2: Full Model Fine-Tuning (Epochen {epoch}-{args.epochs})")
            print(f"  Auflösung: {args.p1_image_width}×{args.p1_image_height} → "
                  f"{args.image_width}×{args.image_height}")
            print(f"{'─' * 60}")
            freeze_encoder(model, freeze=False)

            # DataLoader mit Phase-2-Auflösung neu erstellen
            # Alte Loader freigeben, VRAM/RAM für Encoder-Backward nötig
            del train_loader, val_loader
            torch.cuda.empty_cache()
            train_loader, val_loader = _create_loaders(
                args.image_height, args.image_width)

            # Optimizer mit differentiellen LRs neu erstellen
            encoder_params = []
            decoder_params = []
            for name, param in model.named_parameters():
                if any(enc in name for enc in ['cnn_backbone', 'feat_pyramid', 'transformer']):
                    encoder_params.append(param)
                else:
                    decoder_params.append(param)
            optimizer = torch.optim.AdamW([
                {'params': encoder_params, 'lr': args.lr_encoder},
                {'params': decoder_params, 'lr': args.lr},
            ], weight_decay=1e-4)

        # Bei Resume in Phase 2: sicherstellen dass Loader die richtige Auflösung hat
        # NICHT bei Phase 3 (translucent) — dort wird der Encoder bewusst eingefroren
        if epoch == start_epoch and epoch > args.warmup_epochs and not args.translucent:
            del train_loader, val_loader
            torch.cuda.empty_cache()
            train_loader, val_loader = _create_loaders(
                args.image_height, args.image_width)
            freeze_encoder(model, freeze=False)
            encoder_params = []
            decoder_params = []
            for name, param in model.named_parameters():
                if any(enc in name for enc in ['cnn_backbone', 'feat_pyramid', 'transformer']):
                    encoder_params.append(param)
                else:
                    decoder_params.append(param)
            optimizer = torch.optim.AdamW([
                {'params': encoder_params, 'lr': args.lr_encoder},
                {'params': decoder_params, 'lr': args.lr},
            ], weight_decay=1e-4)

        if args.translucent:
            criterion = criterion_phase3
        elif epoch <= args.warmup_epochs:
            criterion = criterion_phase1
        else:
            criterion = criterion_phase2
        model.train()
        epoch_loss = 0.0
        epoch_parts = {'photo': 0, 'anchor': 0, 'sign': 0, 'smooth': 0}
        if args.translucent:
            epoch_parts['translucent'] = 0
        n_batches = 0

        with tqdm(train_loader, desc=f'Ep {epoch}/{args.epochs} [Train]') as pbar:
            for batch in pbar:
                left = batch['left'].to(device, non_blocking=True)
                right = batch['right'].to(device, non_blocking=True)
                left_u8 = batch['left_u8'].to(device, non_blocking=True)
                right_u8 = batch['right_u8'].to(device, non_blocking=True)
                gt_disp = batch['gt_disp'].to(device, non_blocking=True)
                gt_conf = batch['gt_conf'].to(device, non_blocking=True)
                gt_occ = batch['gt_occ'].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type='cuda', dtype=torch.float16):
                    # S²M² Forward (erwartet uint8 [B,3,H,W])
                    pred_disp, pred_occ, pred_conf = model(left_u8, right_u8)

                    # Loss
                    loss, parts = criterion(
                        pred_disp, left, right,
                        gt_disp, gt_conf, gt_occ)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                for k in epoch_parts:
                    epoch_parts[k] += parts[k]
                n_batches += 1

                # Mid-Epoch-Checkpoint
                if (args.checkpoint_interval > 0 and
                        n_batches % args.checkpoint_interval == 0):
                    mid_ckpt = {
                        'epoch': epoch,
                        'batch': n_batches,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'no_improve': no_improve,
                        'phase': "P3" if args.translucent else ("P1" if epoch <= args.warmup_epochs else "P2"),
                        'mid_epoch': True,
                    }
                    torch.save(mid_ckpt, str(SAVE_DIR / 'latest.pth'))
                    avg_so_far = epoch_loss / n_batches
                    pbar.write(f"  💾 Mid-Epoch Checkpoint: Ep {epoch}, "
                               f"Batch {n_batches}, Avg Loss {avg_so_far:.4f}")

                postfix_dict = {
                    'loss': f'{loss.item():.4f}',
                    'pho': f'{parts["photo"]:.4f}',
                    'anc': f'{parts["anchor"]:.4f}',
                    'sgn': f'{parts["sign"]:.4f}',
                }
                if args.translucent:
                    postfix_dict['trn'] = f'{parts["translucent"]:.4f}'
                pbar.set_postfix(postfix_dict)

        avg_train = epoch_loss / max(n_batches, 1)
        avg_parts = {k: v / max(n_batches, 1) for k, v in epoch_parts.items()}

        # ── Validierung ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_parts = {'photo': 0, 'anchor': 0, 'sign': 0, 'smooth': 0}
        if args.translucent:
            val_parts['translucent'] = 0
        n_val = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Ep {epoch}/{args.epochs} [Val]',
                              leave=False):
                left = batch['left'].to(device, non_blocking=True)
                right = batch['right'].to(device, non_blocking=True)
                left_u8 = batch['left_u8'].to(device, non_blocking=True)
                right_u8 = batch['right_u8'].to(device, non_blocking=True)
                gt_disp = batch['gt_disp'].to(device, non_blocking=True)
                gt_conf = batch['gt_conf'].to(device, non_blocking=True)
                gt_occ = batch['gt_occ'].to(device, non_blocking=True)

                with autocast(device_type='cuda', dtype=torch.float16):
                    pred_disp, pred_occ, pred_conf = model(left_u8, right_u8)
                    loss, parts = criterion(
                        pred_disp, left, right,
                        gt_disp, gt_conf, gt_occ)

                val_loss += loss.item()
                for k in val_parts:
                    val_parts[k] += parts[k]
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        avg_val_parts = {k: v / max(n_val, 1) for k, v in val_parts.items()}

        # ── Logging ───────────────────────────────────────────────────────────
        if args.translucent:
            phase = "P3"
        elif epoch <= args.warmup_epochs:
            phase = "P1"
        else:
            phase = "P2"
        improved = avg_val < (best_val_loss - args.min_delta)
        marker = " ★" if improved else ""

        log_line = (f"  [{phase}] Ep {epoch}: Train={avg_train:.4f}  Val={avg_val:.4f}  "
              f"Photo={avg_val_parts['photo']:.4f}  "
              f"Anchor={avg_val_parts['anchor']:.4f}  "
              f"Sign={avg_val_parts['sign']:.4f}  "
              f"Smooth={avg_val_parts['smooth']:.4f}")
        if args.translucent:
            log_line += f"  Trans={avg_val_parts['translucent']:.4f}"
        log_line += marker
        print(log_line)

        # CSV-Log
        log_path = SAVE_DIR / 'training_log.csv'
        if epoch == start_epoch:
            with open(str(log_path), 'a') as f:
                if log_path.stat().st_size == 0:
                    header = "epoch,phase,train_loss,val_loss,photo,anchor,sign,smooth"
                    if args.translucent:
                        header += ",translucent"
                    f.write(header + "\n")
        with open(str(log_path), 'a') as f:
            csv_line = (f"{epoch},{phase},{avg_train:.6f},{avg_val:.6f},"
                    f"{avg_val_parts['photo']:.6f},{avg_val_parts['anchor']:.6f},"
                    f"{avg_val_parts['sign']:.6f},{avg_val_parts['smooth']:.6f}")
            if args.translucent:
                csv_line += f",{avg_val_parts['translucent']:.6f}"
            f.write(csv_line + "\n")

        # QS-Monitoring
        if qs is not None:
            qs_s2m2cinematic_epoch(
                qs, epoch=epoch, phase=phase,
                train_loss=avg_train, val_loss=avg_val,
                parts=avg_val_parts,
            )

        # ── Referenz-Frame-Validierung ────────────────────────────────────────
        # 4 Referenz-Frames: Asymmetrie, NegPix, P98, Min — nach jeder Epoche
        _ref_frames = ['frame_043420', 'frame_050768', 'frame_068500', 'frame_1037074']
        _val_out = Path(__file__).parent / 'validation'
        _val_out.mkdir(parents=True, exist_ok=True)

        print(f"\n  {len(_ref_frames)} Frames zum Vergleich")
        model.eval()
        for _stem in _ref_frames:
            _sbs_path = sbs_dir / f'{_stem}.jpg'
            if not _sbs_path.exists():
                continue

            # SBS laden
            _sbs_img = cv2.imread(str(_sbs_path))
            if _sbs_img is None:
                continue
            _sbs_img = cv2.cvtColor(_sbs_img, cv2.COLOR_BGR2RGB)
            _h_sbs, _w_sbs = _sbs_img.shape[:2]
            _half_w = _w_sbs // 2
            _H_val, _W_val = args.image_height, args.image_width
            _left_np = cv2.resize(_sbs_img[:, :_half_w], (_W_val, _H_val))
            _right_np = cv2.resize(_sbs_img[:, _half_w:], (_W_val, _H_val))

            _left_t = torch.from_numpy(_left_np.transpose(2, 0, 1).copy()).unsqueeze(0).contiguous().to(device)
            _right_t = torch.from_numpy(_right_np.transpose(2, 0, 1).copy()).unsqueeze(0).contiguous().to(device)

            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                t0 = time.time()
                _pred, _, _ = model(_left_t, _right_t)
                _t_inf = time.time() - t0
            _disp = _pred.squeeze().cpu().numpy().astype(np.float32)

            # Statistik
            _neg_pct = float((_disp < -0.5).sum() / _disp.size * 100)
            _p98 = float(np.percentile(_disp, 98))
            _dmin = float(_disp.min())
            _W_d = _disp.shape[1]
            _half = _W_d // 2
            _L_med = np.median(np.abs(_disp[:, :_half][np.abs(_disp[:, :_half]) > 0.5]))
            _R_med = np.median(np.abs(_disp[:, _half:][np.abs(_disp[:, _half:]) > 0.5]))
            _asym = _L_med / max(_R_med, 0.01)

            # GT-Anker laden für Original-Vergleich
            _gt_npz = disp_dir / f'{_stem}.npz'
            if _gt_npz.exists():
                _gt_data = np.load(str(_gt_npz))
                _gt_disp = _gt_data['disparity'].squeeze()
                if _gt_disp.shape[0] != _H_val or _gt_disp.shape[1] != _W_val:
                    _gt_disp = cv2.resize(_gt_disp, (_W_val, _H_val), interpolation=cv2.INTER_NEAREST)
                _gt_p98 = float(np.percentile(_gt_disp, 98))
                _gt_L = np.median(_gt_disp[:, :_half][_gt_disp[:, :_half] > 0.5]) \
                    if (_gt_disp[:, :_half] > 0.5).any() else 0
                _gt_R = np.median(_gt_disp[:, _half:][_gt_disp[:, _half:] > 0.5]) \
                    if (_gt_disp[:, _half:] > 0.5).any() else 0
                _gt_asym = _gt_L / max(_gt_R, 0.01)
                print(f"  ── {_stem} ──")
                print(f"    Cinematic: P98={_p98:.1f}px  Min={_dmin:.1f}px  "
                      f"NegPix={_neg_pct:.1f}%  ({_t_inf:.2f}s)")
                print(f"    Original:  P98={_gt_p98:.1f}px  Min=0.0px  ({0:.2f}s)")
                print(f"    Asymmetrie Cinematic: L={_L_med:.2f}  R={_R_med:.2f}  "
                      f"Ratio={_asym:.2f}")
                print(f"    Asymmetrie Original: L={_gt_L:.2f}  R={_gt_R:.2f}  "
                      f"Ratio={_gt_asym:.2f}")
            else:
                print(f"  ── {_stem} ──")
                print(f"    Cinematic: P98={_p98:.1f}px  Min={_dmin:.1f}px  "
                      f"NegPix={_neg_pct:.1f}%  ({_t_inf:.2f}s)")
                print(f"    Asymmetrie: L={_L_med:.2f}  R={_R_med:.2f}  Ratio={_asym:.2f}")

            print(f"    → {_val_out / _stem}_*")

        print(f"  Ergebnisse in: {_val_out}\n")

        # ── Early Stopping ────────────────────────────────────────────────────
        if improved:
            best_val_loss = avg_val
            no_improve = 0
        else:
            no_improve += 1

        # Checkpoint
        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'no_improve': no_improve,
            'phase': phase,
        }
        torch.save(ckpt, str(SAVE_DIR / 'latest.pth'))
        if improved:
            best_name = 's2m2translucent_best.pth' if args.translucent else 's2m2cinematic_best.pth'
            torch.save(ckpt, str(SAVE_DIR / best_name))
            print(f"  ★ Neuer Best Val: {best_val_loss:.4f}")

        # Early Stop Check (nur in Phase 2 — Phase 1 immer durchlaufen)
        if epoch > args.warmup_epochs and no_improve >= args.patience:
            print(f"\n  ⏹ Early Stopping: {no_improve} Epochen ohne Verbesserung")
            print(f"    Best Val Loss: {best_val_loss:.4f}")
            break

    print(f"\n{'=' * 60}")
    print(f"  Fine-Tuning abgeschlossen (Epoche {epoch}/{args.epochs})")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {SAVE_DIR}")
    print(f"  Log: {SAVE_DIR / 'training_log.csv'}")
    print(f"{'=' * 60}")
    print(f"\n  Nächster Schritt:")
    if args.translucent:
        print(f"    python validate.py --root {args.root} "
              f"--weights {SAVE_DIR / 's2m2translucent_best.pth'}"
              f" --frames frame_043420 frame_050768 frame_068500 frame_1037074")
    else:
        print(f"    python validate.py --root {args.root} "
              f"--weights {SAVE_DIR / 's2m2cinematic_best.pth'}"
              f" --frames frame_043420 frame_050768 frame_068500 frame_1037074")


if __name__ == '__main__':
    main()
