#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S²M²cinematic Engine-Export
=============================

Exportiert das Fine-Tuned S²M²cinematic Modell als ONNX und baut
die TensorRT-Engine. Die resultierende Engine ist ein Drop-in-Replacement
für die S²M² positivity-Engine in der CCS-Pipeline.

Nutzung:
  python export_engine.py --weights checkpoints/s2m2cinematic_best.pth

Schritte:
  1. PyTorch → ONNX (mit use_positivity=False)
  2. AveragePool-Patch (für NVIDIA Blackwell sm_120)
  3. ONNX → TensorRT FP16 Engine

Die resultierende Engine hat identische I/O:
  Input:  input_left [1,3,1056,1920] uint8
          input_right [1,3,1056,1920] uint8
  Output: output_disp [1,1,1056,1920] float16
          output_occ  [1,1,1056,1920] float16
          output_conf [1,1,1056,1920] float16
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_S2M2_SRC = Path(__file__).parent.parent / 's2m2' / 'src'
if _S2M2_SRC.exists():
    sys.path.insert(0, str(_S2M2_SRC))
from s2m2.core.model.s2m2 import S2M2


def model_params(model_type: str) -> dict:
    """
    Gibt S²M²-Konstruktor-Parameter fuer L oder XL zurueck.
      L:  feature_channels=256, ~180.7M Parameter (see S²M² repo for filename)
      XL: feature_channels=384, ~406M Parameter  (see S²M² repo for filename)
    """
    if model_type.upper() == 'L':
        return dict(feature_channels=256, dim_expansion=1,
                    num_transformer=3, refine_iter=3)
    elif model_type.upper() == 'XL':
        return dict(feature_channels=384, dim_expansion=1,
                    num_transformer=3, refine_iter=3)
    else:
        raise ValueError(f"Unbekannter model_type '{model_type}' — verwende 'L' oder 'XL'")


def main():
    parser = argparse.ArgumentParser(description='S²M²cinematic Engine-Export')
    parser.add_argument('--weights', type=str, required=True,
                        help='Pfad zu s2m2cinematic_best.pth')
    parser.add_argument('--img_width', type=int, default=1920)
    parser.add_argument('--img_height', type=int, default=1056)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='L',
                        choices=['L', 'XL'],
                        help='S²M²cinematic Modell-Groesse: L (256ch, default) '
                             'oder XL (384ch)')
    args = parser.parse_args()

    OUT_DIR = Path(args.output_dir) if args.output_dir else Path(__file__).parent / 'engine'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    H, W = args.img_height, args.img_width

    print(f"\n{'=' * 60}")
    print(f"  S²M²cinematic Engine-Export")
    print(f"{'=' * 60}")

    # ── 1. Modell laden ──────────────────────────────────────────────────────
    print(f"\n  1. Modell laden: {args.weights} (model_type={args.model_type})")
    params = model_params(args.model_type)
    model = S2M2(**params, use_positivity=False)
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=True)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.my_load_state_dict(sd)
    model.eval().cpu()
    print(f"     OK — feature_channels={params['feature_channels']} "
          f"(Epoche {ckpt.get('epoch', '?')}, Loss {ckpt.get('best_loss', '?')})")

    # ── 2. ONNX-Export ───────────────────────────────────────────────────────
    onnx_path = OUT_DIR / f'S2M2cinematic_{args.model_type}_{W}_{H}.onnx'
    print(f"\n  2. ONNX-Export: {onnx_path.name}")

    left_dummy = torch.zeros(1, 3, H, W, dtype=torch.uint8)
    right_dummy = torch.zeros(1, 3, H, W, dtype=torch.uint8)

    torch.onnx.export(
        model,
        (left_dummy, right_dummy),
        str(onnx_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=False,  # Wichtig: wie beim Original
        input_names=['input_left', 'input_right'],
        output_names=['output_disp', 'output_occ', 'output_conf'],
        dynamic_axes=None,
    )
    print(f"     OK ({onnx_path.stat().st_size / 1024 / 1024:.0f} MB)")

    # Verifizierung: kein Trilu-Operator
    import onnx
    m = onnx.load(str(onnx_path), load_external_data=False)
    trilu_ops = [n.op_type for n in m.graph.node if 'Trilu' in n.op_type]
    if trilu_ops:
        print(f"     ⚠  WARNUNG: Trilu-Operator gefunden! use_positivity war nicht False?")
    else:
        print(f"     ✓ Kein Trilu-Operator (bidirektional bestätigt)")

    # ── 3. AveragePool-Patch ─────────────────────────────────────────────────
    patch_script = Path(__file__).parent.parent / 's2m2' / 'patch_onnx_avgpool.py'
    onnx_patched = OUT_DIR / f'S2M2cinematic_{args.model_type}_{W}_{H}_patched.onnx'

    if patch_script.exists():
        print(f"\n  3. AveragePool-Patch")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(patch_script), str(onnx_path), str(onnx_patched)],
            capture_output=True, text=True)
        if result.returncode == 0:
            print(f"     OK: {onnx_patched.name}")
        else:
            print(f"     FEHLER: {result.stderr[:200]}")
            onnx_patched = onnx_path
    else:
        print(f"\n  3. AveragePool-Patch übersprungen (patch_onnx_avgpool.py nicht gefunden)")
        onnx_patched = onnx_path

    # ── 4. TRT-Engine bauen ──────────────────────────────────────────────────
    engine_path = OUT_DIR / f'S2M2cinematic_{args.model_type}_{W}_{H}_fp16.engine'
    print(f"\n  4. TensorRT Engine: {engine_path.name}")

    try:
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with trt.Builder(TRT_LOGGER) as builder, \
             builder.create_network(0) as network, \
             trt.OnnxParser(network, TRT_LOGGER) as parser_trt, \
             builder.create_builder_config() as config:

            with open(str(onnx_patched), 'rb') as f:
                if not parser_trt.parse(f.read()):
                    for i in range(parser_trt.num_errors):
                        print(f"     ONNX-Error: {parser_trt.get_error(i)}")
                    return
            print(f"     ONNX geparst ({network.num_layers} Layer)")

            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * 1024 ** 3)
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

            # linalg_vector_norm_2 in FP32 (wie beim Original)
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                if 'linalg_vector_norm_2' in layer.name:
                    layer.precision = trt.DataType.FLOAT
                    layer.set_output_type(0, trt.DataType.FLOAT)

            print(f"     Kompiliere... (dauert ~15 Minuten)")
            t0 = time.time()
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                print(f"     FEHLER: Engine-Build fehlgeschlagen")
                return

            with open(str(engine_path), 'wb') as f:
                f.write(serialized)
            print(f"     OK: {engine_path.name} "
                  f"({engine_path.stat().st_size / 1024 / 1024:.0f} MB, "
                  f"{time.time() - t0:.0f}s)")

    except ImportError:
        print(f"     TensorRT nicht verfügbar — nur ONNX exportiert")

    # ── Zusammenfassung ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Export abgeschlossen")
    print(f"  ONNX:   {onnx_patched}")
    print(f"  Engine: {engine_path if engine_path.exists() else 'nicht gebaut'}")
    print(f"{'=' * 60}")
    print(f"\n  Nächster Schritt:")
    print(f"  1. Engine in S²M² weights Verzeichnis kopieren:")
    print(f"     cp {engine_path} "
          f"<root>/s2m2/weights/trt_save/S2M2_XL_{W}_{H}_fp16.engine")
    print(f"  2. Stage 2 in der CCS-Pipeline neu starten")


if __name__ == '__main__':
    main()
