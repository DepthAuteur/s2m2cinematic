#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S²M²cinematic GT-Vorbereitung
================================

Erzeugt S²M²-positivity-Disparitäten als GT-Anker für das Fine-Tuning.
Statt sequentiell ~1M Frames zu verarbeiten, werden gezielt Stichproben
aus allen 22 Filmen extrahiert — gleichmäßig verteilt für maximale Diversität.

Nutzung:
  python prepare_gt.py --root G:\\CCS --sample_per_film 2500

Ergebnis: ~55.000 NPZ-Dateien mit disparity + confidence + occlusion
Dauer:    ~6 Stunden bei 2.7 fps auf RTX 5090

Voraussetzungen:
  - S²M² positivity TRT-Engine
  - Extrahierte fSBS-Frames
  - film_structure.json
"""
import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# TRT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class _CudaRT:
    H2D = 1; D2H = 2
    def __init__(self):
        for c in ['cudart64_12',
                   r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/cudart64_12.dll',
                   r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/cudart64_12.dll',
                   r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/cudart64_12.dll']:
            try: self.lib = ctypes.WinDLL(c); break
            except OSError: continue
        else: raise RuntimeError("cudart64_12.dll nicht gefunden")
        def _s(fn, *a): fn.restype = ctypes.c_int; fn.argtypes = list(a)
        _s(self.lib.cudaMalloc, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t)
        _s(self.lib.cudaFree, ctypes.c_void_p)
        _s(self.lib.cudaHostAlloc, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint)
        _s(self.lib.cudaFreeHost, ctypes.c_void_p)
        _s(self.lib.cudaMemcpyAsync, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)
        _s(self.lib.cudaStreamCreate, ctypes.POINTER(ctypes.c_void_p))
        _s(self.lib.cudaStreamSynchronize, ctypes.c_void_p)
        _s(self.lib.cudaStreamDestroy, ctypes.c_void_p)
    def malloc(self, n):
        p = ctypes.c_void_p(); assert self.lib.cudaMalloc(ctypes.byref(p), n) == 0; return p.value
    def host_alloc(self, n):
        p = ctypes.c_void_p(); assert self.lib.cudaHostAlloc(ctypes.byref(p), n, 0) == 0; return p.value
    def free(self, p): self.lib.cudaFree(ctypes.c_void_p(p))
    def free_host(self, p): self.lib.cudaFreeHost(ctypes.c_void_p(p))
    def h2d(self, dst, src_np, stream):
        src = np.ascontiguousarray(src_np)
        assert self.lib.cudaMemcpyAsync(ctypes.c_void_p(dst), src.ctypes.data_as(ctypes.c_void_p),
                                         src.nbytes, self.H2D, ctypes.c_void_p(stream)) == 0
    def d2h(self, dst_np, src, stream):
        assert self.lib.cudaMemcpyAsync(dst_np.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(src),
                                         dst_np.nbytes, self.D2H, ctypes.c_void_p(stream)) == 0
    def stream_create(self):
        p = ctypes.c_void_p(); assert self.lib.cudaStreamCreate(ctypes.byref(p)) == 0; return p.value
    def sync(self, s): self.lib.cudaStreamSynchronize(ctypes.c_void_p(s))
    def stream_destroy(self, s): self.lib.cudaStreamDestroy(ctypes.c_void_p(s))


# ══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ══════════════════════════════════════════════════════════════════════════════

def load_sbs_frame(path):
    """Laedt SBS-JPEG, splittet in L/R, gibt uint8 [1,3,H,W] zurueck."""
    sbs = cv2.imread(str(path))
    if sbs is None:
        return None, None
    h, w = sbs.shape[:2]
    half_w = w // 2
    h32 = (h // 32) * 32
    w32 = (half_w // 32) * 32
    left_rgb = cv2.cvtColor(sbs[:h32, :w32], cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(sbs[:h32, half_w:half_w + w32], cv2.COLOR_BGR2RGB)
    left_np = np.ascontiguousarray(left_rgb.transpose(2, 0, 1)[np.newaxis])
    right_np = np.ascontiguousarray(right_rgb.transpose(2, 0, 1)[np.newaxis])
    return left_np, right_np


def save_npz(out_path, disp, conf, occ):
    """Speichert Disparitaet + Konfidenz + Okklusion als NPZ."""
    # np.savez hängt automatisch .npz an den Dateinamen an.
    # Daher tmp-Datei OHNE .npz-Extension, dann wird .npz angehängt.
    tmp = out_path.with_suffix('.tmp')
    np.savez(str(tmp),
             disparity=disp,
             confidence=conf.astype(np.float16),
             occlusion=occ.astype(np.float16))
    # np.savez hat .npz angehängt → tmp heißt jetzt .tmp.npz
    tmp_actual = tmp.with_suffix('.tmp.npz')
    if tmp_actual.exists():
        tmp_actual.replace(out_path)
    elif tmp.with_suffix('.npz').exists():
        tmp.with_suffix('.npz').replace(out_path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='S²M²cinematic GT-Vorbereitung')
    parser.add_argument('--root', type=str, default='G:\\CCS')
    parser.add_argument('--sample_per_film', type=int, default=2500,
                        help='Frames pro Film (default: 2500)')
    parser.add_argument('--engine', type=str, default=None,
                        help='Pfad zur S²M² positivity TRT-Engine')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Ausgabe-Verzeichnis fuer NPZ-Dateien')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--reuse_existing', action='store_true',
                        help='Bereits vorhandene NPZ-Dateien ueberspringen')
    parser.add_argument('--full_films', type=str, nargs='*', default=[],
                        help='Filme die NICHT gesampelt werden (alle Frames). '
                             'Z.B. --full_films MyFilm AnotherFilm')
    args = parser.parse_args()

    ROOT = Path(args.root)
    OUT_DIR = Path(args.output_dir) if args.output_dir else \
        Path(__file__).parent / 'gt_anchor'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # SBS-Frames
    sbs_dir = ROOT / 'frames' / 'sbs'
    if not sbs_dir.exists():
        sbs_dir = ROOT / 'frames' / 'fSBS_3D'
    if not sbs_dir.exists():
        print(f"FEHLER: SBS-Verzeichnis nicht gefunden"); sys.exit(1)

    sbs_files = sorted(sbs_dir.glob('frame_*.jpg'))
    print(f"  SBS-Frames gesamt: {len(sbs_files):,}")

    # Film-Struktur laden
    film_json = ROOT / 'film_structure.json'
    if not film_json.exists():
        print(f"FEHLER: {film_json} nicht gefunden"); sys.exit(1)
    films = json.loads(film_json.read_text(encoding='utf-8'))
    print(f"  Filme: {len(films)}")

    # Pro Film gleichmäßig verteilte Stichprobe
    rng = np.random.RandomState(args.seed)
    full_film_set = set(args.full_films)
    selected = []
    for film in films:
        # Frames dieses Films
        film_frames = [f for f in sbs_files
                       if film['start'] <= int(f.stem.split('_')[1]) <= film['end']]
        if not film_frames:
            print(f"  ⚠  {film['name']}: keine Frames gefunden")
            continue

        # Filme in --full_films: alle Frames verwenden (kein Sampling)
        if film['name'] in full_film_set:
            sampled = film_frames
            print(f"  [{film['nr']:02d}] {film['name']:35s}  "
                  f"{len(film_frames):>6,} Frames → {len(sampled):>5,} ALLE (full_films)")
            selected.extend(sampled)
            continue

        n_sample = min(args.sample_per_film, len(film_frames))
        if n_sample == len(film_frames):
            sampled = film_frames
        else:
            # Gleichmäßig verteilt, nicht rein zufällig — erfasst alle Szenentypen
            step = len(film_frames) / n_sample
            base_indices = [int(i * step) for i in range(n_sample)]
            # Leichte Jitter-Verschiebung für Diversität (+/- 20% des Abstands)
            jitter = int(step * 0.2)
            if jitter > 0:
                offsets = rng.randint(-jitter, jitter + 1, size=n_sample)
                indices = np.clip(np.array(base_indices) + offsets,
                                  0, len(film_frames) - 1)
            else:
                indices = np.array(base_indices)
            indices = sorted(set(indices))  # Duplikate entfernen
            sampled = [film_frames[i] for i in indices]

        selected.extend(sampled)
        print(f"  [{film['nr']:02d}] {film['name']:35s}  "
              f"{len(film_frames):>6,} Frames → {len(sampled):>5,} Stichprobe")

    # Bereits vorhandene überspringen
    if args.reuse_existing:
        existing_stems = {f.stem for f in OUT_DIR.glob('frame_*.npz')}
        before = len(selected)
        selected = [f for f in selected if f.stem not in existing_stems]
        skipped = before - len(selected)
        if skipped > 0:
            print(f"\n  {skipped:,} bereits vorhanden → übersprungen")

    # Auch bereits aus Stage 2 vorhandene NPZ-Dateien wiederverwenden
    existing_disp_dir = ROOT / 'disparities'
    reused = 0
    if existing_disp_dir.exists():
        to_process = []
        for f in selected:
            existing_npz = existing_disp_dir / f'{f.stem}.npz'
            if existing_npz.exists():
                # Prüfen ob Konfidenz + Okklusion enthalten sind
                try:
                    data = np.load(str(existing_npz))
                    if 'confidence' in data and 'occlusion' in data:
                        # Kopieren statt neu berechnen
                        out_path = OUT_DIR / f'{f.stem}.npz'
                        if not out_path.exists():
                            import shutil
                            shutil.copy2(str(existing_npz), str(out_path))
                        reused += 1
                        continue
                except Exception:
                    pass
            to_process.append(f)
        if reused > 0:
            print(f"  {reused:,} aus Stage 2 übernommen (mit Konfidenz+Okklusion)")
        selected = to_process

    total = len(selected)
    print(f"\n  Zu verarbeiten: {total:,} Frames")
    if total == 0:
        print("  Nichts zu tun.")
        return

    est_hours = total / 2.7 / 3600
    print(f"  Geschätzte Dauer: {est_hours:.1f}h ({est_hours / 24:.1f} Tage)")

    # ── Engine laden ──────────────────────────────────────────────────────────
    import tensorrt as trt

    engine_candidates = []
    if args.engine:
        engine_candidates.append(Path(args.engine))
    engine_candidates.extend([
        ROOT / 's2m2' / 'weights' / 'trt_save' / 'S2M2_XL_1920_1056_fp16.engine',
        ROOT / 's2m2' / 'weights' / 'trt_save' / 'S2M2_XL_1920_1056_fp16_positivity_backup.engine',
    ])
    engine_path = None
    for c in engine_candidates:
        if c.exists():
            engine_path = c
            break
    if engine_path is None:
        print("FEHLER: S²M² TRT Engine nicht gefunden"); sys.exit(1)

    print(f"\n  Engine: {engine_path.name}")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(str(engine_path), 'rb') as f:
        engine_data = f.read()
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    def _np_dtype(td):
        return {trt.DataType.FLOAT: np.float32, trt.DataType.HALF: np.float16,
                trt.DataType.UINT8: np.uint8, trt.DataType.INT8: np.int8}.get(td, np.float32)

    io = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = _np_dtype(engine.get_tensor_dtype(name))
        nb = int(np.prod(shape)) * np.dtype(dtype).itemsize
        mode = 'in' if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else 'out'
        io[name] = dict(shape=shape, dtype=dtype, nb=nb, mode=mode)

    cuda = _CudaRT()
    stream = cuda.stream_create()
    gpu = {n: cuda.malloc(v['nb']) for n, v in io.items()}
    for n, ptr in gpu.items():
        context.set_tensor_address(n, ptr)

    # Warmup
    dummy = np.zeros(io['input_left']['shape'], dtype=np.uint8)
    for _ in range(2):
        cuda.h2d(gpu['input_left'], dummy, stream)
        cuda.h2d(gpu['input_right'], dummy, stream)
        context.execute_async_v3(stream)
        cuda.sync(stream)
    print("  Warmup OK\n")

    # ── Inferenz ──────────────────────────────────────────────────────────────
    prefetch = ThreadPoolExecutor(max_workers=2)
    save_pool = ThreadPoolExecutor(max_workers=3)

    t_start = time.time()
    processed = 0
    errors = 0

    # Erstes Frame vorladen
    future_load = prefetch.submit(load_sbs_frame, selected[0])

    with tqdm(total=total, desc='  GT-Anker') as pbar:
        for i, frame_path in enumerate(selected):
            left_np, right_np = future_load.result()

            # Nächstes vorladen
            if i + 1 < total:
                future_load = prefetch.submit(load_sbs_frame, selected[i + 1])

            if left_np is None:
                errors += 1
                pbar.update(1)
                continue

            # S²M² Inferenz
            cuda.h2d(gpu['input_left'], left_np, stream)
            cuda.h2d(gpu['input_right'], right_np, stream)
            context.execute_async_v3(stream)

            disp_out = np.empty(io['output_disp']['shape'], dtype=io['output_disp']['dtype'])
            occ_out = np.empty(io['output_occ']['shape'], dtype=io['output_occ']['dtype'])
            conf_out = np.empty(io['output_conf']['shape'], dtype=io['output_conf']['dtype'])
            cuda.d2h(disp_out, gpu['output_disp'], stream)
            cuda.d2h(occ_out, gpu['output_occ'], stream)
            cuda.d2h(conf_out, gpu['output_conf'], stream)
            cuda.sync(stream)

            disp = np.squeeze(disp_out).astype(np.float32)
            conf = np.squeeze(conf_out).astype(np.float32)
            occ = np.squeeze(occ_out).astype(np.float32)

            # Speichern
            out_path = OUT_DIR / f'{frame_path.stem}.npz'
            save_pool.submit(save_npz, out_path, disp, conf, occ)

            processed += 1
            pbar.update(1)

            # Periodische Statistik
            if processed % 1000 == 0 and processed > 0:
                elapsed = time.time() - t_start
                fps = processed / elapsed
                remaining = (total - processed) / fps / 3600
                pbar.write(f"  [{processed:,}/{total:,}] {fps:.1f} fps  "
                           f"ETA: {remaining:.1f}h")

    save_pool.shutdown(wait=True)
    prefetch.shutdown(wait=True)

    # Cleanup
    for ptr in gpu.values():
        cuda.free(ptr)
    cuda.stream_destroy(stream)

    elapsed = time.time() - t_start
    fps = processed / elapsed if elapsed > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  GT-Anker Erzeugung abgeschlossen")
    print(f"  Verarbeitet: {processed:,} Frames in {elapsed / 3600:.1f}h ({fps:.1f} fps)")
    if errors > 0:
        print(f"  Fehler: {errors}")
    if reused > 0:
        print(f"  Aus Stage 2: {reused:,} übernommen")
    print(f"  Gesamt: {processed + reused:,} NPZ-Dateien in {OUT_DIR}")
    print(f"{'=' * 60}")
    print(f"\n  Nächster Schritt:")
    print(f"    python finetune.py --root {args.root} "
          f"--disparity_dir {OUT_DIR} --epochs 15")


if __name__ == '__main__':
    main()
