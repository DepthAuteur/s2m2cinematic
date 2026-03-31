"""
S²M²cinematic Dataset
======================
Lädt SBS-Stereopaare für Self-Supervised Fine-Tuning.

Pro Sample:
  - left_img:  [3, H, W] float32 [0,1]  (linkes Auge)
  - right_img: [3, H, W] float32 [0,1]  (rechtes Auge)
  - gt_disp:   [1, H, W] float32 px     (S²M² positivity-Disparität, nur positiv)
  - gt_conf:   [1, H, W] float32 [0,1]  (S²M²-Konfidenz)
  - gt_occ:    [1, H, W] float32 [0,1]  (S²M²-Okklusion)

Die GT dient als Anker für positive Disparitäten.
Für negative Disparitäten liefert der Photometric Loss das Signal.
"""
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class StereoFilmDataset(Dataset):
    """
    Dataset für S²M²cinematic Fine-Tuning.

    Lädt SBS-Frames und die zugehörigen S²M²-positivity-Disparitäten.
    Unterstützt Stichproben-Modus (sample_per_film) für effizientes Training.
    """

    def __init__(self, sbs_dir, disparity_dir, film_structure_path=None,
                 sample_per_film=2500, full_films=None,
                 image_height=1056, image_width=1920,
                 training=True, seed=2026):
        """
        sbs_dir:            Verzeichnis mit SBS-Frames (frame_NNNNNN.jpg)
        disparity_dir:      Verzeichnis mit S²M²-NPZ (frame_NNNNNN.npz)
        film_structure_path: Pfad zu film_structure.json (für per-Film Sampling)
        sample_per_film:    Frames pro Film für Training (None = alle)
        full_films:         Liste von Filmnamen die NICHT gesampelt werden
        image_height/width: Zielauflösung (muss durch 32 teilbar sein)
        training:           True = Augmentation aktiv
        seed:               Random Seed für reproduzierbare Stichprobe
        """
        self.sbs_dir = Path(sbs_dir)
        self.disp_dir = Path(disparity_dir)
        self.H = image_height
        self.W = image_width
        self.training = training
        full_film_set = set(full_films) if full_films else set()

        # Alle verfügbaren Frames (SBS + Disparität müssen beide existieren)
        sbs_files = sorted(self.sbs_dir.glob('frame_*.jpg'))
        disp_stems = {f.stem for f in self.disp_dir.glob('frame_*.npz')}
        all_frames = [f for f in sbs_files if f.stem in disp_stems]

        if not all_frames:
            raise RuntimeError(f"Keine passenden SBS+Disparität-Paare gefunden.\n"
                               f"  SBS: {self.sbs_dir}\n  Disp: {self.disp_dir}")

        # Per-Film Stichprobe (gleichmäßig über alle Filme)
        if sample_per_film and film_structure_path and Path(film_structure_path).exists():
            films = json.loads(Path(film_structure_path).read_text(encoding='utf-8'))
            rng = np.random.RandomState(seed)
            sampled = []
            n_full = 0
            for film in films:
                film_frames = [f for f in all_frames
                               if film['start'] <= int(f.stem.split('_')[1]) <= film['end']]
                # full_films: alle Frames verwenden
                if film['name'] in full_film_set:
                    sampled.extend(film_frames)
                    n_full += len(film_frames)
                elif len(film_frames) <= sample_per_film:
                    sampled.extend(film_frames)
                else:
                    idx = rng.choice(len(film_frames), sample_per_film, replace=False)
                    sampled.extend([film_frames[i] for i in sorted(idx)])
            self.frames = sampled
            if n_full > 0:
                print(f"  StereoFilmDataset: {len(sampled):,} Frames "
                      f"({sample_per_film}/Film × {len(films) - len(full_film_set)} Filme "
                      f"+ {n_full:,} full_films)")
            else:
                print(f"  StereoFilmDataset: {len(sampled):,} Frames "
                      f"({sample_per_film}/Film × {len(films)} Filme)")
        else:
            self.frames = all_frames
            print(f"  StereoFilmDataset: {len(all_frames):,} Frames (alle)")

    def __len__(self):
        return len(self.frames)

    def _load_sbs(self, path):
        """SBS-Frame laden, in L/R splitten, auf Zielgröße bringen."""
        sbs = cv2.imread(str(path))
        if sbs is None:
            return None, None
        sbs = cv2.cvtColor(sbs, cv2.COLOR_BGR2RGB)
        h, w = sbs.shape[:2]
        half_w = w // 2
        left = sbs[:, :half_w]
        right = sbs[:, half_w:]
        # Auf Zielgröße
        left = cv2.resize(left, (self.W, self.H), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return left, right

    def _load_disp(self, stem):
        """S²M²-Disparität + Konfidenz + Okklusion laden."""
        npz_path = self.disp_dir / f'{stem}.npz'
        try:
            data = np.load(str(npz_path))
            disp = np.squeeze(data['disparity']).astype(np.float32)
            conf = np.squeeze(data['confidence']).astype(np.float32) if 'confidence' in data \
                else np.ones_like(disp)
            occ = np.squeeze(data['occlusion']).astype(np.float32) if 'occlusion' in data \
                else np.ones_like(disp)
            # Auf Zielgröße (NEAREST für Disparität, LINEAR für Konfidenz/Okklusion)
            if disp.shape != (self.H, self.W):
                # Disparität skalieren proportional zur Breitenänderung
                scale_w = self.W / disp.shape[1]
                disp = cv2.resize(disp, (self.W, self.H),
                                  interpolation=cv2.INTER_NEAREST) * scale_w
                conf = cv2.resize(conf, (self.W, self.H),
                                  interpolation=cv2.INTER_LINEAR)
                occ = cv2.resize(occ, (self.W, self.H),
                                 interpolation=cv2.INTER_LINEAR)
            return disp, conf, occ
        except Exception:
            return (np.zeros((self.H, self.W), dtype=np.float32),
                    np.ones((self.H, self.W), dtype=np.float32),
                    np.ones((self.H, self.W), dtype=np.float32))

    def __getitem__(self, idx):
        frame_path = self.frames[idx]
        stem = frame_path.stem

        left, right = self._load_sbs(frame_path)
        if left is None:
            # Fallback: schwarze Bilder
            left = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            right = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        disp, conf, occ = self._load_disp(stem)

        # Augmentation (nur Training)
        if self.training:
            # Vertikaler Flip (p=0.1) — horizontal Flip ist NICHT erlaubt,
            # da er die Disparitätsrichtung umkehrt
            if np.random.random() < 0.1:
                left = left[::-1].copy()
                right = right[::-1].copy()
                disp = disp[::-1].copy()
                conf = conf[::-1].copy()
                occ = occ[::-1].copy()

            # Color Jitter — SYNCHRON auf beide Augen (gleiche Belichtung)
            if np.random.random() < 0.3:
                b = float(np.random.uniform(0.85, 1.15))
                left = np.clip(left.astype(np.float32) * b, 0, 255).astype(np.uint8)
                right = np.clip(right.astype(np.float32) * b, 0, 255).astype(np.uint8)

        # Zu Tensoren: S²M² erwartet uint8 [B,3,H,W] als Input,
        # aber für den Photometric Loss brauchen wir float [0,1]
        left_f = left.astype(np.float32) / 255.0
        right_f = right.astype(np.float32) / 255.0

        left_t = torch.from_numpy(left_f.transpose(2, 0, 1).copy())    # [3,H,W]
        right_t = torch.from_numpy(right_f.transpose(2, 0, 1).copy())  # [3,H,W]

        # uint8 Tensoren für S²M² Forward
        left_u8 = torch.from_numpy(
            left.transpose(2, 0, 1).copy()).contiguous()                # [3,H,W] uint8
        right_u8 = torch.from_numpy(
            right.transpose(2, 0, 1).copy()).contiguous()               # [3,H,W] uint8

        disp_t = torch.from_numpy(disp.copy()).unsqueeze(0)             # [1,H,W]
        conf_t = torch.from_numpy(conf.copy()).unsqueeze(0)             # [1,H,W]
        occ_t = torch.from_numpy(occ.copy()).unsqueeze(0)               # [1,H,W]

        return {
            'left': left_t,         # [3,H,W] float32 [0,1] — für Photometric Loss
            'right': right_t,       # [3,H,W] float32 [0,1] — für Photometric Loss
            'left_u8': left_u8,     # [3,H,W] uint8 — für S²M² Forward
            'right_u8': right_u8,   # [3,H,W] uint8 — für S²M² Forward
            'gt_disp': disp_t,      # [1,H,W] float32 px — positivity GT-Anker
            'gt_conf': conf_t,      # [1,H,W] float32 — Trust-Gewicht
            'gt_occ': occ_t,        # [1,H,W] float32 — Okklusion
            'stem': stem,
        }
