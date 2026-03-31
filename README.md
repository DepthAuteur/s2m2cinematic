# S²M²cinematic

### Bidirectional Stereo Matching for Stereoscopic Source Material

> **First stereo matching model to predict negative disparities** from
> stereoscopic source material pairs — enabling accurate Ground Truth generation
> for objects placed in front of the screen plane (pop-out / negative parallax effects).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen)]()

---

## The Problem

Every stereo matching model in the literature — S²M², FoundationStereo,
RAFT-Stereo, and all others — predicts only **positive disparities**. This is
correct for automotive and robotics, where objects never appear in front of the
camera baseline.

In stereoscopic source material, this assumption fails completely.

Depth compositing artists deliberately place objects **in front of the convergence plane**
(the screen surface) to create pop-out effects. These objects produce
**negative pixel shifts** between the left and right eye views. A stereo
matching engine that clips negative disparities to zero silently discards
this information — producing systematically wrong Ground Truth for any
scene with deliberate pop-out composition.

```
Screen plane (zero disparity)      ─────────────────────────
Objects behind screen (positive)   ─────────────── ←→ positive shift
Objects in front of screen (pop)   ←─────────────────────────→ negative shift
                                        (clipped to 0 by all existing models)
```

---

## The Solution

Three minimal changes to S²M² unlock bidirectional matching:

```python
# 1. Remove Optimal Transport positivity mask
use_positivity = False        # was: True  (torch.triu() in OT solver)

# 2. Remove output clamp (monkey-patched in forward())
# was: disp = disp.clamp(min=0)
# now: unclamped — negative disparities flow through

# 3. Bidirectional occlusion mask
# was: corr_x >= 0
# now: (corr_x >= 0) & (corr_x < w)
```

These changes are necessary but not sufficient. Without fine-tuning, the base
model produces noisy, unphysical negative values. **SignMagnitudeLoss** trains
the model to produce negative disparities that correspond to actual physical
pop-out geometry, as measured by NCC block-matching.

---

## Key Innovation: SignMagnitudeLoss

```
SignMagnitudeLoss = 0.30 × L_hinge(sign) + 0.70 × L_regression(magnitude)
                 + w_photo × L_photometric
                 + w_anchor × L_gt_anchor
                 + w_smooth × L_smoothness
```

| Component | Weight | Purpose |
|---|---|---|
| **Hinge (sign)** | 30% | Penalizes wrong sign — ensures pop-out regions go negative |
| **Regression (magnitude)** | 70% | Pulls toward NCC-measured disparity magnitude |
| Photometric | — | Left-right reconstruction consistency |
| GT Anchor | — | Prevents collapse of positive disparity quality |
| Smoothness | — | Spatial regularization |

**Why magnitude regression matters:** A pure sign loss teaches the model
*that* pop-out regions should be negative, but not *how negative*. The 70%
magnitude regression component drives the prediction toward the physically
measured depth of each pop-out element — not just its direction.

The NCC block-matcher provides the regression target: actual pixel displacement
between left and right eye views, without semantic priors, at 1/2 resolution.

---

## Results — Epoch 1, Phase 1 (Encoder Frozen)

Validation on four reference frames after one complete training epoch,
encoder frozen, trained on ~103K GT-anchor frames from 23 original stereoscopic
source materials spanning 1953–2018.

| Frame | Scene Type | Original S²M² | S²M²cinematic | NegPix | Asym. Ratio |
|---|---|---|---|---|---|
| Ref-A | Dramatic work, strong positive parallax | 0.0 px | **−24.5 px** | 7.4% | 0.87 |
| Ref-B | — | 0.0 px | **−17.8 px** | 17.2% | 2.66 |
| Ref-C | — | 0.2 px | **−15.2 px** | **52.4%** | 0.57 |
| Ref-D | Large-format work, strong pop-out | 0.0 px | **−23.3 px** | 13.0% | **0.90** |

**Column notes:**
- **Original S²M²**: always ≤ 0.2px — the positivity constraint makes negative values structurally impossible
- **S²M²cinematic**: negative values = objects in front of the screen plane
- **NegPix**: fraction of pixels with negative disparity in this frame
- **Asymmetry Ratio**: left/right disparity balance (1.0 = perfectly symmetric). Original S²M²: 0.09 and 13.05 (severely skewed). S²M²cinematic: 0.57–2.66 (genuinely bidirectional)

**Training loss after Epoch 1:**
```
Train = 8.5214    Val = 8.0864    (Val < Train → no overfitting)
```

Training continues through Phase 2 (unfrozen encoder). Final weights will be
stronger — the proof of concept is established after Epoch 1.

---

## Architecture

```
Input: Full-SBS stereo pair (left + right, side-by-side)
         │
         ▼
  S²M² L backbone (180.7M parameters, pretrained weights from S²M² repository)
  ┌─────────────────────────────────────────────────────┐
  │  use_positivity = False   ← OT positivity mask off  │
  │  clamp(min=0) removed     ← negative output allowed │
  │  bidirectional occlusion  ← symmetric search        │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  SignMagnitudeLoss fine-tuning on stereoscopic source GT
  ┌─────────────────────────────────────────────────────┐
  │  NCC block-matching  → signed magnitude target       │
  │  Hinge loss (30%)    → correct sign enforcement      │
  │  Regression (70%)    → correct magnitude             │
  │  GT Anchor loss      → preserves positive quality    │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  Signed disparity map [H, W]
     > 0  →  object behind screen (standard depth)
     = 0  →  convergence plane (screen surface)
     < 0  →  object in front of screen (pop-out)
```

### Training Phases

| Phase | Epochs | Encoder | Resolution | Trainable Params |
|---|---|---|---|---|
| Phase 1 | 1–5 | Frozen | 1920×1056 | 44.2M |
| Phase 2 | 6–50 | Unfrozen | 960×512 | 180.7M |
| Phase 3 (optional) | 3–5 | Unfrozen | 960×512 | 180.7M |

Early stopping in Phase 2: patience=7, min-delta=0.0005.

---

## Repository Contents

```
s2m2cinematic/
├── finetune.py              Main fine-tuning script (two-phase training)
├── losses.py                SignMagnitudeLoss + PhotometricLoss + GTAnchorLoss
├── dataset.py               StereoDataset (Full-SBS stereo source pairs)
├── quicktest.py             100-step proof-of-concept test on single frames
├── validate.py              Comparison: cinematic vs. original on reference frames
├── prepare_gt.py            Sample GT-anchor disparities from stereoscopic source corpus
├── export_engine.py         ONNX graph surgery + TRT FP16 export
├── translucency_loss.py     TranslucencyLoss for semi-transparent surfaces
└── s2m2translucent_loss.py  Combined loss with translucency awareness
```

**Not included:** Pre-trained weights, training frames, GT disparity files.
See [Note on Weights and Data](#note-on-weights-and-data).

---

## Requirements

- Python 3.10+
- PyTorch 2.8+ with CUDA 12.x
- [S²M² codebase](https://github.com/junhong-3dv/s2m2)
- NVIDIA GPU ≥ 24 GB VRAM (tested on Blackwell-architecture GPU)
- Full-SBS stereoscopic source material (your own collection)

For TRT export on NVIDIA Blackwell (sm_120 / RTX 5090), `export_engine.py`
includes an automatic ONNX graph surgery workaround for 25 AveragePool nodes
incompatible with the sm_120 architecture.

---

## Quick Start

### 1. Prepare GT anchors

```bash
python prepare_gt.py \
    --root /path/to/your/data \
    --disparity_dir /path/to/your/data/gt_anchor \
    --n_per_source 2500
```

### 2. Quick proof-of-concept (~5 minutes)

```bash
python quicktest.py \
    --root /path/to/your/data \
    --disparity_dir /path/to/your/data/gt_anchor \
    --frames frame_001000 \
    --steps 100
```

Expected: negative Min disparity on frames with genuine pop-out content.

### 3. Full fine-tuning

```bash
python finetune.py \
    --root /path/to/your/data \
    --disparity_dir /path/to/your/data/gt_anchor \
    --model_type L \
    --p1_image_height 1056 \
    --p1_image_width 1920 \
    --image_height 512 \
    --image_width 960 \
    --restart
```

### 4. Validate

```bash
python validate.py \
    --root /path/to/your/data \
    --weights checkpoints/s2m2cinematic_best.pth \
    --frames frame_001000 frame_005000 frame_010000
```

### 5. Export to TensorRT FP16

```bash
python export_engine.py \
    --weights checkpoints/s2m2cinematic_best.pth
```

---

## Note on Weights and Data

Pre-trained weights are **not included**.

The model was fine-tuned on original stereoscopic source material
(Full-SBS format). The legal status of using such material as AI training
data under German copyright law (§44b UrhG) and EU DSM Directive Art. 4
is subject to ongoing review.

You can train your own model using Full-SBS stereoscopic source material
from your personal collection. No source-specific data is included in this repository.

---

## Comparison with Existing Models

| Model | Negative Disparities | Physical Measurement | Stereo Domain |
|---|---|---|---|
| S²M² (original) | ❌ clipped to 0 | ✅ stereo matching | ❌ |
| FoundationStereo | ❌ clipped to 0 | ⚠️ semantic priors | ❌ |
| RAFT-Stereo | ❌ clipped to 0 | ✅ stereo matching | ❌ |
| MiDaS / DPT | ❌ affine-invariant | ❌ pseudo-labels | ❌ |
| Depth Anything | ❌ affine-invariant | ❌ pseudo-labels | ❌ |
| **S²M²cinematic** | ✅ **bidirectional** | ✅ NCC-anchored | ✅ |

---

## Relation to CCS

S²M²cinematic is the Ground Truth engine for the
[Creative Cinematic Stereographing (CCS)](https://github.com/DepthAuteur/creative-cinematic-stereographing)
project, which trains a monocular depth model for 2D-to-3D conversion
on physically measured, bidirectional disparities from original stereoscopic source material.

The key insight: any GT engine that clips negative disparities to zero produces
systematically wrong Ground Truth for stereoscopic source material. Pop-out elements —
the most deliberate depth decisions in a depth compositing artist's work — are silently
converted to zero disparity. A depth model trained on this GT learns an
incomplete picture of compositional depth.

---

## Citation

```bibtex
@misc{artoist2026s2m2cinematic,
  author       = {DepthAuteur},
  title        = {S2M2cinematic: Bidirectional Stereo Matching
                  for Stereoscopic Source Material},
  year         = {2026},
  month        = {March},
  address      = {},
  howpublished = {\url{https://github.com/DepthAuteur/s2m2cinematic}},
  note         = {Fine-tuning infrastructure for bidirectional stereo
                  matching. Pre-trained weights not included.}
}
```

If you use this work, please also cite the original S²M²:

```bibtex
@inproceedings{min2025s2m2,
  title     = {S2M2: Stereo Matching on Stereo Matching},
  author    = {Min, Juhyung and others},
  booktitle = {ICCV},
  year      = {2025}
}
```

---

## Acknowledgments

Built on [S²M²](https://github.com/junhong-3dv/s2m2) by Min et al. (ICCV 2025).

Architect: DepthAuteur
Architectural review: DepthAuteur, Claude Opus 4.6 (Anthropic).
Implementation: DepthAuteur, Claude Sonnet 4.6 (Anthropic).

---

## License

Code: **Apache 2.0** — see [LICENSE](LICENSE).

Pre-trained weights (when released): **CC BY-NC 4.0**.

---

*DepthAuteur — Cinephile · March 2026*
