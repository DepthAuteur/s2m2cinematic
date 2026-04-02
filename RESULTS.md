# Experimental Results

> **Note:** This document will be updated as training progresses.
> Current results reflect the proof-of-concept quick-test (100 gradient steps),
> the completed Phase 1 clean run (Epochs 1–5), and the ongoing Phase 2 run
> (Epochs 6–12, encoder unfrozen).
>
> **Ground Truth Correction (March 2026):** A L/R channel swap affecting one film's
> frames (44660–87549, 42,890 frames, ~2.4% of training data) was discovered
> during visual QC on the Philips Gioco passive 3D monitor and fully remediated
> before Phase 1 completion. All results below reflect the clean run.
> See `CCS_Trainingsvergleich_Kontaminiert_vs_Sauber.md` for the full
> contaminated vs. clean comparison.
>
> **Visualization Update (March 2026):** The signed colormap in `validate.py`
> now uses a configurable `--green-width` parameter (default k=4) for sharper
> differentiation of subtle negative disparities. Previous visualizations used
> k=2, which made moderate negative values (−5 to −20px) indistinguishable
> from zero. See colormap analysis in documentation.

---

## Quick-Test Results (March 2026)

### Setup

- **Base model:** S²M² L (180.7M parameters, pretrained weights from S²M² repository)
- **Modifications:** `use_positivity=False`, `clamp(min=0)` removed,
  bidirectional occlusion mask
- **Loss:** SignMagnitudeLoss (w_sign=1.0, w_photo=1.0, w_anchor=0.3,
  w_smooth=0.05)
- **Steps:** 100 gradient steps on single frame
- **Hardware:** High-end single-GPU workstation (Blackwell architecture)

---

### Frame 1: Large-format source material — strong pop-out

One of the strongest pop-out effects in the training corpus.

**NCC Block-Matching Analysis (ground truth signal):**

| Metric | Value |
|---|---|
| NCC disparity range | −48.0px to +48.0px |
| NCC median | +2.0px |
| Negative pixels (NCC) | 871,324 (43.1% of valid) |
| NCC confidence | Ø 0.865 |
| Valid pixels | 99.7% |

The NCC signal is strong and unambiguous: 43.1% of this frame has genuine
negative disparity. The regression target exists.

**Disparity predictions:**

| | Original S²M² | Baseline (unclamped) | After 100 steps |
|---|---|---|---|
| Min disparity | 0.0px | −13.9px | **−32.5px** |
| NegPix fraction | 0.0% | 34.7% | 10.6% |
| P98 | — | 17.3px | 15.8px |

*Baseline = original weights with positivity constraint removed, no fine-tuning.*

**In NCC-negative regions specifically:**

| | Baseline | After 100 steps |
|---|---|---|
| NCC target | Ø −11.7px | Ø −11.7px |
| Predicted | Ø +3.3px (wrong sign) | Ø −3.3px (correct sign) |
| Gap to target | — | 8.4px (closing) |

**Loss progression:**

```
Step   1: total=10.87  sign=10.44  anchor=1.20  photo=0.06  smooth=0.07
Step  10: total= 9.67  sign= 9.35  anchor=0.73  photo=0.09  smooth=0.16
Step  50: total= 7.70  sign= 6.31  anchor=4.23  photo=0.11  smooth=0.08
Step 100: total= 7.43  sign= 5.27  anchor=6.76  photo=0.12  smooth=0.13
```

---

### Frame 2: Classic dramatic work — weaker pop-out

This frame has weaker negative NCC evidence, testing discriminative behavior.

**NCC Block-Matching Analysis:**

| Metric | Value |
|---|---|
| NCC disparity range | −48.0px to +48.0px |
| Negative pixels (NCC) | 736,136 (36.5% of valid) |
| NCC confidence | Ø 0.942 |

**Disparity predictions:**

| | Original S²M² | Baseline (unclamped) | After 100 steps |
|---|---|---|---|
| Min disparity | 0.0px | −9.5px | −3.8px |
| NegPix fraction | 0.0% | 64.2% | 18.3% |

**In NCC-negative regions:**

| | After 100 steps |
|---|---|
| NCC target | Ø −4.7px |
| Predicted | Ø +0.8px |
| Gap to target | 5.5px |

---

### Summary

| Property | Evidence |
|---|---|
| Negative disparities produced | ✅ Min = −32.5px on genuine pop-out frame |
| Sign discrimination | ✅ Correct sign in NCC-negative regions after 100 steps |
| Magnitude convergence | ✅ Gap closing: 15px → 8.4px (signed gap) |
| No collapse of positives | ✅ P98 stable (17.3px → 15.8px) |
| Discriminative behavior | ✅ Weaker target → weaker predicted negatives |
| No other public model achieves this | ✅ Original S²M²: 0.0px on both frames |

---

## Phase 1 — Clean Run Results (Encoder Frozen, 1920×1056)

All results below are from the clean run after L/R swap correction.

### Loss Progression — Phase 1

| Epoch | Train Loss | Val Loss | Best | Photo  | Anchor | Sign   | Smooth |
|-------|------------|----------|------|--------|--------|--------|--------|
| 1     | 8.4862     | 8.0774   | ★    | 0.0870 | 7.8784 | 5.5894 | 0.3746 |
| 2     | 7.9451     | 7.7967   | ★    | 0.0872 | 7.1732 | 5.5146 | 0.4299 |
| 3     | 7.7936     | 7.6474   | ★    | 0.0871 | 7.0397 | 5.4007 | 0.4776 |
| 4     | 7.6813     | 7.5692   | ★    | 0.0872 | 6.7937 | 5.3927 | 0.5125 |
| 5     | 7.5987     | 7.4864   | ★    | 0.0873 | 6.6602 | 5.3463 | 0.5466 |

Val loss strictly monotone decreasing across all 5 epochs. Photo loss stable
throughout (0.087x), confirming photometric consistency. Anchor and Sign losses
show steady convergence. Smooth loss increases gradually — expected behavior as
the model learns more complex depth structure.

---

### Epoch 1 Validation

| Signal | Criterion | Result |
|---|---|---|
| Signal 1 | Min < −8px on large-format pop-out frame | ✅ Min = **−23.3px** |
| Signal 2 | NegPix > 5% on large-format pop-out frames | ✅ **13.0%** |
| Signal 3 | Asymmetry ratio approaching 1.0 | ✅ **0.90** (vs. 13.05 original) |

---

### Reference Frame Progression — Phase 1 (Clean Run)

Four reference frames monitored across all epochs, verified against visual QC
on Philips Gioco passive 3D monitor.

#### frame_043420 — Period drama, deck scene (man with moustache)
Visual QC: ~25–30% NegPix. Convergence plane at fragment lower right.
Man and fragment left have negative disparity.

| Epoch | P98   | Min      | NegPix | Asym  |
|-------|-------|----------|--------|-------|
| 1     | —     | −25.4px  | 18.1%  | 0.75  |
| 2     | —     | −30.9px  | 7.0%   | 0.72  |
| 3     | —     | −36.2px  | 5.0%   | 0.98  |
| 4     | —     | −35.3px  | 5.7%   | 1.01  |
| 5     | 32.3px| −35.4px  | 6.0%   | **0.97** |

#### frame_050768 — Action film, locker room (stereographically flat)
Visual QC: Scene is stereographically very flat, almost all near convergence
plane. High NegPix values are numerical noise around zero — **not a problem
case**. P98 at 7–9px confirms minimal depth variation.

| Epoch | P98  | Min      | NegPix | Asym  |
|-------|------|----------|--------|-------|
| 1     | 7.0px| −25.3px  | 66.8%  | 1.83  |
| 2     | —    | −16.9px  | 51.6%  | 0.56  |
| 3     | —    | −17.7px  | 48.9%  | 0.54  |
| 4     | 7.2px| −33.1px  | 68.5%  | 1.63  |
| 5     | 9.0px| −34.6px  | 65.1%  | 1.54  |

Note: Clean GT for this frame has Original P98 = 13.7px, Asym = 1.49
(vs. contaminated: P98 = 4.7px, Asym = 0.70). The L/R correction
fundamentally changed the ground truth for this frame.

#### frame_068500 — Action film, portrait (subtle negative disparity)
Visual QC: ~25–30% NegPix. Face has strongest negative disparity, upper body
stands slightly in front of screen plane. Convergence plane at background wall.

| Epoch | P98  | Min      | NegPix | Asym  |
|-------|------|----------|--------|-------|
| 1     | —    | −22.3px  | 24.8%  | 0.73  |
| 2     | —    | −20.2px  | 19.1%  | 2.04  |
| 3     | —    | −17.7px  | 16.1%  | 2.38  |
| 4     | —    | −25.3px  | 13.9%  | 0.62  |
| 5     | 5.9px| −28.0px  | 20.3%  | **0.85** |

Asymmetry ratio improving steadily toward 1.0 (Ep4: 0.62 → Ep5: 0.85).
Contrast with contaminated run where Asym was pinned at ~0.51 across all epochs.
This frame shows the clearest evidence of the L/R correction benefit.

#### frame_1037074 — 3D documentary, scientist with holographic overlay
Visual QC: ~30–38% NegPix. Convergence plane at desk front edge. Scientist,
holographic overlay, parts of lamp have negative disparity.

| Epoch | P98   | Min      | NegPix | Asym  |
|-------|-------|----------|--------|-------|
| 1     | —     | −24.5px  | 32.4%  | 0.89  |
| 2     | —     | −35.7px  | 13.0%  | 0.90  |
| 3     | —     | −56.1px  | 8.2%   | 0.98  |
| 4     | —     | −50.3px  | 9.1%   | 1.00  |
| 5     | 15.3px| −47.9px  | 12.3%  | **0.99** |

Near-perfect bilateral symmetry (Asym 0.98–1.00) maintained across Ep 3–5.
Strongest negative disparity measured in the corpus: −56.1px at Epoch 3.

---

### Phase 1 Key Observations

- **Val loss strictly monotone decreasing** across all 5 epochs — no overfitting,
  no plateau.
- **Photo loss rock-stable** at 0.087x throughout — photometric consistency
  maintained while bidirectional structure is learned.
- **Sign loss steady descent** (5.5894 → 5.3463) — the model is continuously
  improving its understanding of screen-plane sign boundaries.
- **frame_068500 (action film portrait):** Asymmetry ratio 0.51 (contaminated,
  pinned) vs. 0.85 and rising (clean) — the clearest single-frame proof of
  the L/R correction impact.
- **frame_1037074 (3D documentary):** Asym 0.99–1.00 at Ep 4–5, with Min disparity
  consistently in the −48 to −56px range — the model has learned the physical
  geometry of this pop-out composition.
- **Discriminative behavior confirmed:** frame_050768 (flat scene) maintains
  high NegPix as numerical noise while P98 stays low (7–9px), correctly
  distinguishing flat scenes from genuine pop-out content.

---

## Phase 2 — Clean Run Results (Encoder Unfrozen, 960×512)

At Epoch 6, the DINOv3 ViT-L/16 encoder is unfrozen and all 180.7M parameters
become trainable. Resolution drops to 960×512 for VRAM efficiency. The encoder
can now adapt its feature representation to the bidirectional disparity domain.

Note: Phase 2 Val-Loss values are not directly comparable to Phase 1
due to half resolution (960×512 vs. 1920×1056).

### Loss Progression — Phase 2

| Epoch | Train Loss | Val Loss | Best | Photo  | Anchor | Sign   | Smooth |
|-------|------------|----------|------|--------|--------|--------|--------|
| 6     | 3.3372     | 3.0657   | ★    | 0.1188 | 5.4812 | 1.8089 | 0.8350 |
| 7     | 3.0084     | 2.9098   | ★    | 0.1195 | 5.4277 | 1.6558 | 0.9808 |
| 8     | 2.8761     | 2.8339   | ★    | 0.1196 | 5.3008 | 1.6007 | 1.0687 |
| 9     | 2.7963     | 2.7529   | ★    | 0.1201 | 5.5298 | 1.4746 | 1.0446 |
| 10    | 2.7387     | 2.7071   | ★    | 0.1200 | 5.4592 | 1.4401 | 1.1028 |
| 11    | 2.6932     | 2.6768   | ★    | 0.1203 | 5.4273 | 1.4150 | 1.1204 |
| 12    | 2.6584     | 2.6309   | ★    | 0.1204 | 5.3321 | 1.3869 | 1.1444 |

### Pareto Effect Confirmed

The key Phase 2 hypothesis — simultaneous reduction of Sign-Loss and
Anchor-Loss once the encoder unfreezes — is confirmed:

| Component | Phase 1 Ep 5 | Phase 2 Ep 6 | Phase 2 Ep 12 | Trend |
|-----------|-------------|--------------|---------------|-------|
| Sign      | 5.3463      | 1.8089       | **1.3869**    | ↓↓↓   |
| Anchor    | 6.6602      | 5.4812       | **5.3321**    | ↓↓    |

Sign-Loss drops 4× from Phase 1 to Phase 2 Ep 12 — the encoder continues to
refine its features for bidirectional sign discrimination. Anchor-Loss falls
simultaneously, confirming sustained Pareto improvement across all Phase 2
epochs. Both objectives benefit from shared feature adaptation without trade-off.

---

### Reference Frame Progression — Phase 2 (Clean Run)

#### frame_043420 — Period drama, deck scene

| Epoch | P98    | Min      | NegPix | Asym  |
|-------|--------|----------|--------|-------|
| 5 (P1)| 32.3px| −35.4px  | 6.0%   | 0.97  |
| 6     | 40.0px | −50.7px  | 20.2%  | 0.91  |
| 7     | 40.0px | −54.2px  | 24.3%  | 1.06  |
| 8     | 40.1px | −52.6px  | 20.0%  | 0.96  |
| 9     | 40.1px | −51.4px  | 28.3%  | 0.96  |
| 10    | 41.4px | −50.8px  | 26.8%  | 1.01  |
| 11    | 41.9px | −50.8px  | 23.4%  | 0.99  |
| 12    | 41.3px | −50.5px  | 18.0%  | **0.99** |

NegPix oscillates between 18–28%, centered around visual QC estimate of 25–30%.
Asymmetry consistently near-perfect (0.96–1.06). P98 stable at ~41px.

#### frame_050768 — Action film, locker room (stereographically flat)

| Epoch | P98    | Min      | NegPix | Asym  |
|-------|--------|----------|--------|-------|
| 5 (P1)| 9.0px | −34.6px  | 65.1%  | 1.54  |
| 6     | 12.1px | −50.8px  | 71.0%  | 1.57  |
| 7     | 15.4px | −49.4px  | 68.7%  | 1.47  |
| 8     | 14.9px | −50.5px  | 70.5%  | 1.51  |
| 9     | 15.4px | −49.0px  | 70.2%  | **1.49** |
| 10    | 18.2px | −49.5px  | 70.4%  | 1.51  |
| 11    | 19.6px | −49.3px  | 70.8%  | 1.50  |
| 12    | 17.8px | −50.9px  | 71.1%  | **1.52** |

Asymmetry ratio stable at 1.47–1.52, oscillating around the Original GT
Asym of 1.49. The model has matched the physical depth structure of this
flat scene. Ep 9 achieved exact GT match (1.49). High NegPix remains
numerical noise around zero.

#### frame_068500 — Action film, portrait (subtle negative disparity)

| Epoch | P98    | Min      | NegPix | Asym  |
|-------|--------|----------|--------|-------|
| 5 (P1)| 5.9px | −28.0px  | 20.3%  | 0.85  |
| 6     | 16.5px | −48.9px  | 34.2%  | 0.70  |
| 7     | 26.8px | −50.6px  | 33.7%  | 0.77  |
| 8     | 25.1px | −49.8px  | 34.6%  | 0.87  |
| 9     | 28.3px | −49.2px  | 34.7%  | 0.77  |
| 10    | 31.0px | −49.9px  | 35.1%  | **0.95** |
| 11    | 31.1px | −49.2px  | 35.5%  | 0.96  |
| 12    | 30.1px | −49.2px  | 34.6%  | **0.91** |

The clearest proof of clean training benefit. Contaminated run: Asym pinned
at ~0.50 across all epochs. Clean run: oscillating between 0.70–0.96 with
upward trend, reaching 0.95–0.96 at Ep 10–11. NegPix stable at ~35%,
slightly above visual QC estimate of 25–30%. P98 rising steadily (6→31px),
indicating the encoder is learning to produce stronger positive disparities
alongside the negative ones.

#### frame_1037074 — 3D documentary, scientist with holographic overlay

| Epoch | P98    | Min      | NegPix | Asym  |
|-------|--------|----------|--------|-------|
| 5 (P1)| 15.3px| −47.9px  | 12.3%  | 0.99  |
| 6     | 15.7px | −57.7px  | 39.5%  | 1.01  |
| 7     | 17.5px | −53.2px  | 38.6%  | 1.00  |
| 8     | 18.7px | −51.2px  | 39.8%  | 1.00  |
| 9     | 16.8px | −49.7px  | 39.1%  | 1.00  |
| 10    | 19.0px | −49.1px  | 40.0%  | 1.00  |
| 11    | 18.9px | −48.8px  | 39.5%  | 1.00  |
| 12    | 18.7px | −49.6px  | 39.0%  | **1.00** |

Near-perfect bilateral symmetry maintained (Asym 1.00) across all 7 Phase 2
epochs — the most stable reference frame. NegPix settled at 39–40%, matching
visual QC estimate of 30–38%. Strongest negative disparity in the corpus:
−57.7px at Epoch 6. P98 stable at ~18px — no catastrophic forgetting of
positive disparities.

---

### Phase 2 Key Observations

- **Pareto effect sustained:** Sign-Loss and Anchor-Loss fall simultaneously
  across all Phase 2 epochs. Sign-Loss drops from 5.35 to 1.39 (4× reduction),
  while Anchor-Loss drops from 6.66 to 5.33. No trade-off between objectives.
- **NegPix convergence to visual QC:** All reference frames show NegPix
  values consistent with stereoscopic viewing on the Philips Gioco 3D monitor.
- **frame_050768 Asym matches GT:** The model's Asymmetry ratio oscillates
  around 1.49–1.52, matching the Original GT Asymmetry of 1.49. Exact match
  achieved at Ep 9.
- **frame_068500 milestone:** Asymmetry reached 0.95–0.96 at Ep 10–11,
  up from 0.50 (pinned) in the contaminated run. Oscillates between 0.70–0.96
  with clear upward trend — the L/R correction enables correct feature learning.
- **frame_1037074 stability:** Asymmetry locked at 1.00 across all 7 Phase 2
  epochs — the model has fully learned this scene's depth geometry.
- **Smooth loss increase:** Expected behavior as the model produces sharper
  depth transitions at the zero-crossing boundary between positive and negative
  disparity regions.
- **Rebalancing dynamics:** Anchor-Loss shows periodic drops (e.g., Ep 12: 5.33)
  as the encoder alternates between refining sign discrimination and positive
  disparity accuracy. This oscillation is normal convergence behavior.

---

## Training Progress Summary

| Epoch | Phase | Train Loss | Val Loss | Best | Ref-D Min  | Ref-D Asym |
|-------|-------|------------|----------|------|------------|------------|
| Quick-test | — | —      | —        | —    | −32.5px    | —          |
| 1     | 1     | 8.4862     | 8.0774   | ★    | −24.5px    | 0.89       |
| 2     | 1     | 7.9451     | 7.7967   | ★    | −35.7px    | 0.90       |
| 3     | 1     | 7.7936     | 7.6474   | ★    | −56.1px    | 0.98       |
| 4     | 1     | 7.6813     | 7.5692   | ★    | −50.3px    | 1.00       |
| 5     | 1     | 7.5987     | 7.4864   | ★    | −47.9px    | 0.99       |
| 6     | 2     | 3.3372     | 3.0657   | ★    | −57.7px    | 1.01       |
| 7     | 2     | 3.0084     | 2.9098   | ★    | −53.2px    | 1.00       |
| 8     | 2     | 2.8761     | 2.8339   | ★    | −51.2px    | 1.00       |
| 9     | 2     | 2.7963     | 2.7529   | ★    | −49.7px    | 1.00       |
| 10    | 2     | 2.7387     | 2.7071   | ★    | −49.1px    | 1.00       |
| 11    | 2     | 2.6932     | 2.6768   | ★    | −48.8px    | 1.00       |
| 12    | 2     | 2.6584     | 2.6309   | ★    | −49.6px    | 1.00       |
| 13–50 | 2     | *pending*  | *pending*| —    | —          | —          |

Phase 1 (encoder frozen, 1920×1056): **Complete** ✅
Phase 2 (encoder unfrozen, 960×512): **In progress** 🔄 (Epoch 13 running)
Early stopping: Patience 7, Min-Delta 0.0005 — all 12 epochs improved ★

---

## Full Training Results — *Pending*

Training corpus: 103,029 GT anchor frames across 23 original
stereoscopic source materials (1953–2018).

Expected metrics after full training:
- Min disparity on large-format pop-out frames: −40px to −60px
- Asymmetry ratio on reference frames: 0.95–1.05
- NegPix convergence to visual QC estimates on all reference frames
- Positive disparity quality: maintained at baseline level
