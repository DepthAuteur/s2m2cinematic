# Conceptual Framework: Bidirectional Stereo Matching

## 1. The Positivity Constraint in Stereo Matching

Conventional stereo matching assumes a specific geometry: the reference camera
(left eye) observes the scene, and the matching camera (right eye) is displaced
horizontally to the right. Under this geometry, all objects produce positive
horizontal pixel shifts — the disparity is always ≥ 0.

This assumption is formalized in most stereo matching architectures as a
hard constraint: disparities are clipped to non-negative values, and the
cost volume is constructed only for positive search directions. In S²M²,
this manifests as a `torch.triu()` mask in the Optimal Transport solver
and explicit `clamp(min=0)` calls in the forward pass.

For automotive and robotics applications, this constraint is correct and
computationally beneficial. Objects do not appear in front of the camera
baseline in standard stereo rig configurations.

---

## 2. Why Negative Disparities Exist in Stereoscopic Source Material

In stereoscopic source domain, the camera rig is configured with an explicit
**convergence point** — the depth plane that appears to coincide with the
screen surface when viewed with stereo glasses. Objects behind the convergence
point produce positive disparities (they appear to recede into the screen).
Objects in front of the convergence point produce **negative disparities**
(they appear to emerge from the screen — the pop-out effect).

Depth compositing artists deliberately compose scenes using both positive
and negative parallax:

- **Screen plane (zero disparity):** The visual anchor — typically the main
  action or dialogue plane.
- **Positive parallax (behind screen):** Background elements, establishing
  depth and scale.
- **Negative parallax (in front of screen):** Foreground elements, creature
  effects, weapons, hands, dramatic close-ups — anything the depth artist
  wants to "reach out" to the audience.

A stereo matching engine that clips negative disparities to zero cannot
represent this compositional intent. It produces a fundamentally incomplete
picture of the depth artist's work.

---

## 3. The Fine-Tuning Approach

S²M²cinematic addresses the positivity constraint through fine-tuning rather
than architectural redesign. The three modifications are minimal:

**Modification 1: Optimal Transport positivity mask**
The `torch.triu()` mask in the Optimal Transport cost aggregation step
restricts the solution space to non-negative disparities. Setting
`use_positivity=False` removes this restriction, allowing the solver to
find matches in both directions.

**Modification 2: Output clamp removal**
Two `disp.clamp(min=0)` calls in the forward pass enforce non-negative
outputs even when the cost volume contains negative evidence. Removing
these via monkey-patch allows negative disparities to propagate through.

**Modification 3: Bidirectional occlusion mask**
The occlusion check `corr_x >= 0` was designed for positive-only search.
Extending to `(corr_x >= 0) & (corr_x < w)` makes the check symmetric,
correctly handling negative search directions.

These three changes are necessary but not sufficient. Without fine-tuning,
the base model produces noisy, unphysical negative values — the network
was not trained to interpret negative correlation patterns meaningfully.
Fine-tuning on real stereo source pairs with NCC-anchored magnitude supervision
teaches the model to produce negative disparities that correspond to actual
physical pop-out geometry.

---

## 4. The SignMagnitudeLoss Design

Previous attempts used a pure sign loss (Hinge loss on sign agreement).
This approach teaches the model *that* certain regions should be negative
but not *how negative* they should be. The result was correct signs but
insufficient magnitude — −2px where −20px was physically correct.

SignMagnitudeLoss combines two objectives:

**Sign component (30% weight, Hinge loss):**
For pixels where the NCC-measured disparity is negative, penalize positive
predictions. This establishes the correct sign direction.

**Magnitude component (70% weight, Regression loss):**
For the same pixels, penalize deviation from the NCC-measured magnitude.
This drives the prediction to the physically correct depth of the pop-out element.

The 70/30 split was chosen empirically: pure magnitude regression
without sign enforcement leads to sign ambiguity early in training.
Pure sign enforcement without magnitude regression produces correct
direction but insufficient depth.

The NCC block-matcher provides the regression target: it measures actual
pixel displacement between left and right eye views at reduced resolution
(downscale factor 2), giving a physically grounded magnitude estimate
without requiring ground truth depth.

---

## 5. Discriminative Behavior

A critical property of S²M²cinematic is that it behaves differently on
different types of negative disparity:

**Genuine pop-out regions (large-format pop-out frame):**
NCC analysis confirms 43.1% of pixels have genuine negative disparity,
with NCC confidence Ø = 0.865. After 100 training steps, predicted
minimum goes from −13.9px (base model noise) to −32.5px (converging
toward NCC target of Ø −11.7px in negative regions).

**Low-evidence regions (classic dramatic frame):**
NCC analysis shows weaker negative evidence (Ø −4.7px, 36.5% negative).
After 100 training steps, the noisy base model negatives (−9.5px) are
reduced to −3.8px — the loss correctly identifies this as a frame where
the negative component should be modest.

This discriminative behavior is the signature of a physically grounded
approach: S²M²cinematic does not push all pixels negative, but drives
each pixel toward its NCC-measured disparity regardless of sign.

---

## 6. Relation to the CCS Ground Truth Problem

S²M²cinematic was developed in the context of the
Creative Cinematic Stereographing (CCS) project, which trains a monocular
depth model on original stereoscopic source materials.

The key insight that motivated S²M²cinematic: any stereo matching engine
that clips negative disparities produces systematically wrong ground truth
for stereoscopic source domain. Pop-out elements — which represent some of the
most compositionally deliberate depth decisions in the work — are
silently converted to zero disparity. A model trained on this ground truth
learns an incomplete and distorted picture of compositional depth.

S²M²cinematic provides the missing piece: physically grounded, bidirectional
ground truth that preserves the full parallax range that depth compositing artists
compose.
