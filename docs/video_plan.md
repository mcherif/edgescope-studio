# Video support plan

## Goals
- Run detection + SAM on video with stable instance IDs.
- Keep controls simple: frame stride, confidence, tracker toggle, output FPS, save annotated clip.
- Remain local-first; avoid cloud calls; fit in laptop GPU/CPU budgets.

## Pipeline sketch
- Decode frames in a stream (no full buffering). Work in RGB uint8.
- Every Nth frame (configurable stride), run RTMDet → class filter/alias → size filter → tracker update.
- Use tracker (e.g., ByteTrack/OC-SORT) to assign IDs; optionally propagate boxes to skipped frames for smoothness.
- For frames with boxes, run SAM with detector boxes + class whitelist to get masks; reuse per-class colors.
- Compose overlays (boxes/labels/masks) and write to an in-memory buffer or temp file for download.

## UI/UX
- New Gradio tab/page: video upload, frame stride slider, confidence slider, tracker toggle, show boxes/masks toggles, output FPS selector, “Save annotated video” checkbox.
- Compact/mobile-friendly layout for phone viewing; show a few key metrics (FPS, frame count).

## Performance guardrails
- Defaults: small stride (e.g., 2–3), resize option to cap long side (e.g., 720p), limit max concurrent SAM masks.
- Async pipeline: decode → detect → track → segment → render, with small queues; avoid GPU/CPU oversubscription.
- Allow CPU-only fallback (skip SAM or reduce stride).

## Tracking choice
- Start with ByteTrack or OC-SORT (no re-ID backbone needed). Inputs: detector boxes/scores; Outputs: track IDs + boxes.
- Keep per-track color stable using track ID hashing.

## Outputs
- Preview a handful of frames in UI; offer download of annotated MP4/WebM.
- Optional: JSON of tracks (frame, bbox, score, label, track_id).

## Open questions / next steps
- Choose tracker library (mmtracking vs lightweight pure-python).
- Decide on max resolution and stride defaults for laptop GPUs.
- Add minimal tests: synthetic video with known boxes; check ID continuity and runtime budget.
