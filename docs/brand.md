# Brand

This page is the source of truth for how nitid presents itself: the logo, the
colours, the typography, and what the project says about itself in public. It
exists so the README and the documentation site stay in step instead of
drifting apart.

![nitid](assets/brand/nitid-logo-on-light.png)

## Name

**nitid**, from Latin *nitidus*: clear, transparent, precise. Always lowercase
in prose and in the wordmark, including at the start of a sentence. `NITID` in
capitals refers only to the Python class.

## Logo

| Asset | Use |
|-------|-----|
| [`assets/brand/nitid-logo-on-light.png`](assets/brand/nitid-logo-on-light.png) | Full logo on light backgrounds — navy wordmark |
| [`assets/brand/nitid-logo-on-dark.png`](assets/brand/nitid-logo-on-dark.png) | Full logo on dark backgrounds — light wordmark |
| [`assets/brand/nitid-mark.svg`](assets/brand/nitid-mark.svg) | The mark alone: favicons, avatars, and anywhere too small for the wordmark |
| [`assets/brand/nitid-background.png`](assets/brand/nitid-background.png) | Abstract background for hero areas |

The mark is three nodes — lime, blue, violet — joined into a detection graph.
Keep clear space around the logo of at least the diameter of the lime dot, do
not recolour the nodes, and do not set the navy wordmark on a dark background:
use the light variant instead.

## Demo images

| Asset | Source |
|-------|--------|
| [`assets/nitid-demo.jpg`](assets/nitid-demo.jpg) | Frame at 00:38 of [DiagonalCrosswalkYongeDundas.webm](https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm) by Raysonho (Wikimedia Commons, CC0). Left: `task="detect"`; right: `task="segment"`, both `model1s`, `conf=0.5` |
| [`assets/nitid-track.gif`](assets/nitid-track.gif) | 00:33–00:38 of the same CC0 video. `model1s` detection with `track()` (ByteTrack), `vid_stride=2`, 600 px wide at 8 fps |

Every image the documentation shows must have a recorded source here, and its
licence must allow redistribution.

## Colour

The dark palette is the product palette: it is what nitid looks like on the
website and in the documentation.

| Role | Hex |
|------|-----|
| Background | `#060b1a` |
| Surface | `#0d1530` |
| Surface, sunken | `#080e21` |
| Text | `#e8ecf7` |
| Text, muted | `#b9c4dd` |
| Text, dimmed | `#8794b8` |
| Text, faint | `#6b7ba3` |
| Border | `#ccd5e8` at low alpha |
| Link | `#7fb0ff` |
| Accent (lime) | `#cfe85b` |
| Primary action (blue) | `#1266f2`, hover `#0b52cc` |
| Violet | `#7c22f0`, hover `#6a17d6`, light `#a78bfa` |

The logo carries its own fixed values, which are deliberately more saturated
than the interface palette: navy `#05254b`, violet `#6227fb`, blue `#045efd`,
lime `#cfe85b`.

Vaelsys, the company behind nitid, has a separate warm design system —
background `#f3f2f2`, ink `#201f1d`, copper accent `#b87a3d`. Those tokens
belong to corporate surfaces. Do not mix them into nitid's own interfaces.

## Typography

| Role | Family | Weight |
|------|--------|--------|
| Headings | Archivo | 800 |
| Body | Inter | 400 / 500 |
| Code | JetBrains Mono | 400 |

Each has a system fallback stack, so nothing breaks when the webfonts are
unavailable.

## Voice

nitid's pitch is not "a familiar API". It is that the code is genuinely usable
in a commercial product. For now the public messaging announces the Apache 2.0
licence of the code only; the licence of the pretrained weights is not announced
until it is decided ([#17](https://github.com/Nitidio/nitid/issues/17)).

- **Headline:** Object detection that's actually open source.
- **Supporting line:** Train, validate, export and run vision models. The code
  is under Apache 2.0, so you can ship it in your product without opening your
  code or paying a license.
- **Short form:** No AGPL, no surprises.

The four things that make nitid different, in this order:

1. **Edge first.** The models are designed to run on the hardware next to your
   cameras, not only on a datacenter GPU.
2. **Self-contained checkpoints.** Every checkpoint carries what it needs to be
   reproduced and checked.
3. **Handles messy datasets.** COCO and YOLO formats are read directly, with no
   conversion step.
4. **Export anywhere.** ONNX, OpenVINO, TorchScript and TensorRT, from the same
   checkpoint.

Write plainly: short sentences, no superlatives, no comparisons to competitors
as a selling point. A technical comparison is fine where it informs a decision;
it is not the pitch.

## What we announce

These are deliberate scope decisions, not oversights.

**Three tasks, five operations.** nitid supports exactly three tasks — object
detection, instance segmentation and semantic segmentation — and the public
messaging announces exactly those. The library, its documentation and the pitch
describe the same scope, which keeps the promise narrow and the "three tasks,
five operations" framing honest. The five operations are `predict`, `track`,
`train`, `val` and `export`.

**Semantic segmentation is fine-tune only.** Detection and instance
segmentation ship pretrained COCO checkpoints; semantic segmentation does not.
`task="semantic"` starts from the instance-segmentation checkpoint and is
trained on the user's own masks. Say so wherever the three tasks are announced,
and never show semantic predictions from the untrained starting point (decided
2026-10-02).

**The README is the pitch, not the manual.** It covers the three announced
tasks and the five operations; details belong on the documentation site. Pose
estimation, oriented bounding boxes and GStreamer video I/O were removed from
the library on 2026-09-25 and are preserved in the `archive/pose-obb-gstreamer`
tag.

**The repository is the source of truth for the API.** Where marketing material
and the code disagree about names or signatures, the code wins and the
marketing material gets corrected.

**Products built on top of nitid are announced elsewhere.** Operating systems,
certified hardware and licensing live on the Vaelsys website. This repository
documents the library and nothing else.
