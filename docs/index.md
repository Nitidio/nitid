# nitid

![nitid](assets/brand/nitid-logo-on-light.png#only-light)
![nitid](assets/brand/nitid-logo-on-dark.png#only-dark)

**Object detection that's actually open source.**

Train, validate, export and run vision models. The code and our pretrained
weights are released under the Apache License 2.0, patent grant included, so you
can ship them inside your own product without opening your code or paying for a
license.

*nitid*, from Latin *nitidus*: clear, transparent, precise. **No AGPL, no
surprises.**

## Why nitid?

- **Edge first.** The models are designed to run at the edge, on the hardware
  next to your cameras, not only on a datacenter GPU.
- **Self-contained checkpoints.** Every checkpoint carries the config and the
  class names it needs to be reproduced and checked.
- **Handles messy datasets.** COCO and YOLO layouts are read directly, with no
  conversion step.
- **Export anywhere.** ONNX, OpenVINO, TorchScript and TensorRT, from the same
  checkpoint.

## One library, three tasks, five operations

Object detection, instance segmentation and semantic segmentation, through one
API: `predict`, `track`, `train`, `val` and `export`.

- Python API and command line
- Automatic download of supported official checkpoints
- Fine-tuning and validation for boxes, masks and dense semantic maps
- Tracking and GStreamer/RTSP ingest

## Getting Started

If you're new to nitid, follow these guides in order:

1. [Quick Start](quickstart.md)
2. [Fine-tuning](fine_tuning.md)
3. [Export](export.md)
4. [Troubleshooting](troubleshooting.md)

## Documentation

- [Quick Start](quickstart.md)
- [Fine-tuning](fine_tuning.md)
- [Export](export.md)
- [API Reference](api_reference.md)
- [Troubleshooting](troubleshooting.md)

## Project Direction

- [Roadmap](roadmap.md) — what is in scope, what is not, and how the phases are sequenced
- [Brand](brand.md) — logo, palette, typography, and what the project announces
- [Decision records](adr/index.md) — why the project is built the way it is

## For Contributors

If you're planning to contribute to nitid, start with the [Onboarding Guide](onboarding.md).
