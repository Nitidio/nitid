# ADR-0002: Native DETRPose integration

- **Status**: Superseded (previously Accepted) —
  Pose/OBB support was removed from the library on 2026-09-25 (#193/#194). The implementation is
  preserved in the `archive/pose-obb-gstreamer` tag.
- **Date**: 2026-08-19
- **Deciders**: nitid maintainers
- **Related**: [DETRPose](https://github.com/SebastianJanampa/DETRPose),

## Context

nitid needs end-to-end multi-person pose estimation with the same coherent API, packaging, device
handling, and quality controls as its D-FINE tasks. DETRPose is derived from D-FINE concepts and uses
compatible HGNetV2 backbone state, but it is a separate checkpoint family with a pose-specific
decoder, query denoising, loss, postprocessing, and positional-embedding behavior.

The official DETRPose source was audited at commit
`4e4a842aaa5afb3d13b40224f070bc3e8e8503f6`. The source and published checkpoints are licensed under
Apache-2.0. The research repository is not an installable runtime dependency and includes assumptions
that do not meet nitid's portability requirements, including fixed-size configuration and CUDA-specific
training paths.

The audit established an upstream inference baseline using the official DETRPose-N COCO checkpoint.
Its checkpoint identity and deterministic zero-input outputs are retained as a golden compatibility
fixture.

## Options considered

1. **Keep DETRPose as a submodule or vendored application tree.** This preserves upstream layout, but
   couples nitid to another configuration, registry, data, and training framework and complicates
   packaging and device support.
2. **Port the pose-specific components and reuse nitid's compatible native core.** This avoids parallel
   frameworks while retaining checkpoint compatibility and one public API.
   architecture and is not compatible with official DETRPose weights.
4. **Depend on the upstream repository at runtime.** The upstream project is not distributed as a
   library and would leak its environment and framework constraints into nitid.

## Decision

Choose option 2. DETRPose support will be implemented natively inside `dfine`; neither a Git submodule
nor a runtime dependency on the reference repository will be shipped.

- The public API is `NITID("nitid1{s,m,l,...}", task="pose")`; internally this maps to the
  matching DETRPose architecture family.
- The operational task registry advertises pose only for compatible pose model configurations.
- Shared HGNetV2 and encoder machinery may be reused only where state and numerical compatibility are
  demonstrated. Pose-specific positional encoding must not change detection or segmentation behavior.
- COCO-17 and CrowdPose-14 keypoint order, horizontal-flip mapping, skeleton, and OKS constants are
  centralized as immutable contracts.
- Official checkpoints are admitted to the download registry only with immutable identity metadata.
  A checkpoint must pass strict loading and upstream golden-output parity before it is considered
  supported.
- Public results represent one `person` class and normalized keypoint coordinates. The official model's
  two internal classification logits are an implementation detail. Genuine detection boxes and learned
  per-keypoint confidence values must not be invented when the model does not produce them.
- Training, validation, export, and device support will use nitid abstractions and must not retain
  CUDA-only paths from the research code.

## Consequences

nitid owns the integration surface, tests, checkpoint conversion, and long-term maintenance. Upstream
updates are adopted deliberately rather than inherited automatically. Reusing verified native layers
keeps the package coherent, while pose-specific numerical modes add an explicit compatibility burden.

The reference checkout is development-only and must not be imported, packaged, or required at runtime.
Apache-2.0 attribution and modification notices remain required for adapted implementation code.

Pose cannot enter the public task registry based on metadata alone. Native model construction,
checkpoint parity, typed result objects, and inference tests are release gates for that API claim.
