# Architecture Decision Records

An ADR records a decision that shaped the project, the context that forced it, and the consequences
accepted along with it. It is written once and then left alone: superseded, never rewritten.

nitid keeps ADRs for a specific reason beyond tidiness. The
[certified LTS runtime](https://github.com/Vaelsys/nitid/issues/125) and
[hardware support matrix](https://github.com/Vaelsys/nitid/issues/126) both commit the project to
answering audit questions — *why is this dependency here, who decided, when, and what did you consider
instead?* — with something better than a commit message and a memory. A dated, immutable record is the
cheapest way to be able to answer.

## Index

| ADR | Title | Status | Date |
|---|---|---|---|
| [0002](0002-native-detrpose-integration.md) | Native DETRPose integration | Superseded (removed, #193) | 2026-08-19 |
| [0003](0003-native-rio-detr-obb-integration.md) | Native RiO-DETR OBB integration | Superseded (removed, #194) | 2026-09-07 |

## Convention

- One file per decision, named `NNNN-kebab-case-title.md`, numbered sequentially and never reused.
- Sections: **Status**, **Context**, **Options considered**, **Decision**, **Consequences**.
- Statuses: **Proposed** → **Accepted** → **Superseded by ADR-NNNN**. A *Rejected* ADR is still kept —
  knowing what was turned down, and why, is most of the value.
- Once Accepted, the body is not edited. If the decision changes, write a new ADR and mark the old one
  superseded. Correcting a typo is fine; revising the reasoning after the fact is not.
- Add the new row to the index table in the same pull request.

## When an ADR is required

- Adding or removing a dependency that carries a licence obligation.
- Changing the public API surface (`DFINE`, the `dfine` CLI, the checkpoint schema).
- Choosing or dropping a runtime, export format, or hardware target.
- Taking on third-party source, in any form — vendored, ported, or adapted.
- Anything a certification auditor or a customer's legal team would plausibly ask about.

Routine work does not need one. If you are unsure, the test is whether someone joining in a year would
be confused by the result and unable to reconstruct why.

## Template

```markdown
# ADR-NNNN: Title

- **Status**: Proposed
- **Date**: YYYY-MM-DD
- **Deciders**: names
- **Related**: issue links, prior ADRs

## Context

What forces the decision. Facts and constraints, not opinions. State what is measured and what is
assumed — and if something load-bearing is unmeasured, say so plainly.

## Options considered

Each option with what it gains, what it costs, and what it forecloses. Include the option you rejected
most confidently; that one carries the most information.

## Decision

What was chosen, and the reasoning chain that gets there from the context.

## Consequences

What is now true, what is given up, what becomes harder, and what would cause this to be revisited.
```
