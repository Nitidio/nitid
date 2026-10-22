# Architecture Decision Records

An ADR records a decision that shaped the project, the context that forced it, and the consequences
accepted along with it. It is written once and then left alone: superseded, never rewritten.

ADRs answer the questions a commit message cannot: *why is this dependency here, who decided, when,
and what was considered instead?* A dated, immutable record is the cheapest way to keep those answers.

## Index

No ADRs are published yet. Numbers up to 0003 were used before the first public release and are
retired, so the next record is ADR-0004.

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
- Changing the public API surface (`NITID`, the `nitid` CLI, the checkpoint schema).
- Choosing or dropping a runtime, export format, or hardware target.
- Taking on third-party source, in any form — vendored, ported, or adapted.
- Anything an auditor or a downstream user's legal team would plausibly ask about.

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
