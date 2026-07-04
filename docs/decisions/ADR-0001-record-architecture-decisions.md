# ADR-0001: Record architecture decisions as ADRs in the repo

**Status:** Accepted
**Date:** 2026-07-04
**Deciders:** white-dev (project owner)

## Context

This project's explicit goals include documenting *why* things were done — for learning, and
as evidence of authorship and process. Decisions that live only in chat logs or memory are
lost; docs kept outside the repo drift away from the code they describe.

## Decision

Every significant, hard-to-reverse decision (tooling, framework, model architecture, data
source, hosting) gets a numbered ADR in `docs/decisions/`, written **when the decision is
made**, following the format of ADR-0002/0003 (Context → Options with trade-offs → Decision →
Consequences). ADRs are immutable history: superseding a decision means writing a new ADR that
references the old one, never editing the old one.

## Consequences

- The repo carries its own decision history — a reviewer (or future me) can audit the
  reasoning, not just the outcome.
- Small writing overhead per decision (~15 min), accepted as part of the project's
  documentation-first mandate.
