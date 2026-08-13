# Architecture Decision Records

This directory holds the architecture decision records (ADRs) for `media-clusterer`. An ADR captures a
single significant decision — the context that forced it, the options weighed, what was chosen, and what
the project now has to live with as a result.

Write one when a change is hard to reverse, constrains later work, or would otherwise leave a future
reader asking "why on earth is it done this way?". Routine feature work does not need an ADR; a
`*_PLAN.md` at the repository root is the right home for that.

## Conventions

- **Filename:** `NNNN-kebab-case-title.md`, numbered sequentially from `0001`, never renumbered.
- **Immutability:** an accepted ADR is not edited to reflect a change of mind. Write a new one and set the
  old record's status to `Superseded by ADR-NNNN`.
- **Status vocabulary:** `Proposed` → `Accepted` → `Superseded by ADR-NNNN` (or `Rejected`).
- **Citations:** point at code as `src/file.ts:line`, matching the style already used in
  `IMPROVEMENT_PLAN.md`. Claims about the codebase should be checkable without a search.
- **Linting:** files here are covered by MegaLinter and `.markdownlint.json` (`MD013` line-length off,
  `MD024` duplicate headings allowed for non-siblings).

## Template

```markdown
# NNNN. Title

- **Status:** Proposed
- **Date:** YYYY-MM-DD

## Context and problem statement

## Decision drivers

## Considered options

## Decision outcome

## Consequences

## Verification and rollback
```

## Index

| ADR                                                | Title                                                            | Status   |
|----------------------------------------------------|------------------------------------------------------------------|----------|
| [0001](0001-small-video-language-model.md)         | Adopt a small video language model for captioning and embeddings | Proposed |
| [0002](0002-openai-compatible-remote-inference.md) | Offer remote inference against any OpenAI-compatible endpoint    | Proposed |
