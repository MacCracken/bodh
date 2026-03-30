# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `emotion` module — Russell circumplex (valence/arousal), Ekman basic emotions, Scherer appraisal dimensions, Gross emotion regulation, Yerkes-Dodson performance curve, mood-congruent memory bias
- `memory` module — ACT-R base-level activation (Anderson & Lebiere), spreading activation, retrieval probability (softmax gate), retrieval latency, partial matching
- `bayesian` module — Bayes' theorem, likelihood ratio, odds conversion, sequential updating, base rate neglect model, conservative updating (Edwards 1968)
- `social` module — Asch conformity, Latané social impact theory, Kelley covariation attribution, fundamental attribution error, Festinger social comparison
- `fitts_law_shannon` / `fitts_law_shannon_full` — ISO 9241-411 Shannon formulation of Fitts' law
- Reference-value tests verified against published academic sources (TK92, A&S 26.2.23, z-tables)
- 8 new benchmarks for v0.2.0 modules (22 total)

### Changed

- `DualProcess::speed_ratio()` returns `f64::INFINITY` instead of `0.0` for zero system1 speed
- `attention_bottleneck` avoids heap allocation on the happy path
- Spaced repetition docs clarified as geometric model, not literal SM-2

## [0.1.0] - 2026-03-29

### Added

- Initial scaffold with all domain modules
- Full test suite with known-good reference values
- Criterion benchmarks
- CI/CD workflows
