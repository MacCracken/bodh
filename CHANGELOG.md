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
- `motivation` module — self-determination theory (basic needs, autonomy index), expectancy-value theory, Csikszentmihalyi flow state, goal gradient effect
- `attention` module — Posner cueing (valid/invalid/neutral), inhibition of return, visual search (feature/conjunction), attentional blink, capacity throughput
- `irt` module — Item Response Theory: 1PL (Rasch), 2PL, 3PL models, item/test information functions, ability standard error
- `bridge` module — cross-crate bridges for pramana statistics: t-stat→d'/Cohen's d, variance→reliability/arousal, regression→learning rate, posterior→odds, F-stat→eta², z-score→valence
- `stress` module — Lazarus transactional stress model (primary/secondary appraisal), coping strategies with controllability fit, resource depletion (Hobfoll COR), stress-performance curve, burnout risk
- Working memory updating: n-back accuracy model, complex span capacity, updating cost with proactive interference (`cognition` module)
- Encoding/retrieval: levels of processing (Craik & Lockhart), encoding strength, generation effect, testing effect (desirable difficulty), encoding specificity (Tulving & Thomson) (`memory` module)
- Big Five personality: dimension scoring with reverse-keyed items, T-score norming, profile distance, profile cosine similarity (`psychometrics` module)

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
