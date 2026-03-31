# Architecture Overview

## Module Map

```
bodh/
├── psychophysics.rs  — Weber-Fechner, Stevens, Fitts (original + Shannon), Hick
├── cognition.rs      — Working memory, dual process, cognitive load
├── learning.rs       — Ebbinghaus, spaced repetition, conditioning
├── decision.rs       — Prospect theory, expected utility, satisficing
├── perception.rs     — Signal detection (d'), Gestalt principles
├── psychometrics.rs  — Cronbach's alpha, reliability, Big Five
├── development.rs    — Piaget stages, Erikson stages
├── emotion.rs        — Russell circumplex, Ekman, appraisal, regulation
├── memory.rs         — ACT-R base-level, spreading activation, retrieval
├── bayesian.rs       — Bayes' theorem, belief updating, cognitive biases
├── social.rs         — Conformity, social impact, attribution, comparison
├── motivation.rs     — SDT, expectancy-value, flow state, goal gradient
├── attention.rs      — Posner cueing, visual search, attentional blink
├── irt.rs            — Item Response Theory (1PL/2PL/3PL), information
├── stress.rs         — Transactional stress, coping, burnout
├── bridge.rs         — Cross-crate pramana bridges (primitives only)
└── error.rs          — BodhError enum
```

## Data Flow

Bodh provides computational models — it does not simulate personalities
(that is bhava's domain). Bodh measures and models cognitive processes.

## Consumers

- bhava: psychometric validation
- kiran/joshua: NPC cognition
- agnosai: agent decision models
