# Architecture Overview

## Module Map

```
bodh/
├── psychophysics.rs  — Weber-Fechner, Stevens, Fitts, Hick
├── cognition.rs      — Working memory, dual process, cognitive load
├── learning.rs       — Ebbinghaus, spaced repetition, conditioning
├── decision.rs       — Prospect theory, expected utility, satisficing
├── perception.rs     — Signal detection (d'), Gestalt principles
├── psychometrics.rs  — Cronbach's alpha, reliability, Big Five
├── development.rs    — Piaget stages, Erikson stages
└── error.rs          — BodhError enum
```

## Data Flow

Bodh provides computational models — it does not simulate personalities
(that is bhava's domain). Bodh measures and models cognitive processes.

## Consumers

- bhava: psychometric validation
- kiran/joshua: NPC cognition
- agnosai: agent decision models
