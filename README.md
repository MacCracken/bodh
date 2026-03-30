# Bodh

**बोध** (Sanskrit: awareness, understanding) — Psychology engine for cognition, perception, learning, and decision-making.

Part of the [AGNOS](https://github.com/MacCracken/agnosticos) science crate ecosystem.

## Key Capabilities

- **Psychophysics**: Weber-Fechner law, Stevens' power law, Fitts' law, Hick's law
- **Cognition**: Working memory (Baddeley), dual process theory (Kahneman), cognitive load
- **Learning**: Ebbinghaus forgetting curve, spaced repetition, Rescorla-Wagner conditioning
- **Decision-making**: Prospect theory (Kahneman/Tversky), expected utility, bounded rationality
- **Perception**: Signal detection theory (d-prime), Gestalt principles
- **Psychometrics**: Cronbach's alpha, split-half reliability, Big Five measurement
- **Development**: Piaget stages, Erikson psychosocial stages

## Quick Start

```rust
use bodh::psychophysics;

// Fitts' law: index of difficulty for a UI target
let id = psychophysics::fitts_law(256.0, 4.0).unwrap();
assert!((id - 7.0).abs() < 1e-10); // 7 bits

// Ebbinghaus forgetting curve
let retention = bodh::learning::ebbinghaus_forgetting(0.0, 1.0).unwrap();
assert!((retention - 1.0).abs() < 1e-10); // perfect at t=0

// Prospect theory value function
let gain = bodh::decision::prospect_theory_value(200.0, 100.0, 0.88, 0.88, 2.25).unwrap();
let loss = bodh::decision::prospect_theory_value(0.0, 100.0, 0.88, 0.88, 2.25).unwrap();
assert!(loss.abs() > gain.abs()); // loss aversion
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Standard library support |
| `hisab` | No | Advanced math via hisab |
| `pramana` | No | Statistics via pramana |
| `logging` | No | Tracing subscriber |
| `full` | No | All features |

## Consumers

- **bhava** — psychometric validation of personality measurements
- **kiran/joshua** — NPC cognition, player modeling
- **agnosai** — decision-making models for agents

## License

GPL-3.0-only
