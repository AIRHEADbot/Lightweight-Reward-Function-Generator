# Lightweight Reward Function Generator 🚀

A minimalist toolkit for generating modular reward functions in reinforcement learning environments. Designed for rapid prototyping and seamless integration with RL frameworks.

## Key Features ✨

- 🧩 **Modular Design** - Combine pre-built reward components via YAML/JSON
- ⚡ **Zero-Dependency** - Pure Python implementation (Compatible with PyTorch/TensorFlow)
- 📊 **Visualization Tools** - Built-in reward landscape plotting utilities
- 🧪 **Testing Suite** - Automated validation for reward function consistency
- 🌱 **Curriculum Support** - Dynamic reward shaping for progressive learning

## Installation 📦

```bash
git clone https://github.com/AIRHEADbot/Lightweight-Reward-Function-Generator.git
cd Lightweight-Reward-Function-Generator
pip install -e .
```

## Quick Start 🚀

```python
from reward_generator import RewardComposer

# Define reward components
config = {
    "safety": {"weight": 0.3, "threshold": -0.5},
    "efficiency": {"weight": 0.7, "target": 1.0},
    "curriculum": {
        "stages": [
            {"episodes": 100, "efficiency_weight": 0.5},
            {"episodes": 200, "safety_weight": 0.6}
        ]
    }
}

composer = RewardComposer(config)
reward_function = composer.build()

# Use in RL training loop
state = env.reset()
action = agent.act(state)
next_state, _, done, _ = env.step(action)
reward = reward_function(state, action, next_state)
```

![Reward Landscape Visualization](docs/reward_surface.png)


## Contributing 🤝

We welcome contributions! Please see our [Contribution Guidelines](CONTRIBUTING.md) and review the [Code of Conduct](CODE_OF_CONDUCT.md) before submitting PRs.

## Citation 📝

If using this work in research, please cite:
```bibtex
@software{Yu_Lightweight_Reward_Function_2025,
  author = {Yu, Antares},
  license = {MIT},
  title = {{Lightweight Reward Function Generator}},
  url = {https://github.com/AIRHEADbot/Lightweight-Reward-Function-Generator}
}
```

## License ⚖️

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

**Optimizing reward functions shouldn't require a supercomputer** - Let's build smarter, not heavier! 🦾
