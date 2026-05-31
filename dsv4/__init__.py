"""DeepSeek V4 hybrid attention reference + Pallas-TPU kernel surface.

Two modules:
- `eager`: eager jax.numpy reference implementation. Ground truth.
- `kernel`: agent-edited Pallas-TPU implementation. Same surface.
"""
