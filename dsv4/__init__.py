"""DeepSeek V4 hybrid attention reference + Pallas-TPU kernel surface.

Two modules:
- `reference`: eager jax.numpy implementation. Ground truth.
- `kernel`: agent-edited Pallas-TPU implementation. Same surface.
"""
