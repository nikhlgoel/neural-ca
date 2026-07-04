"""Model package — the NCA update rule and grid iteration.

Phase 3 fills this in from scratch: a fixed Sobel/identity perception step, a small
1x1-conv update MLP (zero-initialised output), stochastic firing, and alive-masking.
See docs/DESIGN.md §3.
"""
