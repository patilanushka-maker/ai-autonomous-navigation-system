"""Perception module.

Intended Responsibility:
- Provide a known-map perception adapter interfacing the robot with the ground-truth
  environment.
- Query local obstacle positions relative to current robot state.
- Explicit scope boundary: In this 2D simulator, perception functions as an occupancy
  grid adapter providing known obstacle coordinates, NOT camera-based computer vision,
  deep learning detectors, or SLAM.

TODO: Implement KnownMapPerceptionAdapter in the subsequent implementation phase.
"""
