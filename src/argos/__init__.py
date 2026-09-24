# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""ARGOS: reinforcement-learning-driven multidimensional elasticity.

Subpackages and modules:

- ``argos.orchestrator``: orchestrator API, control loop, admission, placement,
  persistence, decision environment, and controllers (``rl``).
- ``argos.node``: worker-node API and analytics runtimes.
- ``argos.services``: geographic heatmap service and spatial fidelity.
- ``argos.domain``: requests, contracts, cost model, and resource monitoring.
- ``argos.experiment``: one controller on one request (training or frozen evaluation).
- ``argos.live_trial``: live traffic client used by the live campaign.
- ``argos.analysis``: campaign validation, statistics, and result tables.
- ``argos.config``, ``argos.settings``, ``argos.common``, ``argos.provenance``:
  configuration loading, fixed settings, shared helpers, and run manifests.
"""

__version__ = "1.0.0"
