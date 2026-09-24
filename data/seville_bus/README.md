# Seville mobility dataset

Input data of the geographic heatmap service: 60 simulated per-user movement
trajectories over Seville, Spain, generated for this project with the
Opportunistic Network Environment (ONE) mobility simulator.

| Item | Value |
|---|---|
| Files | `SB_User0.json` to `SB_User59.json`, one per simulated user |
| Points | 128,103 in total |
| Format | JSON array of `{"lng": <float>, "lat": <float>, "timestamp": <ISO 8601, UTC>}` |
| Time span | 4 to 8 February 2026 (simulated time) |

Each worker node loads the users assigned to its `NODE_ID`
(`get_node_user_assignments` in `src/argos/node.py`), so coverage selects
users and sample selects points within them.

438 points (0.3 %) have coordinates (0, 0). They lie outside the analysis area
and are discarded by the bounding-box filter of the heatmap service
(`src/argos/services/geo_heatmap.py`).

The dataset is part of the repository and is distributed under the same
[MIT License](../../LICENSE).
