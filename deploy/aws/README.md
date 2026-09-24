# Three-node deployment on AWS

This guide deploys ARGOS on three machines of different sizes (`large`,
`medium`, `small`), for example Amazon EC2 instances, with the same code on all
three. Every machine runs the worker-node service (`argos.node`) and can run the
orchestrator API (`argos.orchestrator.api`); the orchestrators elect a leader through
heartbeats and a NATS message bus.

**Running the evaluation campaigns.** The campaign runners start their own
orchestration loop: controlled runs execute it inside `src/argos/experiment.py`,
and every live trial starts a dedicated orchestrator on port 8101 with the
control plane and NATS disabled. The campaigns therefore need only the
worker-node service (`argos-node`, sections 1, 2, 4, and 5) on the three
machines, plus SSH access from the `large` machine to the others for
`scripts/set_runtime_mode.py`. The orchestrator service, NATS, and
leader election (sections 3 and 6 to 9) are only needed for a distributed
deployment serving external clients.

## 1) Prerequisites

- Ubuntu 22.04 or 24.04, or Amazon Linux 2023.
- Security Group open between nodes:
  - `8000/tcp` (worker-node API)
  - `8001/tcp` (orchestrator API)
  - `4222/tcp` (NATS, private network only)
- Repository cloned under `/opt/argos/ARGOS`.

## 2) Per-Node Bootstrap

```bash
cd /opt/argos/ARGOS
chmod +x deploy/aws/bootstrap_ubuntu.sh
sudo deploy/aws/bootstrap_ubuntu.sh
```

The script detects `apt`, `dnf`, or `yum`. On Amazon Linux 2023, dependencies
are installed with `dnf`.

## 3) NATS (Once, Preferably On `large`)

In the 3-machine AWS profile, **NATS is mandatory**. If NATS is unavailable, the
control plane should not be considered ready for distributed tests.

If `docker compose` is available:

```bash
cd /opt/argos/ARGOS
docker compose -f deploy/aws/docker-compose.nats.yml up -d
```

If the image does not include the Compose plugin, use this fallback:

```bash
docker rm -f argos-nats 2>/dev/null || true
docker run -d --name argos-nats --restart unless-stopped -p 4222:4222 nats:2-alpine
```

## 4) Environment Variables

Copy and edit:

```bash
sudo cp deploy/aws/env/node.env.example /etc/argos-node.env
sudo cp deploy/aws/env/orchestrator.env.example /etc/argos-orchestrator.env
```

Key variables:

- `ARGOS_MACHINE_ID`: unique identifier per EC2 instance, for example
  `argos-large-1`, `argos-medium-1`, `argos-small-1`.
- `ARGOS_CONTROL_PLANE_NODE_ID`: must match `ARGOS_MACHINE_ID` on each instance.
- `ARGOS_DEPLOYMENT_PROFILE=aws`.
- `NODE_ID`: integer node id for dataset partitioning.
- `ARGOS_CONTROL_PLANE_NATS_ENABLED=1`.
- `ARGOS_CONTROL_PLANE_NATS_URL=nats://<PRIVATE_NATS_IP>:4222`.
- Use VPC private IPs for NATS and node registration endpoints.
- `STATUS_API_TOKEN` and `ORCHESTRATOR_API_TOKEN` must be non-empty in AWS.
  Use the same `STATUS_API_TOKEN` in orchestrator and nodes so `/configure`
  pushes work.

## 5) Install systemd Services

```bash
sudo cp deploy/aws/systemd/argos-node.service /etc/systemd/system/
sudo cp deploy/aws/systemd/argos-orchestrator.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now argos-node argos-orchestrator
```

Detailed operational runbook: [`deploy/aws/FIRST_RUN_3_NODES.md`](FIRST_RUN_3_NODES.md).

## 6) Verification

```bash
curl -s http://127.0.0.1:8000/health
curl -s -H "X-API-Key: $ORCHESTRATOR_API_TOKEN" http://127.0.0.1:8001/cluster/status
curl -s -H "X-API-Key: $ORCHESTRATOR_API_TOKEN" http://127.0.0.1:8001/cluster/control-plane
```

The current leader should appear in `control_plane_leader_id`.

## 7) Node Registration (Once)

On the leader:

```bash
curl -X POST http://127.0.0.1:8001/nodes/register \
  -H "X-API-Key: $ORCHESTRATOR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"node_id":"node-large","endpoint":"http://<PRIVATE_IP_LARGE>:8000"}'
```

Repeat for `medium` and `small`.

## 8) Start The Orchestration Loop

The systemd services start the APIs, but the main loop starts through the API:

```bash
curl -X POST http://127.0.0.1:8001/orchestrator/start \
  -H "X-API-Key: $ORCHESTRATOR_API_TOKEN"
```

Verify afterwards:

```bash
curl -s -H "X-API-Key: $ORCHESTRATOR_API_TOKEN" http://127.0.0.1:8001/cluster/status
```

The response should include `"orchestrator_running": true`.

## 9) Distributed Smoke Test

1. Submit a job through `/submit-job` with `priority`, `resource_limits`, and
   `placement_limits`.
2. Confirm `/cluster/nodes` and `/job/{id}`.
3. Stop the current leader:
   - `sudo systemctl stop argos-orchestrator`
4. Confirm automatic failover by checking `/cluster/control-plane` on another
   node.

Operational note: the current failover covers leader election and prevents stale
plan versions by observing `current_plan_version` on nodes. It does not yet
replicate active jobs or full loop state between processes; strong HA still
requires a shared state backend.

## 10) Recommended Order Before Long Campaigns

1. Smoke test with 1 job and 1 assigned node.
2. Multi-tenant job with different CPU/memory limits per request.
3. Coverage change to verify real reassignment.
4. Leader failure and failover verification (distributed deployment only).
5. The evaluation campaigns ([docs/reproducibility.md](../../docs/reproducibility.md#3-new-campaigns)).
