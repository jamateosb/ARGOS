# First Run Checklist (large + medium + small)

Step-by-step first run of a distributed ARGOS deployment on three machines
with the same code on all of them.

## 1) Recommended Topology

Use a stable identifier per role:

| Role | `ARGOS_MACHINE_ID` | `ARGOS_CONTROL_PLANE_NODE_ID` | `NODE_ID` |
|------|------------------|-----------------------------|-----------|
| `large` | `argos-large-1` | `argos-large-1` | `1` |
| `medium` | `argos-medium-1` | `argos-medium-1` | `2` |
| `small` | `argos-small-1` | `argos-small-1` | `3` |

Considerations:

- `NODE_ID` only needs to be unique per machine.
- Use VPC private IPs for NATS and node registration endpoints.
- `large` is the recommended NATS host and will normally be the initial leader
  by score.
- `STATUS_API_TOKEN` must be the same in orchestrator and nodes; in AWS it must
  be configured explicitly.
- `ORCHESTRATOR_API_TOKEN` protects orchestrator API operations; in AWS it must
  be configured explicitly.

## 2) Required Data Before Startup

Define this matrix before editing services:

| Role | Private IP | Public IP | Hostname |
|------|------------|-----------|----------|
| `large` | `<PRIVATE_IP_LARGE>` | `<PUBLIC_IP_LARGE>` | `<HOSTNAME_LARGE>` |
| `medium` | `<PRIVATE_IP_MEDIUM>` | `<PUBLIC_IP_MEDIUM>` | `<HOSTNAME_MEDIUM>` |
| `small` | `<PRIVATE_IP_SMALL>` | `<PUBLIC_IP_SMALL>` | `<HOSTNAME_SMALL>` |

Do not commit ephemeral public IPs to the repository. If an EC2 instance has no
Elastic IP, its public IP can change.

## 3) Bootstrap On All 3 Machines

```bash
sudo mkdir -p /opt/argos
sudo chown -R "$USER":"$USER" /opt/argos
cd /opt/argos
git clone https://github.com/jamateosb/ARGOS.git ARGOS
cd ARGOS
git checkout v1.0.0
chmod +x deploy/aws/bootstrap_ubuntu.sh
sudo deploy/aws/bootstrap_ubuntu.sh
```

The bootstrap detects `apt`, `dnf`, or `yum`. On Amazon Linux 2023 it uses
`dnf`.

## 4) Configure Per-Machine Variables

### `large`

`/etc/argos-node.env`

```bash
ARGOS_MACHINE_ID=argos-large-1
NODE_ID=1
STATUS_API_TOKEN=<shared-status-api-token>
ARGOS_MAX_ANALYTICS_PER_NODE=6
```

`/etc/argos-orchestrator.env`

```bash
ORCHESTRATOR_API_TOKEN=<orchestrator-api-token>
STATUS_API_TOKEN=<shared-status-api-token>
ARGOS_MACHINE_ID=argos-large-1
ARGOS_CONTROL_PLANE_NODE_ID=argos-large-1
ARGOS_DEPLOYMENT_PROFILE=aws
ARGOS_CONTROL_PLANE_ENABLED=1
ARGOS_CONTROL_PLANE_LEASE_SECONDS=15
ARGOS_CONTROL_PLANE_HEARTBEAT_SECONDS=5
ARGOS_CONTROL_PLANE_NATS_ENABLED=1
ARGOS_CONTROL_PLANE_NATS_URL=nats://<PRIVATE_IP_LARGE>:4222
ARGOS_CONTROL_PLANE_SUBJECT_PREFIX=argos.control
ARGOS_LOOP_ITERATION_BUDGET_MS=2000
```

### `medium`

`/etc/argos-node.env`

```bash
ARGOS_MACHINE_ID=argos-medium-1
NODE_ID=2
STATUS_API_TOKEN=<shared-status-api-token>
ARGOS_MAX_ANALYTICS_PER_NODE=6
```

`/etc/argos-orchestrator.env`

```bash
ORCHESTRATOR_API_TOKEN=<orchestrator-api-token>
STATUS_API_TOKEN=<shared-status-api-token>
ARGOS_MACHINE_ID=argos-medium-1
ARGOS_CONTROL_PLANE_NODE_ID=argos-medium-1
ARGOS_DEPLOYMENT_PROFILE=aws
ARGOS_CONTROL_PLANE_ENABLED=1
ARGOS_CONTROL_PLANE_LEASE_SECONDS=15
ARGOS_CONTROL_PLANE_HEARTBEAT_SECONDS=5
ARGOS_CONTROL_PLANE_NATS_ENABLED=1
ARGOS_CONTROL_PLANE_NATS_URL=nats://<PRIVATE_IP_LARGE>:4222
ARGOS_CONTROL_PLANE_SUBJECT_PREFIX=argos.control
ARGOS_LOOP_ITERATION_BUDGET_MS=2000
```

### `small`

`/etc/argos-node.env`

```bash
ARGOS_MACHINE_ID=argos-small-1
NODE_ID=3
STATUS_API_TOKEN=<shared-status-api-token>
ARGOS_MAX_ANALYTICS_PER_NODE=6
```

`/etc/argos-orchestrator.env`

```bash
ORCHESTRATOR_API_TOKEN=<orchestrator-api-token>
STATUS_API_TOKEN=<shared-status-api-token>
ARGOS_MACHINE_ID=argos-small-1
ARGOS_CONTROL_PLANE_NODE_ID=argos-small-1
ARGOS_DEPLOYMENT_PROFILE=aws
ARGOS_CONTROL_PLANE_ENABLED=1
ARGOS_CONTROL_PLANE_LEASE_SECONDS=15
ARGOS_CONTROL_PLANE_HEARTBEAT_SECONDS=5
ARGOS_CONTROL_PLANE_NATS_ENABLED=1
ARGOS_CONTROL_PLANE_NATS_URL=nats://<PRIVATE_IP_LARGE>:4222
ARGOS_CONTROL_PLANE_SUBJECT_PREFIX=argos.control
ARGOS_LOOP_ITERATION_BUDGET_MS=2000
```

## 5) Exact Startup Order

1. Start NATS only on `large`.

If `docker compose` is available:

```bash
cd /opt/argos/ARGOS
docker compose -f deploy/aws/docker-compose.nats.yml up -d
```

If unavailable, use this fallback:

```bash
docker rm -f argos-nats 2>/dev/null || true
docker run -d --name argos-nats --restart unless-stopped -p 4222:4222 nats:2-alpine
```

2. Install `systemd` services on all 3 machines.

```bash
sudo cp deploy/aws/systemd/argos-node.service /etc/systemd/system/
sudo cp deploy/aws/systemd/argos-orchestrator.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now argos-node argos-orchestrator
```

3. Verify local health on each machine.

```bash
curl -s http://127.0.0.1:8000/health
curl -s -H "X-API-Key: $ORCHESTRATOR_API_TOKEN" http://127.0.0.1:8001/cluster/status
curl -s -H "X-API-Key: $ORCHESTRATOR_API_TOKEN" http://127.0.0.1:8001/cluster/control-plane
```

4. Wait until `large` appears as leader in at least one `/cluster/control-plane`
   response.
5. Register all 3 nodes from the leader using private IPs.

```bash
curl -X POST http://127.0.0.1:8001/nodes/register \
  -H "X-API-Key: $ORCHESTRATOR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"node_id":"node-large","endpoint":"http://<PRIVATE_IP_LARGE>:8000"}'

curl -X POST http://127.0.0.1:8001/nodes/register \
  -H "X-API-Key: $ORCHESTRATOR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"node_id":"node-medium","endpoint":"http://<PRIVATE_IP_MEDIUM>:8000"}'

curl -X POST http://127.0.0.1:8001/nodes/register \
  -H "X-API-Key: $ORCHESTRATOR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"node_id":"node-small","endpoint":"http://<PRIVATE_IP_SMALL>:8000"}'
```

6. Start the main loop on the leader.

```bash
curl -X POST http://127.0.0.1:8001/orchestrator/start \
  -H "X-API-Key: $ORCHESTRATOR_API_TOKEN"
curl -s -H "X-API-Key: $ORCHESTRATOR_API_TOKEN" http://127.0.0.1:8001/cluster/status
```

7. Submit the first job only after `"orchestrator_running": true` is visible.

## 6) First Smoke Test

```bash
curl -X POST http://127.0.0.1:8001/submit-job \
  -H "X-API-Key: $ORCHESTRATOR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "service_type": "geo_heatmap",
    "coverage_min": 0.34,
    "coverage_max": 0.34,
    "sample_min": 0.50,
    "sample_max": 0.50,
    "freshness_min": 60,
    "freshness_max": 60,
    "tenant_id": "aws-smoke",
    "priority": "standard",
    "algorithm": "qlearning"
  }'
```

Checks:

- `GET /cluster/nodes`
- `GET /cluster/status`
- `GET /job/<request_id>`
- `GET /history/errors`
- `GET /history/slo-violations`

## 7) If Something Fails

- If `cluster/control-plane` does not show `nats_connected=true`, check
  `requirements.txt`, `docker compose ps`, and `ARGOS_CONTROL_PLANE_NATS_URL`.
- If `cluster/status` shows `orchestrator_running=false`, `/orchestrator/start`
  has not been called yet.
- If nodes do not receive a plan, verify that each registered `endpoint` uses
  private IP and port `8000`.
