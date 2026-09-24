#!/usr/bin/env bash
set -euo pipefail

NODE_URL="${NODE_URL:-http://127.0.0.1:8000}"
ORCH_URL="${ORCH_URL:-http://127.0.0.1:8001}"

echo "[node] ${NODE_URL}/health"
curl -fsS "${NODE_URL}/health" | jq .

echo "[orchestrator] ${ORCH_URL}/cluster/status"
curl -fsS "${ORCH_URL}/cluster/status" | jq .

echo "[control-plane] ${ORCH_URL}/cluster/control-plane"
curl -fsS "${ORCH_URL}/cluster/control-plane" | jq .

echo "OK"
