#!/usr/bin/env python3
"""Set the ARGOS node runtime mode across configured node services."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from argos.config import DEFAULT_CONFIG_PATH, load_project_config, node_endpoint_list  # noqa: E402

RUNTIMES = {"thread", "process"}


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _local_ips() -> set[str]:
    ips = {"127.0.0.1", "localhost"}
    with suppress(socket.gaierror):
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    with suppress(Exception):
        ips.update(subprocess.check_output(["hostname", "-I"], text=True).split())
    return ips


def _endpoint_host(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    return parsed.hostname or endpoint


def _node_command(runtime: str) -> str:
    return (
        "set -euo pipefail; "
        "stamp=$(date -u +%Y%m%d_%H%M%S); "
        "sudo cp /etc/argos-node.env /etc/argos-node.env.bak.${stamp}; "
        f"if sudo grep -q '^ARGOS_RUNTIME_EXECUTION_MODE=' /etc/argos-node.env; then "
        f"sudo sed -i 's/^ARGOS_RUNTIME_EXECUTION_MODE=.*/ARGOS_RUNTIME_EXECUTION_MODE={runtime}/' /etc/argos-node.env; "
        f"else echo 'ARGOS_RUNTIME_EXECUTION_MODE={runtime}' | sudo tee -a /etc/argos-node.env >/dev/null; fi; "
        "sudo systemctl restart argos-node"
    )


def set_node_runtime(endpoint: str, runtime: str, *, ssh_key: Path, ssh_user: str, dry_run: bool) -> None:
    host = _endpoint_host(endpoint)
    command = _node_command(runtime)
    if host in _local_ips():
        _run(["bash", "-lc", command], dry_run=dry_run)
        return
    _run(
        [
            "ssh",
            "-i",
            str(ssh_key),
            "-o",
            "StrictHostKeyChecking=no",
            f"{ssh_user}@{host}",
            command,
        ],
        dry_run=dry_run,
    )


def verify_endpoint(endpoint: str, runtime: str, *, timeout_seconds: float) -> dict[str, str]:
    deadline = time.monotonic() + timeout_seconds
    url = endpoint.rstrip("/") + "/metrics"
    last_error = ""
    while time.monotonic() <= deadline:
        try:
            with urllib.request.urlopen(url, timeout=3.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            mode = str(payload.get("runtime_mode") or "")
            if mode == runtime:
                return {"endpoint": endpoint, "runtime_mode": mode, "status": "ok"}
            last_error = f"runtime_mode={mode}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2.0)
    raise RuntimeError(f"{endpoint} did not report runtime={runtime}: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", choices=sorted(RUNTIMES))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--ssh-key", type=Path, default=None, help="Default: runtime.ssh.key_path in the config")
    parser.add_argument("--ssh-user", default=None, help="Default: runtime.ssh.user in the config")
    parser.add_argument("--verify-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_project_config(args.config)
    ssh_config = (config.get("runtime") or {}).get("ssh") or {}
    if args.ssh_user is None:
        args.ssh_user = str(ssh_config.get("user") or "")
    if args.ssh_key is None:
        args.ssh_key = Path(str(ssh_config.get("key_path") or "")).expanduser()
    if not args.ssh_user or "<" in args.ssh_user or "<" in str(args.ssh_key):
        raise RuntimeError("Set runtime.ssh.user and runtime.ssh.key_path in the config, or pass --ssh-user/--ssh-key")
    endpoints = node_endpoint_list(config)
    if not endpoints:
        raise RuntimeError("No node endpoints configured")
    for endpoint in endpoints:
        set_node_runtime(endpoint, args.runtime, ssh_key=args.ssh_key, ssh_user=args.ssh_user, dry_run=args.dry_run)
    if args.dry_run:
        return 0
    rows = [
        verify_endpoint(endpoint, args.runtime, timeout_seconds=args.verify_timeout_seconds) for endpoint in endpoints
    ]
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
