#!/usr/bin/env bash
# Reproducible LXD cluster for neural-pods multi-host validation.
# Creates storage pool, bridge network, node profile and 3 containers.
# Idempotent: existing objects are kept.
set -euo pipefail

NODES=(np-node1 np-node2 np-node3)
POOL=np-pool
PROFILE=np-node
VENV=/srv/ai/workspaces/llm-lora
UV_PY=/home/xrey/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu

lxc storage show "$POOL" >/dev/null 2>&1 || lxc storage create "$POOL" dir
lxc network show lxdbr0 >/dev/null 2>&1 || \
  lxc network create lxdbr0 ipv4.address=10.50.0.1/24 ipv4.nat=true

if ! lxc profile show "$PROFILE" >/dev/null 2>&1; then
  lxc profile create "$PROFILE"
  lxc profile device add "$PROFILE" root disk pool="$POOL" path=/
  lxc profile device add "$PROFILE" eth0 nic name=eth0 nictype=bridged parent=lxdbr0
  lxc profile device add "$PROFILE" project-code disk source=/home/xrey/neural-pods path=/home/xrey/neural-pods
  lxc profile device add "$PROFILE" project-venv disk source="$VENV" path="$VENV"
  lxc profile device add "$PROFILE" uv-python disk source="$UV_PY" path="$UV_PY"
  lxc profile set "$PROFILE" limits.memory=2GiB limits.cpu=4
  lxc profile set "$PROFILE" cloud-init.user-data="#cloud-config
packages: [python3.12, python3.12-venv, python3-pip, postgresql, postgresql-contrib, iputils-ping, netcat-openbsd]"
fi

for node in "${NODES[@]}"; do
  if ! lxc info "$node" >/dev/null 2>&1; then
    lxc launch ubuntu:24.04 "$node" --profile "$PROFILE"
  fi
  lxc start "$node" 2>/dev/null || true
done

for node in "${NODES[@]}"; do
  lxc exec "$node" -- cloud-init status --wait >/dev/null
  echo "$node: $(lxc list "$node" --format csv -c 4 | head -1)"
done
echo '{"cluster_ready":true}'
