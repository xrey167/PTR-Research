#!/usr/bin/env bash
# Configure PostgreSQL 16 in the LXD nodes for cross-host access from the
# quorum benchmark (listen on TCP, trust connections from the lxdbr0 subnet).
# Idempotent. Lab-only trust rule scoped to the private bridge subnet.
set -euo pipefail

NODES=(np-node1 np-node2 np-node3)

for node in "${NODES[@]}"; do
  lxc exec "$node" -- bash -c '
    CFG=/etc/postgresql/16/main/postgresql.conf
    HBA=/etc/postgresql/16/main/pg_hba.conf
    grep -q "^listen_addresses" "$CFG" 2>/dev/null || \
      echo "listen_addresses = '"'"'*'"'"'" >> "$CFG"
    grep -q "10.50.0.0/24" "$HBA" 2>/dev/null || \
      echo "host all postgres 10.50.0.0/24 trust" >> "$HBA"
    systemctl restart postgresql
  '
  echo "$node: $(lxc exec "$node" -- pg_isready -h 127.0.0.1)"
done

# Verify cross-host reachability from node1.
lxc exec np-node1 -- bash -c '
  for peer in np-node2 np-node3; do
    pg_isready -h "$peer" -U postgres >/dev/null && echo "$peer reachable"
  done
'
echo '{"postgres_hosts_ready":true}'
