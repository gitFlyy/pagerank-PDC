#!/usr/bin/env bash
set -euo pipefail

EDGE_FILE="${1:-web-Google.metis.edges}"
NPROCS=(1 2 4 8)
SCENARIOS=(
  pagerank_p2p
  pagerank_collective
  pagerank_async
)

if [[ ! -f "$EDGE_FILE" ]]; then
  echo "Error: edge file not found: $EDGE_FILE"
  echo "Usage: ./run.sh [path/to/edges_file]"
  exit 1
fi

if ! command -v mpirun >/dev/null 2>&1; then
  echo "Error: mpirun not found in PATH"
  exit 1
fi

echo "Building binaries..."
make all

echo "Running PageRank scenarios with edge file: $EDGE_FILE"
for np in "${NPROCS[@]}"; do
  for scenario in "${SCENARIOS[@]}"; do
    echo
    echo "========================================"
    echo "Scenario: $scenario | np=$np"
    echo "========================================"
    mpirun -np "$np" "./$scenario" "$EDGE_FILE"
  done
done

echo

echo "All runs completed."
