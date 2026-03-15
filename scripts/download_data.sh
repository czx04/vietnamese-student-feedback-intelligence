#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base="https://huggingface.co/datasets/tridm/UIT-VSFC/resolve/main"

mkdir -p "$root/data/raw"
for split in train valid test; do
  curl -L --fail --silent --show-error "$base/$split.json" -o "$root/data/raw/$split.json"
done

