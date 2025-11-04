#!/bin/bash
set -euo pipefail

# Create a small swapfile to reduce OOM risk during heavy pip installs
SWAPFILE=/swapfile
if ! swapon --show | grep -q "$SWAPFILE"; then
  if [ ! -f "$SWAPFILE" ]; then
    echo "[prebuild] Creating 1G swapfile…"
    if command -v fallocate >/dev/null 2>&1; then
      fallocate -l 1G "$SWAPFILE" || true
    fi
    if [ ! -s "$SWAPFILE" ]; then
      dd if=/dev/zero of="$SWAPFILE" bs=1M count=1024 status=none
    fi
    chmod 600 "$SWAPFILE"
    mkswap "$SWAPFILE" >/dev/null 2>&1 || true
  fi
  swapon "$SWAPFILE" || true
  if ! grep -q "$SWAPFILE" /etc/fstab; then
    echo "$SWAPFILE swap swap defaults 0 0" >> /etc/fstab
  fi
fi

echo "[prebuild] Swap status:"
swapon --show || true

