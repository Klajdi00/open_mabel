#!/usr/bin/env bash
# Thin wrapper — delegates to scripts/setup/can_bringup.sh.
# Kept at repo root for muscle memory; do not add logic here.
exec "$(dirname "$0")/scripts/setup/can_bringup.sh" "$@"
