#!/usr/bin/env bash
# Wire-framing checks for DxlBus, compiled and run ON THE HOST — no Teensy, no servo.
#
# Why native: a DYNAMIXEL that does not answer gives you one bit of information. These
# checks pin down the half of the problem that is pure arithmetic (CRC, LENGTH, byte
# stuffing, status parsing) so that a silent bus on the bench can only be wiring, baud,
# or power. The PING case is the ROBOTIS e-manual's own documented packet, byte for
# byte, so the CRC implementation is anchored to the spec rather than to itself.
#
#   the upstream arm stack
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${TMPDIR:-/tmp}/dxl_frame_test"
g++ -std=c++17 -Wall -I"$here/stub" -I"$here/../../include" \
    -o "$out" "$here/dxl_frame_test.cpp"
"$out"
