// h264.js — the Annex-B parsing the WebCodecs decoder depends on.
//
// If any of this drifts from what the camera server's encoder emits, the studio
// shows BLACK video (and silently falls back to slow MJPEG), so it is worth
// pinning: the wire contract is baseline profile / Annex-B / SPS+PPS in-band at
// every IDR (hardware_bridge/mabel_hw/cameras/video_server.py).
import test from 'node:test';
import assert from 'node:assert/strict';
import { _internals, supported } from '../../web/js/h264.js';

const { splitNALUs, codecString, buildAvcC } = _internals;
const SC4 = [0, 0, 0, 1];
const SC3 = [0, 0, 1];

test('module loads outside a browser and reports WebCodecs unsupported', () => {
  assert.equal(supported, false);      // node has no VideoDecoder
});

test('splits 4-byte-start-code NALUs and strips the start codes', () => {
  const buf = new Uint8Array([...SC4, 0x67, 0xaa, ...SC4, 0x68, 0xbb, ...SC4, 0x65, 0xcc]);
  const nalus = splitNALUs(buf);
  assert.equal(nalus.length, 3);
  assert.deepEqual([...nalus[0]], [0x67, 0xaa]);   // SPS
  assert.deepEqual([...nalus[1]], [0x68, 0xbb]);   // PPS
  assert.deepEqual([...nalus[2]], [0x65, 0xcc]);   // IDR
});

test('handles 3-byte start codes and mixed lengths', () => {
  const buf = new Uint8Array([...SC3, 0x67, 0x01, ...SC4, 0x65, 0x02, 0x03]);
  const nalus = splitNALUs(buf);
  assert.equal(nalus.length, 2);
  assert.deepEqual([...nalus[0]], [0x67, 0x01]);
  assert.deepEqual([...nalus[1]], [0x65, 0x02, 0x03]);
});

test('NAL type is the low 5 bits — keyframe detection depends on it', () => {
  const nalus = splitNALUs(new Uint8Array([...SC4, 0x65, 0x00]));
  assert.equal(nalus[0][0] & 0x1f, 5);             // 0x65 -> IDR
  const delta = splitNALUs(new Uint8Array([...SC4, 0x41, 0x00]));
  assert.equal(delta[0][0] & 0x1f, 1);             // 0x41 -> non-IDR
});

test('empty / start-code-less buffers do not throw', () => {
  assert.deepEqual(splitNALUs(new Uint8Array([])), []);
  assert.deepEqual(splitNALUs(new Uint8Array([1, 2, 3])), []);
});

test('codecString encodes profile/constraints/level from the SPS', () => {
  // 0x42 = baseline (66), 0xc0 constraints, 0x1e = level 3.0 — what NVENC sends
  const sps = new Uint8Array([0x67, 0x42, 0xc0, 0x1e]);
  assert.equal(codecString(sps), 'avc1.42c01e');
});

test('buildAvcC produces a valid AVCDecoderConfigurationRecord', () => {
  const sps = new Uint8Array([0x67, 0x42, 0xc0, 0x1e, 0xaa]);
  const pps = new Uint8Array([0x68, 0xce, 0x3c, 0x80]);
  const avcc = buildAvcC(sps, pps);

  assert.equal(avcc[0], 1);                      // configurationVersion
  assert.equal(avcc[1], 0x42);                   // AVCProfileIndication  <- sps[1]
  assert.equal(avcc[2], 0xc0);                   // profile_compatibility <- sps[2]
  assert.equal(avcc[3], 0x1e);                   // AVCLevelIndication    <- sps[3]
  assert.equal(avcc[4], 0xff);                   // lengthSizeMinusOne = 3
  assert.equal(avcc[5], 0xe1);                   // numOfSPS = 1

  const spsLen = (avcc[6] << 8) | avcc[7];
  assert.equal(spsLen, sps.length);
  assert.deepEqual([...avcc.slice(8, 8 + spsLen)], [...sps]);

  let o = 8 + spsLen;
  assert.equal(avcc[o], 1);                      // numOfPPS = 1
  const ppsLen = (avcc[o + 1] << 8) | avcc[o + 2];
  assert.equal(ppsLen, pps.length);
  assert.deepEqual([...avcc.slice(o + 3, o + 3 + ppsLen)], [...pps]);
  assert.equal(avcc.length, 11 + sps.length + pps.length);
});
