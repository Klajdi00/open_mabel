// Binary map-plane decoder vs the server/docs/data_streams.md record layouts.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { decodeRecord } from '../../web/js/maplink.js';

function meshBuf({ final = true, indexed = false, verts, tris = [] }){
  const nv = verts.length;
  const size = 8 + nv * 16 + (indexed ? 4 + tris.length * 4 : 0);
  const buf = new ArrayBuffer(size);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x01);
  dv.setUint8(1, (final ? 1 : 0) | (indexed ? 2 : 0));
  dv.setUint32(4, nv, true);
  verts.forEach(([x, y, z, r, g, b, a], i) => {
    const o = 8 + i * 16;
    dv.setFloat32(o, x, true); dv.setFloat32(o + 4, y, true); dv.setFloat32(o + 8, z, true);
    dv.setUint8(o + 12, r); dv.setUint8(o + 13, g); dv.setUint8(o + 14, b); dv.setUint8(o + 15, a);
  });
  if (indexed){
    const io = 8 + nv * 16;
    dv.setUint32(io, tris.length, true);
    tris.forEach((t, i) => dv.setUint32(io + 4 + i * 4, t, true));
  }
  return buf;
}

test('MESH 0x01 indexed', () => {
  const r = decodeRecord(meshBuf({
    indexed: true,
    verts: [[0, 0, 0, 255, 0, 0, 255], [1, 0, 0, 0, 255, 0, 255], [0, 1, 0, 0, 0, 255, 255]],
    tris: [0, 1, 2],
  }));
  assert.equal(r.kind, 'mesh');
  assert.equal(r.final, true);
  assert.equal(r.count, 3);
  assert.deepEqual([...r.idx], [0, 1, 2]);
  assert.equal(r.pos[3], 1);                       // second vertex x
  assert.ok(Math.abs(r.col[0] - 1) < 1e-6);        // red normalized
  assert.ok(Math.abs(r.col[4] - 1) < 1e-6);        // green of v1
});

test('MESH chunk (not final, points-only)', () => {
  const r = decodeRecord(meshBuf({ final: false, verts: [[1, 2, 3, 9, 9, 9, 255]] }));
  assert.equal(r.final, false);
  assert.equal(r.idx, null);
  assert.deepEqual([...r.pos], [1, 2, 3]);
});

test('CLOUD 0x05 shares the vertex layout', () => {
  const buf = meshBuf({ verts: [[4, 5, 6, 10, 20, 30, 255]] });
  new DataView(buf).setUint8(0, 0x05);
  const r = decodeRecord(buf);
  assert.equal(r.kind, 'cloud');
  assert.deepEqual([...r.pos], [4, 5, 6]);
});

test('ODOM 0x02', () => {
  const buf = new ArrayBuffer(32);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x02);
  [1.5, -2, 0.1, 0, 0, 0.7071, 0.7071].forEach((v, i) => dv.setFloat32(4 + i * 4, v, true));
  const r = decodeRecord(buf);
  assert.equal(r.kind, 'odom');
  assert.ok(Math.abs(r.x - 1.5) < 1e-6 && Math.abs(r.y + 2) < 1e-6);
  assert.ok(Math.abs(r.qz - 0.7071) < 1e-4 && Math.abs(r.qw - 0.7071) < 1e-4);
});

test('PATH 0x03 and empty clear', () => {
  const mk = (pts) => {
    const buf = new ArrayBuffer(8 + pts.length * 4);
    const dv = new DataView(buf);
    dv.setUint8(0, 0x03);
    dv.setUint32(4, pts.length / 3, true);
    pts.forEach((v, i) => dv.setFloat32(8 + i * 4, v, true));
    return buf;
  };
  const r = decodeRecord(mk([0, 0, 0, 1, 1, 0]));
  assert.equal(r.kind, 'path');
  assert.equal(r.count, 2);
  assert.ok(Math.abs(r.pts[3] - 1) < 1e-6);
  assert.equal(decodeRecord(mk([])).count, 0);
});

test('GRID 0x04 header + cells', () => {
  const w = 3, h = 2;
  const buf = new ArrayBuffer(32 + w * h);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x04);
  dv.setUint32(4, w, true); dv.setUint32(8, h, true);
  dv.setFloat32(12, 0.05, true);                       // resolution
  dv.setFloat32(16, -1, true); dv.setFloat32(20, -2, true);   // ox oy
  dv.setFloat32(24, 0, true); dv.setFloat32(28, 0.5, true);   // oz yaw
  const cells = [0, 50, 100, 255, 0, 100];
  cells.forEach((c, i) => dv.setUint8(32 + i, c));
  const r = decodeRecord(buf);
  assert.equal(r.kind, 'grid');
  assert.equal(r.w, 3); assert.equal(r.h, 2);
  assert.ok(Math.abs(r.res - 0.05) < 1e-6);
  assert.ok(Math.abs(r.ox + 1) < 1e-6 && Math.abs(r.oy + 2) < 1e-6);
  assert.ok(Math.abs(r.yaw - 0.5) < 1e-6);
  assert.deepEqual([...r.cells], cells);
});

test('unknown tag → null', () => {
  const buf = new ArrayBuffer(8);
  new DataView(buf).setUint8(0, 0x7f);
  assert.equal(decodeRecord(buf), null);
});
