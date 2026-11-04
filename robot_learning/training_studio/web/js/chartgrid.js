// A tiled grid of metric plots inside the bottom bar. "Split view" breaks the
// bottom visualizer into 2/3/4 sub-windows (or a preset), each with its own
// metric selector — like adding joint plots in the Data Studio. Every cell is
// fed by the shared metric bus and keeps full history (backfilled on add).
import { MetricsPlot } from './metrics.js';
import * as bus from './bus.js';

const DEFAULTS = ['loss', 'grad', 'system', 'throughput'];
// the user's predefined 3-plot view: train+eval loss · grad norm · CPU/GPU
export const PRESETS = {
  standard: { label: 'Loss · Grad · GPU/CPU', modes: ['loss', 'grad', 'system'] },
};

export class ChartGrid {
  constructor(container) {
    this.el = container;
    this.cells = [];
    this.history = [];
    bus.on('metricRows', (r) => { for (const x of r) this.history.push(x); if (this.history.length > 20000) this.history = this.history.slice(-16000); });
    bus.on('metricReset', () => { this.history = []; });
    this.addCell('loss');
  }

  count() { return this.cells.length; }

  addCell(mode) {
    if (this.cells.length >= 4) return null;
    mode = mode || DEFAULTS[this.cells.length] || 'loss';
    const cell = document.createElement('div');
    cell.className = 'chart-cell';
    cell.innerHTML = `<div class="cell-head"><div class="seg cell-seg"></div>
      <button class="cell-close" title="remove plot">✕</button></div><canvas></canvas>`;
    this.el.appendChild(cell);
    const plot = new MetricsPlot(cell.querySelector('canvas'), cell.querySelector('.cell-seg'), null);
    plot.setMode(mode);
    if (this.history.length) plot.addRows(this.history);   // backfill the new cell
    const off1 = bus.on('metricRows', (r) => plot.addRows(r));
    const off2 = bus.on('metricReset', () => plot.reset());
    const c = { cell, plot, off1, off2 };
    cell.querySelector('.cell-close').onclick = () => this.removeCell(c);
    this.cells.push(c);
    this._grid();
    return c;
  }

  removeCell(c) {
    if (this.cells.length <= 1) return;   // always keep at least one
    c.off1(); c.off2(); c.plot.destroy();
    c.cell.remove();
    this.cells = this.cells.filter(x => x !== c);
    this._grid();
  }

  // grow/shrink to exactly n cells, keeping existing cells + their chosen modes
  setLayout(n) {
    n = Math.max(1, Math.min(4, n));
    while (this.cells.length > n) this._drop(this.cells[this.cells.length - 1]);
    while (this.cells.length < n) this.addCell();
    this._grid();
  }

  // replace the whole grid with a fixed set of metric modes (the presets)
  setPreset(modes) {
    while (this.cells.length) this._drop(this.cells[this.cells.length - 1]);
    for (const m of modes) this.addCell(m);
    this._grid();
  }

  _drop(c) {   // unconditional teardown (used by setLayout/setPreset)
    c.off1(); c.off2(); c.plot.destroy(); c.cell.remove();
    this.cells = this.cells.filter(x => x !== c);
  }

  _grid() {
    const n = this.cells.length;
    // always a SINGLE horizontal row — every plot side by side (incl. the 4-up split)
    this.el.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
    this.el.style.gridTemplateRows = '1fr';
    // hide the close button when there's only one cell
    this.cells.forEach(c => c.cell.querySelector('.cell-close').style.display = n > 1 ? '' : 'none');
    requestAnimationFrame(() => this.cells.forEach(c => c.plot.draw()));
  }
}
