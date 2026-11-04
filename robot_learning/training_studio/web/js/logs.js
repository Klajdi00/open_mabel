// Training log + issues stream (bottom timeline area).
export class LogView {
  constructor(el) { this.el = el; this.tab = 'log'; this.lines = []; }
  setTab(t) { this.tab = t; this.render(); }
  reset() { this.lines = []; this.render(); }
  add(rows) {
    for (const r of rows) this.lines.push(r.line != null ? r.line : String(r));
    if (this.lines.length > 4000) this.lines = this.lines.slice(-3000);
    this.render();
  }
  render() {
    const show = this.tab === 'issues'
      ? this.lines.filter(l => /error|fail|warn|nan|exception|traceback/i.test(l))
      : this.lines;
    const atBottom = this.el.scrollTop + this.el.clientHeight >= this.el.scrollHeight - 40;
    this.el.innerHTML = show.slice(-800).map(l => {
      const cls = /error|fail|exception|traceback/i.test(l) ? 'err'
        : /warn|nan/i.test(l) ? 'warn'
        : /\bloss\b|step|checkpoint|run /i.test(l) ? 'ok' : '';
      return `<div class="log-line ${cls}">${escapeHtml(l)}</div>`;
    }).join('');
    if (atBottom) this.el.scrollTop = this.el.scrollHeight;
  }
}
function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}
