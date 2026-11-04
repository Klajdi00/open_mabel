// REST + WebSocket client for the Trainer Studio backend.
import { postAuthed } from './auth.js';
const J = (r) => { if (!r.ok) return r.text().then(t => { throw new Error(t || r.status); }); return r.json(); };
const post = (url, body) => fetch(url, {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body || {}),
}).then(J);
// Mutating training actions go through the operator sign-in gate (styled popup +
// signed session); `action` is the verb phrase shown in the popup subtitle.
const postGated = (url, body, action) => postAuthed(url, body, action);

export const api = {
  health: () => fetch('/api/health').then(J),
  compute: () => fetch('/api/compute').then(J),
  embodiment: () => fetch('/api/embodiment').then(J),
  datasets: () => fetch('/api/datasets').then(J),
  importDataset: (path) => post('/api/datasets/import', { path }),
  archSchema: () => fetch('/api/arch/schema').then(J),

  selection: (body) => post('/api/selection', body),
  archSpec: (body) => post('/api/arch/spec', body),
  archCompile: (body) => post('/api/arch/compile', body),
  archValidate: (body) => post('/api/arch/validate', body),

  trainStart: (body) => postGated('/api/train/start', body, 'launch this training run'),
  trainRetrain: (body) => postGated('/api/train/retrain', body, 'start this retraining run'),
  pause: (runId) => postGated(`/api/train/${runId}/pause`, {}, 'pause this run'),
  stop: (runId) => postGated(`/api/train/${runId}/stop`, {}, 'stop / kill this run'),
  resume: (runId, body) => postGated(`/api/train/${runId}/resume`, body || {}, 'resume this run'),
  whoami: () => fetch('/api/whoami', { cache: 'no-store' }).then(J),
  runs: () => fetch('/api/train/runs').then(J),
  run: (runId) => fetch(`/api/train/${runId}`).then(J),
  logs: (runId) => fetch(`/api/train/${runId}/logs`).then(J),
  series: (runId, key, mode = 'train') =>
    fetch(`/api/train/${runId}/series?key=${encodeURIComponent(key)}&mode=${mode}`).then(J),

  evalRun: (runId, body) => postGated(`/api/eval/${runId}`, body, 'run evaluation'),
  deployRun: (runId, body) => postGated(`/api/deploy/${runId}`, body, 'deploy this policy'),
  job: (jid) => fetch(`/api/jobs/${jid}`).then(J),

  epArrays: (ds, ep) => fetch(`/api/ds/${ds}/ep/${ep}/arrays`).then(J),
  frameUrl: (ds, ep, cam, i) => `/api/ds/${ds}/ep/${ep}/cam/${cam}/frame/${i}.jpg`,

  wsTrain(runId, onMsg) {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/train/${runId}`);
    ws.onmessage = (e) => { try { onMsg(JSON.parse(e.data)); } catch {} };
    return ws;
  },

  // poll a job until done/error
  async pollJob(jid, onProgress, interval = 400) {
    for (;;) {
      const j = await this.job(jid);
      if (onProgress) onProgress(j);
      if (j.status === 'done') return j.result;
      if (j.status === 'error') throw new Error(j.error || 'job failed');
      await new Promise(r => setTimeout(r, interval));
    }
  },
};
