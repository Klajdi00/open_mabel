// Backend client. Episodes are addressed as (dataset, episodeFile-or-index).
const J = (r) => { if (!r.ok) throw new Error(r.status + ' ' + r.url); return r.json(); };
const B = async (r) => { if (!r.ok) throw new Error(r.status + ' ' + r.url); return new Uint8Array(await r.arrayBuffer()); };
const E = (s) => encodeURIComponent(s);

const base = (ds, ep) => `/api/ds/${E(ds)}/ep/${E(ep)}`;

export const api = {
  health: () => fetch('/api/health').then(J),
  catalog: () => fetch('/api/catalog').then(J),
  meta: (ds, ep) => fetch(`${base(ds, ep)}/meta`).then(J),
  arrays: (ds, ep) => fetch(`${base(ds, ep)}/arrays`).then(B),
  frame: (ds, ep, i) => fetch(`${base(ds, ep)}/frame/${i}`).then(J),
  cameras: (ds, ep) => fetch(`${base(ds, ep)}/cameras`).then(J),
  camFrameUrl: (ds, ep, name, i) => `${base(ds, ep)}/cam/${E(name)}/frame/${i}.jpg`,
  camMp4Url: (ds, ep, name) => `${base(ds, ep)}/cam/${E(name)}.mp4`,
  depthUrl: (ds, ep, name, i) => `${base(ds, ep)}/cam/${E(name)}/depth/${i}.png`,
  lidar: (ds, ep, s, e) => fetch(`${base(ds, ep)}/lidar.bin?start=${s}&end=${e}`).then(B),
  lidarCloud: (ds, ep, i) => fetch(`${base(ds, ep)}/lidar/cloud/${i}.bin`).then(B),
  camCloud: (ds, ep, name, i, stride = 4, maxPoints = 0) =>
    fetch(`${base(ds, ep)}/cam/${E(name)}/cloud/${i}.bin?stride=${stride}${maxPoints ? `&max_points=${maxPoints}` : ''}`).then(B),
  occupancy: (ds, ep) => fetch(`${base(ds, ep)}/occupancy.json`).then(r => r.ok ? r.json() : null),
  calibration: (ds, ep) => fetch(`${base(ds, ep)}/calibration`).then(J),
  extrinsics: (ds, ep, i) => fetch(`${base(ds, ep)}/extrinsics/${i}`).then(J),
  extrinsicsAll: (ds, ep) => fetch(`${base(ds, ep)}/extrinsics_all`).then(r => r.ok ? r.json() : null),
  scene: (ds, ep, voxel = 0.05, stride = 2) =>
    fetch(`${base(ds, ep)}/scene.bin?voxel=${voxel}&stride=${stride}`).then(B),
  recon: (ds, ep, source = 'head', kind = 'cloud', frames = 0, ps = 0, voxel = 0) =>
    fetch(`${base(ds, ep)}/recon.bin?source=${E(source)}&kind=${kind}`
      + `${frames ? `&frames=${frames}` : ''}${ps ? `&ps=${ps}` : ''}${voxel ? `&voxel=${voxel}` : ''}`).then(B),
  quality: (ds, ep) => fetch(`${base(ds, ep)}/quality.json`).then(J),
  frequency: (ds, ep) => fetch(`${base(ds, ep)}/frequency.json`).then(J),
  align: (ds, ep) => fetch(`${base(ds, ep)}/align.json`).then(J),
  datasetQuality: (ds) => fetch(`/api/ds/${E(ds)}/quality`).then(J),
  validate: (edl) => post('/api/edl/validate', edl),
  preview: (edl) => post('/api/edl/preview', edl),
  save: (body) => post('/api/save', body),
  exportEpisodes: (items, out_dataset, task) => post('/api/export/episodes', { items, out_dataset, task }),
  exportLeRobot: (body) => post('/api/export/lerobot', body),
  exportLeRobotDataset: (dataset, repo_id) => post('/api/export/lerobot/dataset', { dataset, repo_id }),
  job: (id) => fetch(`/api/jobs/${id}`).then(J),
  generate: (body) => post('/api/dev/generate-sample', body),
  putProject: (name, edl) => fetch(`/api/projects/${E(name)}`, {
    method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify(edl) }).then(J),
};

async function post(url, body){
  const r = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body) });
  return J(r);
}

// Poll a job until done/error.
export async function waitJob(id, onProg){
  for (;;){
    const j = await api.job(id);
    onProg?.(j);
    if (j.status === 'done') return j;
    if (j.status === 'error') throw new Error(j.error || 'job failed');
    await new Promise(r => setTimeout(r, 500));
  }
}
