// POV-transition screen recorder (?record=pov[&fps=60]) — deterministic:
// instead of sampling a live animation (which headless browsers throttle), it
// steps the SAME PovFx.applyStep() the UI uses at exact frame times, renders,
// composites the fpv overlay at its scheduled alpha, and posts each frame to
// the dev sink. Frames → mp4 via ffmpeg on the host. Because every frame is
// u = i / (fps·2s), the output is exactly the 2-second choreography.
export async function recordPov(cc, { fps = 60, holdS = 0.5 } = {}){
  const vp = cc.viewport, povfx = window.__povfx;
  const canvas = vp.stage.canvas;
  const fpvImg = document.getElementById('fpvImg');

  // same-origin head frame so compositing doesn't taint the canvas
  const head = new Image();
  await new Promise((res) => {
    head.onload = res; head.onerror = res;
    head.src = '/api/cam/main/frame.jpg?t=' + Math.random();
  });
  const haveHead = head.naturalWidth > 0;

  const comp = document.createElement('canvas');
  comp.width = canvas.width; comp.height = canvas.height;
  const cx = comp.getContext('2d');

  const drawFpv = (alpha) => {
    if (!alpha) return;
    cx.globalAlpha = alpha;
    cx.fillStyle = '#000';
    cx.fillRect(0, 0, comp.width, comp.height);
    if (haveHead){
      // object-fit: contain
      const s = Math.min(comp.width / head.naturalWidth, comp.height / head.naturalHeight);
      const w = head.naturalWidth * s, h = head.naturalHeight * s;
      cx.drawImage(head, (comp.width - w) / 2, (comp.height - h) / 2, w, h);
    }
    cx.globalAlpha = 1;
  };

  let n = 0;
  const post = async () => {
    const blob = await new Promise((r) => comp.toBlob(r, 'image/jpeg', 0.92));
    await fetch(`/api/dev/frame?name=pov&i=${n++}`, { method: 'POST', body: blob });
  };
  const snap = async (alpha) => {
    vp.stage.render();
    cx.globalAlpha = 1;
    cx.drawImage(canvas, 0, 0, comp.width, comp.height);
    drawFpv(alpha);
    await post();
  };

  const F = Math.round(fps * 2);              // 2 s of transition
  const H = Math.round(fps * holdS);

  // make sure the fpv <img> shows the same head frame the composite uses
  if (haveHead) fpvImg.src = head.src;

  // hold 3rd person
  for (let i = 0; i < H; i++) await snap(0);

  // 3rd → 1st
  povfx.begin('to1st');
  for (let i = 0; i <= F; i++){
    const { a } = povfx.applyStep(i / F);
    await snap(a);
  }
  povfx.finish();
  document.getElementById('viewport3d').classList.remove('pip-3d');   // record full-bleed

  // hold 1st person
  for (let i = 0; i < H; i++) await snap(1);

  // 1st → 3rd
  povfx.begin('to3rd');
  for (let i = 0; i <= F; i++){
    const { a } = povfx.applyStep(i / F);
    await snap(a);
  }
  povfx.finish();

  // hold 3rd person
  for (let i = 0; i < H; i++) await snap(0);

  document.title = `POV-RECORD-DONE ${n} frames`;
  return n;
}
