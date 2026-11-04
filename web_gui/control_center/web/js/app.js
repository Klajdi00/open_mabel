// MABEL Control Center — orchestrator. Wires the four network planes
// (teleop :9090, monitor :9090/monitor, map :9092, our /ws/telemetry) into the
// studio shell: 3D viewer + teleop overlay, networking rail, status dock,
// 59-DOF joint band, floating camera windows, dexterity dock, choreography
// timeline, webcam full-body teleop.
import * as bus from './bus.js';
import { initHands } from './hands.js';
import { requireLogin } from './auth.js';
import { loadConfig } from './config.js';
import { CamView } from './camview.js';
import { RobotLink } from './link.js';
import { MonitorLink } from './monitor.js';
import { MapLink } from './maplink.js';
import { TelemetryLink } from './telemetry.js';
import { Viewport3D } from './viewport3d.js';
import { JointPanel } from './jointpanel.js';
import { NetPanel } from './netpanel.js';
import { SkillStore, SkillsPanel } from './skills.js';
import { StatusPanel } from './statuspanel.js';
import { CamStrip } from './camwall.js';
import { TeleopCtl } from './teleopctl.js';
import { BodyTeleop } from './bodyteleop.js';
import { Timeline } from './timeline.js';
import { PanelManager } from './panels.js';
import { initSplitters } from './splitters.js';
import { PovFx } from './povfx.js';
import { StreamLink } from './streamlink.js';
import { TelemetryHistory } from './history.js';
import { CamWindow, TfTreeWindow, Grid2DWindow, JointHistoryWindow,
         SimViewWindow } from './windows.js';
import { ModulePanel, ModuleRail } from './jointspace.js';
import { moduleById } from './modules.js';
import { HAND_JOINTS } from './handmodel.js';
import { ViewRail, RAIL_ITEMS } from './viewrail.js';
import { audioLink } from './audiolink.js';
import { SlamMenu } from './slammenu.js';
import { HistBand } from './histband.js';
import { REST_POSE, homeDuration, homePoseAt, isAtHome, homeError, liftHome } from './homepose.js';
// the ONE easing law for commanded motion in this studio (timeline blends,
// Home glides and now the task-space wrist plan all share it)
import { ease, blendMinDur, BLEND_THRESH } from './motions.js';

const $ = (id) => document.getElementById(id);

// ── shared control state (mirrored into the right dock) ─────────────────────
// `space` is the ONE control decision the operator makes — drive the base, move
// a wrist (task), or command joints directly (joint). `mode` is kept as its
// derived drive/manip shorthand because the pump (teleopctl) and the webcam
// path (bodyteleop) speak in those terms.
// `teleop` is TRUE from boot — there is no Start gate any more (visionOS
// parity). The controls for the selected space are live the moment the page
// opens: Drive shows its sticks, Joint shows its sliders.
//
// THIS IS NOT AN ENERGISE DECISION, and it must never become one. The robot
// comes up LIMP on purpose (MABEL_TORQUE_AUTO_ON=0 — bringing the server up
// must not power the arms) and only energises on an explicit e-stop-release
// edge from a client. That edge used to ride on "Start Teleop"; it now rides
// ONLY on the top bar's E-STOP / Enable Torque button, which is the operator
// SAYING "move". So a live stick on a freshly-loaded page commands a limp
// robot and nothing moves — which is exactly the intent. Do not "helpfully"
// send estop{active:false} on load to make the sticks work: that would
// energise the arms because someone opened a browser tab.
const ctl = {
  teleop: true, pov: 'third', space: 'drive', mode: 'drive', feel: 'stiff', grav: true,
  speedCaps: false,          // the gateway advertises live speed caps (hello.speedCaps)
  // base link liveness from robot_state.modules — see the robotState handler.
  // Defaults TRUE so a gateway that sends no roster never cries wolf.
  baseLive: true, baseDetail: '',
  region: 'arm', module: 'arm_right', stiffness: 0.6, head: false,
  // `torque` is the ROBOT's state, not a wish: the bridge comes up LIMP by design
  // (MABEL_TORQUE_AUTO_ON=0 — bringing the server up must not energise the arms)
  // and only enables torque on an explicit e-stop-release edge from a client. This
  // used to default to `true`, so the studio showed "⚡ Disable Torque" while the
  // hands were actually limp: every joint_command published fine and nothing moved.
  // It is now reflected from robot_state.torque (server-authoritative) — see below.
  estop: false, torque: false, recording: false, task: 'default',
  webcam: false,           // webcam body-teleop owns the wire (bodyteleop.js)
};

// robot_state.torque = {left_hand, right_hand, arms, base}; null = the bridge has
// not reported that module yet. "Energised" means every module that HAS reported
// is live — an unreported module is unknown, never assumed on.
const torqueKnown = (t) => Object.values(t || {}).filter((v) => v != null);
const torqueOn = (t) => {
  const v = torqueKnown(t);
  return v.length > 0 && v.every(Boolean);
};

// first-person head-cam surface — created on the first switch into 1st person
let fpvView = null;

function toast(msg, err = false){
  const t = $('toast');
  t.textContent = msg;
  t.className = 'toast show' + (err ? ' err' : '');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove('show'), 2600);
}
bus.on('toast', ({ msg, err }) => toast(msg, err));

async function main(){
  await loadConfig();
  initHands(document.getElementById('viewport3d'));   // which ORCA hands are mounted (v1 / v2)
  initSplitters();

  // ── links ── (?nolinks=1 boots the UI with no live planes, for debugging)
  const qp = new URLSearchParams(location.search);
  const link = new RobotLink();
  const monitor = new MonitorLink();
  const maps = new MapLink();
  const telemetry = new TelemetryLink();
  const streams = new StreamLink();
  if (!qp.get('nolinks')){
    link.connect(); monitor.connect(); maps.connect(); telemetry.connect();
  }
  // rolling joint history since boot (feeds the history windows)
  const history = new TelemetryHistory();
  bus.on('telemetrySchema', (s) => history.setSchema(s));
  bus.on('telemetry', (t) => history.push(t));

  // the task-space plan controller (defined after refreshOverlay); declared
  // here so the viewport's onJointTargets closure can route into it.
  let wristPlan = null;

  // ── viewport ──
  const viewport = new Viewport3D($('viewport3d'), {
    onNavGoal: (x, y, yaw) => { link.navGoal(x, y, yaw); toast(`nav goal → (${x.toFixed(2)}, ${y.toFixed(2)})`); },
    onNavPreview: (x, y, yaw) => link.navPreview(x, y, yaw),
    // Browser-solved joint targets from dragging the wrist ball / rings / sticks.
    // TASK SPACE is MoveIt-style: the solution POSES a preview and waits for
    // Execute — it never streams to the robot. Elsewhere (joint-space direct
    // manipulation) it commands live. A limp/soft/e-stopped robot takes nothing.
    onJointTargets: (m) => {
      if (ctl.estop || !ctl.torque || ctl.feel === 'soft') return;
      if (ctl.space === 'task'){ wristPlan?.stage(m); return; }
      link.joints(m);
    },
  });
  bus.on('robotState', (s) => viewport.updateRobot(s));
  // THE ROBOT IS THE SOURCE OF TRUTH for e-stop + torque. The bridge streams both
  // in robot_state precisely "so EVERY client can reflect them" — the studio used
  // to ignore that and just assume torque:true at boot, so the top bar read
  // "⚡ Disable Torque" (i.e. energised) while the actuators were actually LIMP.
  // Every joint_command then published happily and moved nothing, and the first
  // click of the Torque button sent estop{active:true} — the OPPOSITE of what the
  // operator wanted. Reflect, never assume; and only repaint on a real change
  // (robot_state arrives at 50 Hz).
  bus.on('robotState', (s) => {
    let dirty = false;
    if (typeof s?.estop === 'boolean' && s.estop !== ctl.estop){
      ctl.estop = s.estop;
      dirty = true;
    }
    if (s?.torque){
      const on = torqueOn(s.torque);
      if (torqueKnown(s.torque).length && on !== ctl.torque){
        ctl.torque = on;
        dirty = true;
      }
    }
    // THE WHEELS CAN GO DEAF WHILE EVERYTHING ELSE LOOKS HEALTHY. A base-link
    // hiccup (a carrier bounce on the base controller's lane) takes the base out for
    // SECONDS while this socket, robot_state, the joysticks and the gateway's
    // /mabel_cmd all keep running perfectly. From the operator's seat that is
    // "it drove for a second, then stopped" with nothing on screen to explain
    // it — the exact report this exists to answer. modules[base].ok is the
    // gateway's own measurement (module_status.py: a joint arrived on the base's
    // /joint_states within TTL), so reflect it, never guess.
    const baseMod = s?.modules?.find?.((m) => m.id === 'base');
    if (baseMod && baseMod.ok !== ctl.baseLive){
      ctl.baseLive = baseMod.ok;
      ctl.baseDetail = baseMod.detail || '';
      dirty = true;
    }
    // The keyboard hint explains WHY a live stick isn't moving the robot
    // ("LIMP — press Enable Torque"), so it has to repaint on the robot's own
    // torque edge, not only on a click of ours.
    if (dirty){ syncTopButtons(); refreshOverlay(); bus.emit('ctl'); }
  });
  bus.on('map:mesh', (r) => viewport.onMesh(r));
  bus.on('map:cloud', (r) => viewport.onCloud(r, 'live'));
  bus.on('map:grid', (r) => { if (!streams.grid) viewport.onGrid(r); });
  bus.on('map:path', (r) => viewport.onPath(r));
  bus.on('map:odom', (r) => viewport.onOdom(r));
  bus.on('opPoints', (r) => viewport.onOperatorPoints(r));
  // the opt-in is per-connection on the gateway: re-assert it after reconnect
  bus.on('hello', () => { if (viewport?.layers?.vptrack) link.operatorOverlay(true); });

  // MAX SPEED slider -> set_speed_caps (server/wire_protocol.py). Shown ONLY when the gateway
  // advertises live caps in its hello (speedCaps: the cap in force and the ceiling it clamps to,
  // vmax / vmaxCeil), so the slider's range is the robot's, not a guess — and a gateway without
  // live caps never gets a control that would silently do nothing.
  {
    const sl = $('speedSl'), out = $('speedOut');
    // A gateway-side tip-over governor (robot_state.base.stability.speed_max, when reported) can
    // hold translation BELOW the slider for the current pose and lift height; say so, or the
    // slider reads as a speed the robot never reaches.
    let tipCap = null;
    const show = (v) => {
      if (!out) return;
      const capped = tipCap != null && tipCap < +v;
      out.textContent = capped ? `${(+v).toFixed(2)} → ${tipCap.toFixed(2)} m/s` : `${(+v).toFixed(2)} m/s`;
      out.title = capped ? `tip-over governor: ${tipCap.toFixed(2)} m/s max in this pose / lift height` : '';
    };
    bus.on('robotState', (s) => {
      const c = s?.base?.stability?.speed_max;
      const next = Number.isFinite(c) ? c : null;
      if (next !== tipCap){ tipCap = next; if (sl) show(sl.value); }
    });
    if (sl){
      show(sl.value);
      sl.addEventListener('input', () => {
        show(sl.value);
        // yaw scales with translation, so the robot feels the same across the range
        link.send('set_speed_caps', { vmax: +sl.value, wmax: +(sl.value * 2).toFixed(3) });
      });
      bus.on('hello', (h) => {
        const c = h?.speedCaps;
        const had = ctl.speedCaps; ctl.speedCaps = !!c;
        if (had !== ctl.speedCaps) bus.emit('speedCaps', ctl.speedCaps);   // -> refreshOverlay
        if (!c) return;
        if (c.vmaxCeil) sl.max = c.vmaxCeil;          // same field the iOS and visionOS sliders read
        if (c.vmax && document.activeElement !== sl){ sl.value = c.vmax; show(c.vmax); }
      });
    }
  }
  bus.on('stream:cloud', (r) => viewport.onCloud(r, r.source));
  bus.on('stream:grid', (r) => viewport.onGrid(r));
  bus.on('telemetry', (t) => { if (t.scan) viewport.updateLidar(t.scan); });
  // (camera-strip tiles and 3D projections are independent: the Cam 3D ▾
  // dropdown owns which cameras project into the scene). Default = the head
  // ZED projection + the RealSense on the mobile base ('main' falls back to
  // the front stream while the ZED is offline).
  viewport.setActiveCams(new Set(['front', 'main']));

  // ── layers: WHAT is drawn. The rail (js/viewrail.js) owns where these sit on
  // screen and which of them expand; this map stays the source of truth for
  // their DEFAULT state. The keys must match RAIL_GROUPS — a node test fails if
  // the two ever drift apart, so a layer can never go unreachable.
  const LAYER_DEFAULTS = {
    robot: true, grid: true, tf: false, lidar: false, cloud: true,
    mesh: true, occupancy: false, path: true, navgoal: true, wrist: false,
    vptrack: false, frustum: true, traj: false,
  };
  const CAM_STREAMS = [
    ['front', 'front base (realsense)'],
    ['main', 'head (zed)'],
    ['wrist_left', 'wrist L'],
    ['wrist_right', 'wrist R'],
  ];
  let gridSource = null;                 // null = the bridge's map plane feed
  const MENU_ITEMS = {
    cams: () => [
      ...CAM_STREAMS.map(([n, label]) => {
        // hardware truth: the head slot is only offered when the ZED is real
        // ('main' falls back to the front cam, so probe head_left instead)
        const live = camstrip?.live || {};
        const dead = (n === 'main' ? live.head_left : live[n]) === false;
        return {
          label: dead ? label + ' — offline' : label,
          on: viewport.camOn(n),
          toggle: (on) => {
            if (dead && on) { toast(label + ' is not connected', true); return; }
            viewport.toggleCam(n, on);
            if (on) viewport.setLayer('frustum', true);
          },
        };
      }),
      { slider: { label: 'image size', min: 0.1, max: 1.0, step: 0.02,
        value: viewport.camProjSize, fmt: (v) => v.toFixed(2) + 'm',
        cb: (v) => viewport.setCamProjDistance(v) } },
      { label: 'undistort (rectified)', on: viewport.camFlat,
        toggle: (on) => viewport.setCamFlat(on) },
    ],
    tf: () => viewport.tfFrames().map((f) => ({
      label: f, on: viewport.tfOn(f),
      toggle: (on) => viewport.setTfFrame(f, on),
    })),
    cloud: () => [
      { label: 'live (SLAM feed)', on: viewport.cloudSourceOn('live'),
        toggle: (on) => viewport.setCloudSource('live', on) },
      { label: 'head depth', on: viewport.cloudSourceOn('head'),
        toggle: (on) => { viewport.setCloudSource('head', on); streams.setCloud('head', on); } },
      { label: 'front base depth', on: viewport.cloudSourceOn('front'),
        toggle: (on) => { viewport.setCloudSource('front', on); streams.setCloud('front', on); } },
      { slider: { label: 'point size', min: 0.005, max: 0.08, step: 0.005,
        value: viewport.pointSize, fmt: (v) => v.toFixed(3),
        cb: (v) => viewport.setPointSize(v) } },
    ],
    traj: () => [
      ...['left wrist', 'right wrist', 'head', 'odom'].map((k) => ({
        label: k, on: viewport.trajOn(k),
        toggle: (on) => { viewport.setTrajectory(k, on); if (on) viewport.setLayer('traj', true); },
      })),
      { slider: { label: 'window', min: 30, max: 300, step: 10,
        value: viewport.traj?.windowS ?? 120, fmt: (v) => v + 's',
        cb: (v) => viewport.setTrajWindow(v) } },
    ],
    grid: () => [
      { label: 'lidar SLAM', on: gridSource === 'lidar',
        toggle: (on) => { gridSource = on ? 'lidar' : null; streams.setGrid(gridSource);
          if (on) viewport.setLayer('occupancy', true); } },
      { label: 'visual SLAM', on: gridSource === 'visual',
        toggle: (on) => { gridSource = on ? 'visual' : null; streams.setGrid(gridSource);
          if (on) viewport.setLayer('occupancy', true); } },
    ],
  };
  // ?layers=robot,grid overrides the defaults (handy for e2e screenshots)
  const layerParam = new URLSearchParams(location.search).get('layers');
  const forced = layerParam ? new Set(layerParam.split(',')) : null;
  for (const key in LAYER_DEFAULTS)
    viewport.setLayer(key, forced ? forced.has(key) : LAYER_DEFAULTS[key]);
  // Default cloud = the head depth camera (colored by the sensor), streamed
  // from our backend's /ws/stream; the SLAM 'live' overlay stays opt-in via
  // the Cloud ▾ menu. Skipped under ?layers= so e2e screenshots stay minimal.
  if (!forced){
    viewport.setCloudSource('live', false);
    viewport.setCloudSource('head', true);
    streams.setCloud('head', true);
  }

  // Render one expandable entry's sub-options into the rail's flyout. Same
  // item vocabulary the old chip dropdowns used (toggles + sliders) — only the
  // surface moved, so MENU_ITEMS above needs no per-rail special-casing.
  //
  // `overlays` is the synthetic one: on a short viewport the rail collapses to
  // Center · Fit · Follow · Overlays▸, and THIS renders every layer plus each
  // expandable's sub-options into that single panel — the visionOS layers
  // popover, which is where the pattern comes from.
  const overlayItems = () => {
    const out = [];
    for (const it of RAIL_ITEMS){
      if (!it.layer) continue;
      out.push({ label: it.label, on: !!viewport.layers[it.layer],
        toggle: (on) => viewport.setLayer(it.layer, on) });
    }
    for (const it of RAIL_ITEMS){
      if (!it.menu) continue;
      out.push({ section: it.label });
      out.push(...MENU_ITEMS[it.menu]());
    }
    return out;
  };
  const buildRailMenu = (kind, host, onChange) => {
    const items = () => (kind === 'overlays' ? overlayItems() : MENU_ITEMS[kind]());
    const rebuild = () => {
      host.innerHTML = '';
      for (const it of items()){
        if (it.section){
          host.appendChild(Object.assign(document.createElement('div'),
            { className: 'rf-sec', textContent: it.section }));
          continue;
        }
        if (it.slider){
          const row = document.createElement('div');
          row.className = 'vp-mslider';
          row.innerHTML = `<span>${it.slider.label}</span>
            <input type="range" min="${it.slider.min}" max="${it.slider.max}"
                   step="${it.slider.step}" value="${it.slider.value}">
            <b class="mono">${it.slider.fmt(it.slider.value)}</b>`;
          const inp = row.querySelector('input'), val = row.querySelector('b');
          inp.oninput = () => { it.slider.cb(+inp.value); val.textContent = it.slider.fmt(+inp.value); };
          // the flyout closes on outside pointerdown — dragging a slider is
          // not "outside", so stop it reaching that listener
          row.addEventListener('pointerdown', (e) => e.stopPropagation());
          host.appendChild(row);
          continue;
        }
        const b = document.createElement('button');
        b.className = 'vp-mi' + (it.on ? ' on' : '');
        b.innerHTML = `<span class="vmi-dot"></span>${it.label}`;
        b.onclick = () => { it.toggle(!it.on); rebuild(); onChange(); };
        host.appendChild(b);
      }
    };
    rebuild();
  };

  // ── VOICE (the :9610 plane) — the web twin of the iOS speak/mute pair.
  // audioLink owns the wire; the rail owns the two buttons. Availability is
  // event-driven: in same-origin (https) mode, or on a rig with no sound card,
  // the buttons DIM and say why rather than pretending to work (the house rule).
  let audioReady = false, listening = false;

  const viewRail = new ViewRail($('vpRail'), {
    isOn: (it) => it.layer ? !!viewport.layers[it.layer]
      : it.action === 'follow' ? !!viewport.follow
      : it.action === 'listen' ? listening : false,
    setLayer: (k, on) => {
      viewport.setLayer(k, on);
      // the VP-tracking overlay is server-fed: opt this client in/out so the
      // gateway only streams operator points to tabs actually showing them
      if (k === 'vptrack') link.operatorOverlay(on);
    },
    action: (id, on) => {
      if (id === 'recenter') viewport.recenter();
      else if (id === 'fit') viewport.frameRobot();
      else if (id === 'follow'){
        viewport.setFollow(on);
        toast(on ? 'orbiting the head cam — rotate/zoom only' : 'follow off');
      } else if (id === 'listen'){
        if (!audioReady){ toast('no robot audio on this link', true); return; }
        listening = on;
        // startListening resumes the AudioContext — it needs a user gesture, and
        // this click IS one, so hearing the robot begins on the first unmute.
        if (on){ audioLink.startListening(); toast('listening to the robot'); }
        else { audioLink.stopListening(); toast('robot audio muted'); }
      }
    },
    // hold = push-to-talk: mic open only while pressed. Also starts listening so
    // it is a full intercom, and never leaves a hot mic if the link is not ready.
    hold: (id, down) => {
      if (id !== 'speak') return;
      if (down){
        if (!audioReady){ toast('no robot audio on this link', true); return; }
        if (!listening){ listening = true; audioLink.startListening(); viewRail.sync(); }
        audioLink.startTalking();
      } else {
        audioLink.stopTalking();
      }
    },
    buildMenu: buildRailMenu,
  });
  viewRail.sync();
  // Voice wiring lives AFTER the rail exists (its buttons are referenced here).
  audioLink.connect();
  bus.on('audio:state', (st) => {
    audioReady = !!st.available;
    if (viewRail.btns.speak)  viewRail.btns.speak.disabled  = !audioReady;
    if (viewRail.btns.listen) viewRail.btns.listen.disabled = !audioReady;
    if (!audioReady){ listening = false; viewRail.sync(); }
  });
  bus.on('audio:talking', (on) => viewRail.btns.speak?.classList.toggle('on', on));
  // ── NAV GOAL: click the floor, no arming ────────────────────────────────
  // The old flow made you press "Nav" to arm a cursor before the floor would
  // take a goal. The apps never did that — iOS's header says it plainly: "tap
  // the floor to send a Nav2 goal". So the floor is ALWAYS live: a tap places
  // the goal (drag still orbits the camera — viewport3d tells them apart), the
  // path previews, and the GO bar appears as the consequence. Dragging the
  // placed marker re-aims it, exactly as before.
  const goalBar = $('goalBar'), goBtn = $('goBtn'), goalWhere = $('goalWhere');
  let navigating = false;
  const showGoalBar = (on) => goalBar.classList.toggle('hidden', !on);
  const clearNav = () => {
    link.send('nav_cancel', {});
    viewport.clearNavGoal();
    navigating = false;
    showGoalBar(false);
  };
  viewport.onGoalPlaced = (g) => {
    if (!g){ showGoalBar(false); return; }
    navigating = false;
    goBtn.className = 'vp-chip go';
    goBtn.innerHTML = '➤&nbsp;GO';
    goBtn.title = 'Drive to the placed goal';
    goalWhere.textContent = `${g.x.toFixed(2)}, ${g.y.toFixed(2)}`;
    showGoalBar(true);
  };
  goBtn.onclick = () => {
    if (navigating){                       // second press = STOP NAV
      clearNav();
      toast('navigation stopped');
      return;
    }
    const g = viewport.getGoal();
    if (!g){ showGoalBar(false); return; }
    link.navGoal(g.x, g.y, g.yaw);
    navigating = true;
    goBtn.className = 'vp-chip stopnav';
    goBtn.innerHTML = '■&nbsp;STOP&nbsp;NAV';
    goBtn.title = 'Cancel the drive';
    toast(`GO → (${g.x.toFixed(2)}, ${g.y.toFixed(2)})`);
  };
  $('goalClear').onclick = () => { clearNav(); toast('nav target cleared'); };

  // ── SLAM: live mapping · saved rooms · reset — one top-right menu ────────
  // The saved rooms come from the GATEWAY (server/map_library.py over the
  // teleop wire), which is the same store iOS and Vision Pro read — save a room
  // on the phone and it is here, and vice versa. Only pause/reset are this
  // backend's job, because they act on the running SLAM node.
  const slam = new SlamMenu($('slamBtn'), $('slamMenu'), link, toast, {
    onFreeze: (frozen) => viewport.setMapFrozen(frozen),
    onClear: () => viewport.clearMap(),
  });
  bus.on('mapList', (m) => slam.onMapList(m));

  // (point-size / projection-size / undistort adjusters live INSIDE the
  // Cloud ▾ and Cam 3D ▾ dropdowns — the viewer stays clean)
  $('vpCamCtl').classList.add('hidden');

  // ── right-dock telemetry: joint state + trends (moved off the bottom band) ──
  // The joint-state bars and the 2-minute history graph now live in the Status
  // panel as collapsible sections; the bottom of the viewer keeps only cameras.
  const jointPanel = new JointPanel($('jointCanvas'), $('plotsHint'));
  bus.on('telemetrySchema', (s) => {
    jointPanel.setSchema(s);
    const t = document.getElementById('jsDof');   // the title follows the server's own DOF count
    if (t && s?.dof) t.textContent = `Joint State · ${s.dof} DOF`;
  });
  bus.on('telemetry', (t) => jointPanel.update(t));
  $('plotMode').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    e.stopPropagation();                    // the pos/vel/torque seg is not a collapse
    $('plotMode').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    jointPanel.setMode(b.dataset.m);
  });
  const histBand = new HistBand($('histBand'), history);
  bus.on('telemetrySchema', (s) => histBand.setSchema(s));

  // Each section shows/hides on a header click (the operator's toggle). Both
  // start open; collapsing pauses the trends draw. `studio-resize` re-lays the
  // canvases when a section opens.
  const secShown = { joints: true, trends: true };
  const applySections = () => {
    $('jointSec').classList.toggle('collapsed', !secShown.joints);
    $('trendsSec').classList.toggle('collapsed', !secShown.trends);
    histBand.show(secShown.trends);
    window.dispatchEvent(new Event('studio-resize'));
  };
  document.querySelectorAll('.dt-head').forEach((h) => {
    h.addEventListener('click', (e) => {
      if (e.target.closest('.dt-mode')) return;    // clicking pos/vel/torque isn't a collapse
      const k = h.dataset.sec;
      secShown[k] = !secShown[k];
      applySections();
    });
  });
  applySections();

  // The bottom band is CAMERAS ONLY now — size it to the strip (an inline
  // custom property outranks the media-query defaults, so it is computed here).
  const applyCamBand = () => {
    const compact = window.matchMedia('(max-width: 900px)').matches;
    document.querySelector('.studio').style.setProperty('--plots-h',
      (compact ? 104 : 138) + 'px');
    window.dispatchEvent(new Event('studio-resize'));
  };
  applyCamBand();
  window.matchMedia('(max-width: 900px)').addEventListener('change', applyCamBand);

  // ── collapsible panels: network rail · status dock · timeline ─────────────
  // Each panel folds from its own header and comes back from the stub it left
  // on that edge, so there is no separate toggle bar to keep in sync. Persisted,
  // because a layout you rebuild on every reload is not a layout.
  const PANES_LS = 'mabel.studio.panes';
  const studio = document.querySelector('.studio');
  const hidden = Object.assign({ left: false, right: false, tl: false },
    JSON.parse(localStorage.getItem(PANES_LS) || '{}'));
  const applyPanes = () => {
    studio.classList.toggle('hide-left', !!hidden.left);
    studio.classList.toggle('hide-right', !!hidden.right);
    studio.classList.toggle('hide-tl', !!hidden.tl);
    localStorage.setItem(PANES_LS, JSON.stringify(hidden));
    // the 3-D canvas and the plots own explicit pixel sizes — tell them the
    // grid changed or they keep rendering at the old width
    window.dispatchEvent(new Event('resize'));
    window.dispatchEvent(new Event('studio-resize'));
  };
  document.querySelectorAll('[data-hide]').forEach((b) => b.onclick = () => {
    hidden[b.dataset.hide] = true; applyPanes();
  });
  document.querySelectorAll('[data-show]').forEach((b) => b.onclick = () => {
    hidden[b.dataset.show] = false; applyPanes();
  });
  applyPanes();

  new NetPanel($('netBody'));
  // The left rail is NETWORK only. The Library tab is gone: the clip library it
  // showed is already reachable from the timeline's own + menus and the Skills
  // dock pane, so it was a third door onto the same content — and it stole the
  // rail out from under the operator whenever they touched the timeline.
  $('netRefresh').onclick = async () => {
    const s = await fetch('/api/system').then(r => r.json()).catch(() => null);
    if (s) bus.emit('system', s);
  };
  const statusPanel = new StatusPanel($('statusPane'), $('dataPane'), ctl);

  // dock tabs
  document.querySelectorAll('.dock-tab').forEach(t => t.onclick = () => {
    document.querySelectorAll('.dock-tab').forEach(x => x.classList.toggle('on', x === t));
    document.querySelectorAll('.dock-pane').forEach(p =>
      p.classList.toggle('on', p.dataset.pane === t.dataset.tab));
  });

  // ── permanent camera strip (above the joint band) + floating windows ──
  const camstrip = new CamStrip($('camStrip'));
  const panels = new PanelManager($('viewport3d'));

  // (The hands are no longer a dock of their own: a hand is a MODULE, and its
  // ORCA controls live in the module panel — see the module rail below.)

  panels.register('cam', 'Camera', (panel) => new CamWindow(panel));
  panels.register('tftree', 'TF Tree', (panel) => new TfTreeWindow(panel, viewport));
  panels.register('grid2d', '2D Map', (panel) => new Grid2DWindow(panel));
  panels.register('jointhist', 'Joint History',
    (panel) => new JointHistoryWindow(panel, history,
      telemetry.schema || { names: [], modules: [] }));
  panels.register('simview', 'MuJoCo Viewer', (panel) => new SimViewWindow(panel));
  panels.register('previewview', 'PHYSICS PREVIEW',
    (panel) => new SimViewWindow(panel, { instance: 'preview', title: 'PHYSICS PREVIEW' }));
  // (no windows auto-open — extra views come from + Window)
  $('addWinBtn').onclick = () => {
    const m = $('addWinMenu');
    m.classList.toggle('hidden');
    if (!m.childElementCount){
      for (const [type, label] of [['simview', 'MuJoCo viewer (robot mirror)'],
        ['cam', 'Camera view'],
        ['jointhist', 'Joint history'], ['grid2d', '2D occupancy map'],
        ['tftree', 'TF tree']]){
        const b = document.createElement('button');
        b.className = 'vp-mi'; b.innerHTML = `<span class="vmi-dot"></span>${label}`;
        b.onclick = () => { panels.add(type); m.classList.add('hidden'); };
        m.appendChild(b);
      }
    }
  };

  // ── teleop overlay ──
  const teleopCtl = new TeleopCtl(link, ctl, viewport);
  teleopCtl.attachSticks($('stickL'), $('stickR'), $('udL'), $('udR'));

  const seg = (id, cb) => {
    $(id).addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      $(id).querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
      cb(b.dataset.v);
    });
  };

  // THE WIRE CONTRACT (server/wire_protocol.py) — two DIFFERENT axes that were
  // being sent in each other's slot:
  //   method      = TeleopMethod — how intent becomes a reference
  //                 (navigation = base twist+lift · joint = per-joint angles)
  //   controlType = ControlType  — how the joints realise it
  //                 (impedance = stiff/spring-to-target · compliance = soft/yield)
  //
  // We had them SWAPPED: 'impedance' went out as the METHOD, which never matches
  // the facade's `_WBC_METHODS = ("joint", "wrist")` (controller/mabel/facade.py),
  // so the torque engine NEVER engaged for this client and joint-space teleop
  // moved nothing on the real robot — while `robot_state` kept streaming, which
  // is why the joints read live but would not drive. Symmetrically, 'compliance'
  // never reached the controlType slot, so SOFT never actually went soft.
  //
  // BOTH manipulate spaces send per-joint ANGLES, so both are method='joint':
  // task space solves the wrist IK in the browser (viewport3d._solveArm →
  // onJointTargets → link.joints), so the server receives joint targets either
  // way. 'wrist' would tell it to run its own palm IK from an SE(3) target this
  // client never sends. This is exactly iOS's applySpaceSpec()/setNavigation().

  // ── HOLD THE MEASURED POSE when joint control engages ─────────────────────
  // Declaring method='joint' hands every WBC-managed joint to this client's
  // targets. The arm/torso/lift targets arrive from the IK drag, but NOTHING in
  // this UI ever names the 17 hand joints or the wrist unless the operator opens
  // that module — so they defaulted to 0, and the WBC dutifully sprang there.
  // Zero is NOT neutral for the ORCA hand: those joints carry a non-zero MJCF
  // `ref`, so "0" is a splayed, contorted hand (which is exactly what it looked
  // like). Seed them from the MEASURED pose first and the hand simply stays
  // where it is — the client-side twin of the bridge's facade.resume_hold.
  const HAND_SUFFIXES = HAND_JOINTS.map(([n]) => n);
  const holdHandsAtMeasured = () => {
    const jp = link.state?.jointPositions;
    if (!jp) return;                       // no telemetry yet — nothing to hold
    const hold = {};
    for (const side of ['left', 'right'])
      for (const suf of HAND_SUFFIXES){
        const n = `${side}_${suf}`;
        if (Number.isFinite(jp[n])) hold[n] = jp[n];
      }
    if (Object.keys(hold).length) link.joints(hold);
  };

  const applyControlSpec = () => {
    // WEBCAM TELEOP OWNS THE METHOD. While body tracking drives the wire the
    // server must be in `retarget` — that is the engine that turns teleop_frame
    // envelopes into arm/neck/finger motion. Deriving the method from ctl.space
    // alone meant every later call here (IK reach, Stiff/Soft, a space tap)
    // quietly re-declared 'joint' underneath a live webcam session: the frames
    // kept streaming, the skeleton kept tracking, and the robot did nothing —
    // "the webcam tracks fine but it is not retargeted to the robot".
    const method = ctl.webcam ? 'retarget'
                 : (ctl.space === 'drive' ? 'navigation' : 'joint');
    const type = ctl.feel === 'stiff' ? 'impedance' : 'compliance';
    // seed first: the hand targets must exist BEFORE joint mode engages
    if (method === 'joint' && type === 'impedance') holdHandsAtMeasured();
    link.controlMode(method, type, ctl.region, ctl.stiffness * 45.0, ctl.grav);
    bus.emit('ctl');
  };

  // ── the MANIPULATE surface: a module rail + the focused module's controls ──
  // The rail is the single "what am I working on" selector, in BOTH manipulate
  // spaces (iOS parity). A hand is just a module — the old hand dock is gone.
  const modulePanel = new ModulePanel($('vpJoints'), link,
    () => link.state?.jointPositions);
  bus.on('viewportReady', () => {
    modulePanel.build(viewport.manifest);
    modulePanel.setModule(ctl.module, true);
  });

  const setModule = (id) => {
    ctl.module = id;
    modulePanel.setModule(id);
    // the rail also picks the side the sticks / arrow keys drive, so "[ ]" and
    // the rail can never disagree about which wrist is live
    const side = moduleById(id)?.side;
    if (side) teleopCtl.activeSide = side;
    viewport.focusModule?.(id);
    refreshOverlay();
  };
  const modRail = new ModuleRail($('modRail'), setModule);
  modRail.select(ctl.module, true);
  // Grabbing a green wrist ball IS the module choice in Task space (where the
  // rail is hidden): it keeps ctl.module, the sticks' active side and the
  // Status panel agreeing with the arm you are actually dragging. Camera-flying
  // to the module here would fight the drag, so select without the focus flight.
  bus.on('wristGrab', (side) => {
    const id = side === 'left' ? 'arm_left' : 'arm_right';
    if (ctl.module === id) return;
    ctl.module = id;
    teleopCtl.activeSide = side;
    modRail.select(id, true);
    bus.emit('ctl');
  });
  // keyboard "[ ]" side-switch keeps the rail in step (teleopctl owns the keys)
  bus.on('sideSwitch', (side) => {
    const m = moduleById(ctl.module);
    if (!m?.side || m.side === side) return;
    const twin = m.id.replace(m.side, side);      // arm_left ⇄ arm_right, hand_ ⇄
    if (moduleById(twin)){ modRail.select(twin); }
  });

  const refreshOverlay = () => {
    // ENERGISED is the one gate (iOS parity): until the operator presses Torque
    // ON, the robot is limp and the studio is a pure VISUALISER — the space
    // picker, Stiff|Soft, the joysticks and the manipulate surface are all
    // hidden. Pressing Torque ON energises the robot AND reveals them; the red
    // E-STOP folds them away again. `live` additionally respects Soft.
    const energized = ctl.torque && !ctl.estop;
    const kb = energized && !ctl.webcam;
    const live = kb && ctl.feel === 'stiff';
    const manip = ctl.space !== 'drive';
    modulePanel.build(viewport.manifest);

    // the two top-bar control pickers appear only once energised
    $('spaceSeg').classList.toggle('hidden', !kb);
    $('feelSeg').classList.toggle('hidden', !energized);
    // Gravity comp appears with Stiff/Soft: it is only meaningful once the arm is energised.
    $('gravSeg')?.classList.toggle('hidden', !energized);

    // The rail is JOINT space's anchor — it is what picks whose sliders show.
    // Task space does NOT get it: there the control input is the green wrist
    // balls in the 3-D view, and which ball you grab already says which arm you
    // mean (viewport3d emits `wristGrab` → setModule below). A rail there was a
    // second, redundant way to say the same thing, sitting on top of the view
    // you are trying to reach into.
    modRail.show(kb && ctl.space === 'joint');
    modulePanel.show(kb && manip && ctl.space === 'joint');
    // Task space = the wrist sticks; Joint space = the panel (never both).
    const sticks = live && ctl.space !== 'joint';
    const wrist = live && ctl.space === 'task';
    // the keyboard hint keeps to the corridor between the sticks when they are
    // up (CSS reads this class) — long hint text otherwise runs under a thumb
    $('viewport3d').classList.toggle('sticks-live', !!sticks);
    $('stickL').classList.toggle('hidden', !sticks);
    $('stickR').classList.toggle('hidden', !sticks);
    $('udL')?.classList.toggle('hidden', !wrist);
    $('udR')?.classList.toggle('hidden', !wrist);
    const drive = ctl.space === 'drive';
    $('stickL').querySelector('.stick-tag').textContent = drive ? 'TRANSLATE' : 'L WRIST XY';
    $('stickR').querySelector('.stick-tag').textContent = drive ? 'ROTATE · LIFT' : 'R WRIST XY';
    // Task space owns BOTH the IK reach and the webcam-teleop start — inert in
    // Drive and Joint, and meaningless while limp.
    $('vpManip').classList.toggle('hidden', !((kb && ctl.space === 'task') || ctl.webcam));
    $('kbdHint').classList.toggle('hidden', false);
    // The speed cap belongs to Drive space: it is the operator's own limit, sent to the gateway,
    // which clamps it to the robot's ceiling. Hidden elsewhere, where it commands nothing.
    $('speedCap')?.classList.toggle('hidden', !(sticks && drive && ctl.speedCaps));
    $('kbdHint').classList.toggle('fault', drive && energized && !ctl.baseLive);
    // The floor takes a nav goal only where navigating is what you are doing:
    // Drive space, 3rd person, energised. (In 1st person there is no floor.)
    viewport.navTapEnabled = energized && drive && ctl.pov === 'third';
    // The hint doubles as the "why isn't it moving?" answer.
    $('kbdHint').textContent = !energized
      ? (ctl.estop
        ? 'E-STOP engaged — press Torque ON to re-energise and control the robot'
        : 'VIEWER — press ⚡ Torque ON (top bar) to energise and reveal the controls')
      : ctl.feel === 'soft'
        ? 'SOFT — the app is a viewer; switch to Stiff to command'
        : (drive && !ctl.baseLive)
          // `detail` is the gateway's own count ("4/6", "link down"), so say that rather than guess.
          ? `BASE ${ctl.baseDetail || 'offline'} — the sticks command, but not every wheel is answering`
        : drive
          ? 'WASD drive · QE turn · RF lift · click the floor to set a nav goal'
          : ctl.space === 'task'
            ? 'drag the wrist ball to plan · then press Execute · [ ] switch side'
            : 'sliders command the selected module · [ ] switch side';
    bus.emit('ctl');
  };
  bus.on('speedCaps', () => refreshOverlay());

  // ── task-space PLAN → PREVIEW → EXECUTE (MoveIt-style) ────────────────────
  // Dragging the wrist ball solves IK and POSES the rig (the preview), but the
  // joints go into a pending plan instead of the wire. The plan bar is the
  // consequence — Execute runs it on the robot, Cancel drops it. This mirrors
  // the nav GO/STOP bar; on MABEL nothing streams to the arm until you mean it.
  wristPlan = (() => {
    const bar = $('planBar'), where = $('planWhere');
    let pending = {};
    let runner = null;                           // the eased-execute interval
    const count = () => Object.keys(pending).length;
    const show = (on) => bar.classList.toggle('hidden', !on);
    const stopRun = () => { if (runner){ clearInterval(runner); runner = null; } };

    return {
      stage(m){
        Object.assign(pending, m);
        viewport.setPlanHold(m);                 // pin the preview against live
        where.textContent = `${count()} joints`;
        show(true);
      },
      hasPlan(){ return count() > 0; },
      // EASED EXECUTE. `link.joints(pending)` used to send the whole delta in
      // ONE envelope: the arm then moved as fast as the actuators allow, which
      // is the same pose-jump hazard motions.js exists to prevent on the
      // timeline (measured 29 rad/s there against a 2.5 rad/s budget). A wrist
      // plan is a bigger delta than a timeline gap, and it is aimed at whatever
      // the operator was reaching for. So stream it: a raised cosine from the
      // LIVE pose to the plan, sized by motions.js `blendMinDur` — literally the
      // same function the timeline sizes its blends with, so there is ONE
      // velocity budget in the studio — at 20 Hz, the executor's tick rate.
      execute(){
        if (!count() || runner) return;
        const start = viewport.liveJoints();
        const target = pending;
        // The COLLISION-AVOIDED waypoint path staged by the last wrist solve
        // (viewport3d: CBF flow around the torso/head/other arm). Executing
        // along it makes the robot perform exactly what the ghost previewed —
        // a straight joint-space lerp to the same endpoint could sweep the
        // elbow through the torso on the way. Joints in the plan but not on
        // the path (the other arm, grips) keep the straight eased lerp. The
        // server re-filters every target through the same CBF regardless —
        // this just keeps the preview honest.
        const path = viewport.planPathJoints();
        const sample = (u) => {
          if (!path || path.length < 2) return null;
          const s = u * (path.length - 1);
          const i = Math.min(path.length - 2, Math.floor(s));
          const f = s - i;
          const out = {};
          for (const j in path[i]) out[j] = path[i][j] + (path[i + 1][j] - path[i][j]) * f;
          return out;
        };
        let d = 0;
        for (const j in target){
          const a = start[j];
          if (a != null) d = Math.max(d, Math.abs(target[j] - a));
        }
        const n = count();
        pending = {};
        show(false);
        viewport.clearPlanHold();                // let the rig track the motion
        // no measurable delta, or no live pose to ease from → send it straight
        if (!(d > BLEND_THRESH)){
          link.joints(target);
          toast(`executing wrist plan — ${n} joints`);
          return;
        }
        // blendMinDur IS this law (EASE_PEAK·Δ / SAFE_BLEND_VEL, floored and
        // ceilinged) and it is already unit-tested — call it rather than inline a
        // second copy that can drift out of agreement with the timeline's.
        const dur = blendMinDur(d);
        toast(`executing wrist plan — ${n} joints over ${dur.toFixed(1)}s`);
        const t0 = performance.now();
        // ARRIVAL-GATED (2026-07-29, same fix as the Home glide): the eased
        // playback is only the MINIMUM — a torque/speed-limited arm lags the
        // ease and used to be left mid-way when the clock expired. After
        // playback a SETTLE phase keeps the FINAL target on the wire until the
        // measured joints arrive or stop progressing. No time cap; Cancel and
        // the e-stop paths still stop it dead.
        const ARRIVE_RAD = 0.12, STALL_EPS = 0.01, STALL_S = 2.5;
        let bestErr = Infinity, bestAt = performance.now();
        runner = setInterval(() => {
          const u = Math.min(1, (performance.now() - t0) / 1000 / dur);
          const k = ease(u);
          const onPath = sample(k);
          const q = {};
          for (const j in target){
            if (onPath && j in onPath){ q[j] = onPath[j]; continue; }
            const a = start[j];
            q[j] = a == null ? target[j] : a + (target[j] - a) * k;
          }
          link.joints(q);
          if (u < 1) return;
          const live = viewport.liveJoints();
          let err = 0;
          for (const j in target)
            if (live[j] != null) err = Math.max(err, Math.abs(target[j] - live[j]));
          const now = performance.now();
          if (err < bestErr - STALL_EPS){ bestErr = err; bestAt = now; }
          const arrived = err < ARRIVE_RAD;
          const stalled = now - bestAt > STALL_S * 1000;
          if (arrived || stalled){
            stopRun();
            if (!arrived){
              // UNREACHABLE → ABANDON, don't push forever: stopping the stream
              // alone leaves the target LATCHED server-side and the servo keeps
              // a bounded shove into whatever is blocking it. Rebase the plan's
              // joints onto the measured pose so the arm rests where it is.
              const rest = {};
              for (const j in target) if (live[j] != null) rest[j] = live[j];
              if (Object.keys(rest).length) link.joints(rest);
              toast(`wrist plan abandoned ${(err * 57.3).toFixed(0)}° short (unreachable)`);
            }
          }
        }, 50);
      },
      running(){ return !!runner; },
      // silent = leaving Task space (no toast); otherwise the operator cancelled
      cancel(silent){
        // An eased run is live motion on the real arm — cancelling must STOP it
        // where it is, not merely hide the bar and let it keep streaming to the
        // target the operator just abandoned. Also REBASE the latched targets
        // onto the measured pose: the last streamed envelope otherwise keeps a
        // bounded servo pull toward the abandoned goal.
        const wasRunning = !!runner;
        stopRun();
        if (wasRunning){
          const live = viewport.liveJoints();
          if (live && Object.keys(live).length) link.joints(live);
        }
        if (!count() && silent && !wasRunning){ show(false); return; }
        pending = {};
        viewport.clearPlanHold();
        show(false);
        if (!silent) toast(wasRunning ? 'wrist plan stopped' : 'wrist plan discarded');
      },
    };
  })();
  $('planExec').onclick = () => wristPlan.execute();
  $('planCancel').onclick = () => wristPlan.cancel(false);

  // ── cinematic 1st↔3rd person transition (2 s: fade ↔ fly) ──
  const povfx = new PovFx(viewport, $('fpv'), $('viewport3d'), {
    // respect the operator's layer choices: the transition flies through
    // whatever is currently visible — nothing gets force-enabled
    onLayers: () => {},
    onDone: () => bus.emit('ctl'),
  });
  window.__povfx = povfx;

  // ── webcam full-body teleop: MediaPipe body+hands → teleop_frame envelopes
  // (the Vision Pro wire path, synthesized from the webcam — bodyteleop.js)
  const bodyTeleop = new BodyTeleop(link, ctl, viewport, {
    button: $('webcamTeleopBtn'),
    status: $('btStatus'), regionChip: $('btRegion'), partsChip: $('btParts'),
    preview: $('btPreview'), video: $('btVideo'), canvas: $('btCanvas'),
    regionSeg: $('regionSeg'),
    onEngage: () => refreshOverlay(),
    onStop: () => { applyControlSpec(); refreshOverlay(); },
    toast: (m, e) => toast(m, e),
  });
  $('webcamTeleopBtn').onclick = () => bodyTeleop.toggle();
  // webcam-operator dock: ⤢ toggles the large preview
  $('btSizeBtn').onclick = () => $('btStatus').classList.toggle('big');

  // (There is no Start-Teleop button any more — the selected space is live from
  // boot. The explicit "energise" intent it used to carry now lives ONLY on the
  // top bar's E-STOP / Enable Torque button, which is where an operator looks
  // for it anyway. See the `ctl.teleop` comment at the top of this file.)
  seg('povSeg', (v) => {
    if (povfx.running || v === ctl.pov) return;
    ctl.pov = v;
    const img = $('fpvImg');
    // first-person = the head cam, full-bleed: the view that most needs the fast
    // H.264 path (CamView; MJPEG fallback). Torn down in 3rd person so we do not
    // pay for a stream nobody is looking at.
    if (v === 'first'){
      if (!fpvView) fpvView = new CamView(img, 'main');
      else fpvView.connect();
    }
    povfx.run(v === 'first' ? 'to1st' : 'to3rd').then(() => {
      if (ctl.pov === 'third') fpvView?.stop();
    });
    refreshOverlay();
  });
  // ONE control-space decision: Drive · Task · Joint. `mode` stays as the
  // derived drive/manip shorthand the pump and the webcam path already speak.
  seg('spaceSeg', (v) => {
    if (v === ctl.space) return;
    // leaving Drive must not leave the base coasting on the last stick frame
    if (ctl.space === 'drive') link.frame({ lx: 0, ly: 0, rx: 0, ry: 0 });
    // leaving Task drops any un-executed plan (its preview would otherwise hang)
    if (ctl.space === 'task') wristPlan?.cancel(true);
    ctl.space = v;
    ctl.mode = v === 'drive' ? 'drive' : 'manip';
    // The green wrist ball belongs to TASK space ONLY — in Task you steer it, in
    // Joint you command the sliders directly, so a ball there is just clutter
    // (and would imply a targeting mode Joint space does not have).
    viewport.setLayer('wrist', v === 'task');
    // preload the MoveIt ghost so the first drag previews without a load hitch
    if (v === 'task') viewport.ensureGhost?.();
    if (v !== 'drive') viewport.focusModule?.(ctl.module);   // fly to the rail's module
    applyControlSpec(); refreshOverlay();
  });
  seg('feelSeg', (v) => {
    ctl.feel = v;
    if (v === 'soft') toast('SOFT: app is a viewer — move the robot by hand');
    applyControlSpec(); refreshOverlay();
  });
  seg('gravSeg', (v) => {
    ctl.grav = (v === 'on');
    toast(ctl.grav ? 'GRAVITY COMP ON — measured feed-forward carries the arm'
                   : 'PURE PD — no feed-forward; loaded joints will sag');
    // Send it alone so the toggle cannot disturb the control spec mid-drive.
    link.gravityComp(ctl.grav); refreshOverlay();
  });
  seg('regionSeg', (v) => {
    ctl.region = v;
    // widen the wrist-drag IK chain to match (arm / +torso+lift / +base)
    viewport.controlRegion = v;
    applyControlSpec();
    toast(v === 'arm' ? 'IK region: arms only'
      : v === 'upper_body' ? 'IK region: arms + torso + lift'
      : 'IK region: whole body — base follows out-of-reach targets');
  });
  viewport.controlRegion = ctl.region;

  // ── top bar: ONE power button — Torque ON ⇄ E-STOP (iOS parity) ──────────
  // MABEL boots LIMP, so the single switch reads "⚡ Torque ON" (press to
  // energise) and, once live, flips to a red "■ E-STOP" (press to de-energise).
  // Energising REVEALS the controls (refreshOverlay gates on it); stopping folds
  // them away. This is the soft path — instant and reversible, exactly like
  // iOS's STOP/RESUME. The HARD hardware kill (shuts the bring-up down) is the
  // Esc-Esc panic path below, for a genuine emergency.
  const powerBtn = $('powerBtn');
  const syncTopButtons = () => {
    const energized = ctl.torque && !ctl.estop;
    powerBtn.classList.toggle('armed', energized);
    powerBtn.innerHTML = energized ? '■&nbsp;E-STOP' : '⚡&nbsp;Torque&nbsp;ON';
    powerBtn.title = energized
      ? 'Stop the robot — motors go limp (Esc Esc = hard shutdown)'
      : 'Energise the robot and reveal the controls';
    // Home follows the same reveal rule as every other control: it appears once
    // the robot is energised, because a glide to a limp actuator does nothing
    // and reads as a broken button.
    $('homeBtn').classList.toggle('hidden', !energized);
  };
  powerBtn.onclick = () => {
    const energized = ctl.torque && !ctl.estop;
    if (energized){
      cancelHome();                  // never keep streaming a glide into a stop
      wristPlan?.cancel(true);       // ...nor an eased wrist plan
      // soft e-stop: the bridge zeroes drive + drops torque instantly
      ctl.torque = false;
      link.send('estop', { active: true });
      toast('E-STOP — motors limp');
    } else {
      // energise: the bridge hold-then-enables at the MEASURED pose
      // (facade.resume_hold / hand resume), so nothing lurches. Re-seat the
      // module panel to the measured pose and DROP any "owned" targets the
      // operator set while limp — otherwise a slider still displaying a stale
      // target would fling the joint on the next touch, and the panel would
      // disagree with the robot it just told to hold where it stands.
      ctl.estop = false;
      ctl.torque = true;
      link.send('estop', { active: false });
      modulePanel.resync();
      // ...and tell the controller to hold the hands/wrists exactly where they
      // are. Energising is the moment stale-or-absent targets become real
      // motion, and the hand's are absent until someone opens that module.
      holdHandsAtMeasured();
      toast('energising — holding the measured pose');
    }
    syncTopButtons();
    refreshOverlay();
    bus.emit('ctl');
  };

  // Esc Esc — the HARD emergency stop: kill the hardware bring-up so every
  // subsystem releases its motors and serial links entirely, then relaunch on
  // the next Torque ON.
  const hardEstop = async () => {
    cancelHome();
    wristPlan?.cancel(true);
    ctl.estop = true; ctl.torque = false;
    link.estop(true);
    syncTopButtons(); refreshOverlay(); bus.emit('ctl');
    toast('E-STOP ENGAGED — shutting hardware down', true);
    try {
      const r = await fetch('/api/hw/estop', { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ engage: true }) }).then(x => x.json());
      toast(r.found ? 'hardware bridge DOWN — motors released'
        : 'e-stop held by the bridge (hardware launch not found)', true);
    } catch { toast('e-stop held by the bridge only', true); }
  };
  let escAt = 0;
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape'){
      const now = performance.now();
      if (now - escAt < 400) hardEstop();
      escAt = now;
    }
  });
  // ── ⌂ Home — glide every joint to the rest pose (iOS/VP goHome parity) ──
  // Arms hanging naturally down at the sides (the MJCF gravity equilibrium),
  // level wrists, neutral torso/neck, wheels forward, relaxed hands. The pose
  // and the glide math live in homepose.js (byte-identical to the iOS values,
  // node-tested); this is just the button, the wire, and the cancel paths.
  let homeTimer = null;
  function cancelHome(){
    if (!homeTimer) return;
    clearInterval(homeTimer);
    homeTimer = null;
    $('homeBtn').classList.remove('on');
    $('homeBtn').innerHTML = '⌂<span class="gb-lab">&nbsp;Home</span>';
  }
  $('homeBtn').onclick = () => {
    if (homeTimer){ cancelHome(); toast('home glide cancelled'); return; }
    if (ctl.estop || !ctl.torque){
      toast('energise the robot first — the arms are limp', true); return;
    }
    const live = link.state?.jointPositions || {};
    // ALREADY HOME ⇒ do nothing at all (2026-08-07, "pressing home while home
    // makes it oscillate"). Every step below moves the robot: controlMode()
    // reseeds joint targets to the MEASURED pose, and the glide then commands
    // back to REST_POSE — so on a robot sitting at home with a few degrees of
    // sag, each press pulled the arms up and released them again. Bail BEFORE
    // the mode switch; a no-op has to touch nothing.
    if (isAtHome(live)){
      toast(`already home (within ${(homeError(live) * 57.3).toFixed(1)}°)`);
      return;
    }
    // NO control-mode switch. Home is a plain POSE INTERPOLATION down the same
    // wire a slider uses (`link.joints`), so it inherits the deployed PD +
    // gravity feed-forward and the operator's own gains — the robot answers
    // Home exactly the way it answers a dragged torso slider (operator
    // direction, 2026-08-07). Declaring joint/impedance/whole_body here made
    // Home its own control regime AND reseeded every joint target to the
    // measured pose on the way in and out, which is what made a re-press nudge
    // the arms and what abandoned the pose at the end.
    if (ctl.feel === 'soft')
      toast('soft mode — the arms will watch the targets go by; switch to stiff to track');
    // Start from the MEASURED pose where we have one, else from the target
    // itself so an unreported joint holds still (see homepose.js) — never 0.
    const start = {};
    for (const k in REST_POSE) start[k] = live[k] != null ? live[k] : REST_POSE[k];
    const dur = homeDuration(start);
    const t0 = performance.now();
    $('homeBtn').classList.add('on');
    $('homeBtn').innerHTML = '⌂<span class="gb-lab">&nbsp;Homing…</span>';
    toast(`gliding to the rest pose (${dur.toFixed(1)}s)`);
    // THE COLUMN GOES ONCE, not per frame: it is a bang-bang servo, and a streamed
    // trajectory slower than its one speed is tracked by stop-starting (see homepose.js
    // LIFT_JOINTS). One target, and the firmware's own profile and taper do the move.
    link.joints(liftHome());
    // Stream the interpolated pose and STOP when the clock runs out. The last
    // frame IS the rest pose, and joint targets persist on the robot, so the
    // servos keep holding it afterwards exactly as they hold a slider value —
    // there is nothing to "finish".
    //
    // What used to be here (2026-07-29..2026-08-07) was an arrival/stall state
    // machine that kept streaming and then called applyControlSpec() to hand
    // control back. Both halves hurt: applyControlSpec() reseeds every joint
    // target to the MEASURED pose, so declaring a stall ABANDONED the rest pose
    // and latched the robot wherever it had got to (arms ~24° short at J1, held
    // with zero position error). Home is a pose move, not a control regime.
    homeTimer = setInterval(() => {
      const u = Math.min(1, (performance.now() - t0) / 1000 / dur);
      link.joints(homePoseAt(start, u));
      if (u >= 1){
        cancelHome();
        toast('home');
      }
    }, 33);
  };

  syncTopButtons();
  $('recordBtn').onclick = () => {
    ctl.recording = !ctl.recording;
    link.record(ctl.recording, ctl.task);
    $('recordBtn').classList.toggle('rec', ctl.recording);
    $('recordBtn').textContent = ctl.recording ? '■ Stop' : '● Record';
    toast(ctl.recording ? `recording episode · ${ctl.task}` : 'recording stopped');
  };
  $('recTask').onclick = () => {
    const t = prompt('task name for new episodes', ctl.task);
    if (t){ ctl.task = t; link.selectTask(t); $('recTask').textContent = 'task: ' + t; }
  };

  // connection dot
  const updateConn = () => {
    const on = link.connected;
    $('conn').classList.toggle('ok', on);
    $('connLabel').textContent = on
      ? `${link.hello?.name || 'robot'} · ${link.latencyMs != null ? link.latencyMs.toFixed(0) + 'ms' : '—'}`
      : 'offline';
  };
  bus.on('link', updateConn);
  bus.on('latency', updateConn);
  bus.on('hello', updateConn);
  setInterval(updateConn, 2000);

  // ── stream reattach: an e-stop hardware relaunch restarts the camera server
  // too. The control link auto-reconnects; when it comes BACK, re-attach every
  // MJPEG consumer (an <img> never recovers a dropped multipart stream) and
  // rebuild the 3D camera projections. No page reload.
  let wasDown = false;
  bus.on('link', ({ connected }) => {
    if (!connected){ wasDown = true; return; }
    if (!wasDown) return;
    wasDown = false;
    setTimeout(() => {
      bus.emit('streamsRefresh');
      viewport._rebuildCamViz?.();
      if (ctl.pov === 'first') fpvView?.connect();
      toast('streams re-attached');
    }, 2500);                       // camera server needs a moment after the bridge
  });

  // fpv PiP: reparent nothing — CSS shrinks the 3D canvas into the corner.
  // (the fpv <img> sits above; the viewport keeps rendering into its box)

  // ── choreography timeline ──
  const timeline = new Timeline({
    scroll: $('tlScroll'), ruler: $('rulerCanvas'), tracks: $('tlTracks'),
    playhead: $('tlPlayhead'), toolbar: $('tlToolbar'),
  }, link, ctl, {
    getPose: () => link.state?.jointPositions || telemetry.channel('p'),
    getTarget: () => telemetry.channel('tgt'),
    // scrub / preview-play → the MAIN 3D viewer becomes the choreography
    // preview (joints + integrated base path); null hands it back to live
    onPreview: (pose, base) => {
      if (pose || base) viewport.previewPose(pose, base);
      else viewport.endPreview(false);
    },
    // ⚡ Execute → fade the viewer from the preview back to the live robot
    onExecuteView: () => viewport.endPreview(true),
  });
  bus.on('viewportReady', () => viewport.ensureGhost());

  // ── physics-preview window: opened by the timeline's ▶ Preview ──
  let pvPanel = null;
  bus.on('physicsPreviewWindow', () => {
    if (pvPanel && panels.panels.includes(pvPanel)){ panels._raise(pvPanel); return; }
    pvPanel = panels.add('previewview');
    if (pvPanel){
      const ws = panels.wsRect();
      panels._snap(pvPanel, { x: ws.x + ws.w * 0.55, y: ws.y,
                              w: ws.w * 0.45, h: ws.h * 0.6 });
      pvPanel.el.querySelector('.fp-close').addEventListener('click',
        () => { timeline.stopPhysicsPreview(); pvPanel = null; });
    }
  });

  // ── skills: composed choreographies as reusable clips ──
  const skillStore = new SkillStore();
  new SkillsPanel($('skillsPane'), skillStore, timeline);
  bus.on('composeSkill', () => {
    if (!timeline.track.blocks.length && !timeline.seq.keys.length){
      toast('timeline is empty — nothing to compose', true);
      return;
    }
    const title = prompt('skill name', 'my skill');
    if (title){
      skillStore.compose(title, timeline);
      toast(`skill saved: ${title} — see the Library / Skills panels`);
    }
  });

  refreshOverlay();
  window.__cc = { link, monitor, maps, telemetry, streams, history, viewport,
                  jointPanel, timeline, ctl, panels, camstrip, statusPanel,
                  bodyTeleop };

  // ── e2e: deterministic POV-transition recording (?record=pov&fps=60) ──
  if (qp.get('record') === 'pov'){
    const start = () => setTimeout(async () => {
      const { recordPov } = await import('./recorder.js');
      try { await recordPov(window.__cc, { fps: +(qp.get('fps') || 60) }); }
      catch (err){ document.title = 'POV-RECORD-FAIL ' + err.message; }
    }, 6000);
    if (viewport.rig) start(); else bus.on('viewportReady', start);
  }
}

// Gate the whole studio behind the styled login (session cookie); only boot the
// data planes once authenticated so nothing loads for an unauthorized visitor.
requireLogin().then(main).catch((e) => { console.error(e); main(); });
