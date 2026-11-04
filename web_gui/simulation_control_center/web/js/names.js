// Map the 59-actuator vector (display names from the episode schema) to the 60
// snake_case rig-joint names mabel_joints.json / robot-viewer use. One special
// case: the single "Lift" actuator drives lift_lower + lift_upper (each = v/2).
export const norm = (s) => s.toLowerCase().replace(/\s+/g, '_');

export function buildActMap(actuatorNames){
  return actuatorNames.map((name) => name === 'Lift'
    ? { kind: 'split', targets: ['lift_lower', 'lift_upper'], factor: 0.5 }
    : { kind: 'one', target: norm(name) });
}

// vec: Float32Array slice [59]; returns {rigJointName: radians}
export function actuatorsToRig(vec, actMap){
  const q = {};
  for (let i = 0; i < actMap.length; i++){
    const m = actMap[i];
    if (m.kind === 'split') for (const t of m.targets) q[t] = vec[i] * m.factor;
    else q[m.target] = vec[i];
  }
  return q;
}

// Editable modality lanes for a clip: id (matches backend), label, and the
// HDF5 keys (a trailing '/' means prefix-match). Used by the timeline accordion
// and the save/materialize drop mask.
export function modalitiesFor(meta){
  // Expanded-clip lane order: SENSORS (cameras, lidar) → JOINT DATA (proprio) →
  // CONTROL (action) → OTHER (odom/tf, planned, state).
  const cams = (meta.cameras || []).map(c => ({
    id: 'cam:' + c.name, label: 'video · ' + c.name,
    keys: ['obs/cam/' + c.name + '/rgb', 'obs/cam/' + c.name + '/depth'] }));
  const sensors = [...cams, { id: 'lidar', label: 'lidar', keys: ['obs/lidar'] }];
  // proprio (pos+vel+torque merged) and action are module-expandable (7 modules → DOFs)
  const jointCtrl = [
    { id: 'proprio', label: 'proprio', expandable: true,
      keys: ['obs/joint_pos', 'obs/joint_vel', 'obs/joint_torque'] },
    { id: 'action', label: 'action · ctrl', expandable: true, keys: ['action/ctrl'] },
  ];
  const other = [
    { id: 'pose', label: 'odom / tf', keys: ['obs/pose/'] },
    { id: 'teleop', label: 'planned · teleop', keys: ['teleop/'] },
    { id: 'state', label: 'qpos / qvel', keys: ['obs/qpos', 'obs/qvel'] },
  ];
  return [...sensors, ...jointCtrl, ...other];
}

// Friendly group display order + labels for the bar-graph + topic tree.
export const GROUP_ORDER = ['base', 'lift_torso', 'neck', 'left_arm', 'right_arm', 'left_hand', 'right_hand'];
export const GROUP_LABEL = {
  base: 'Base', lift_torso: 'Lift/Torso', neck: 'Neck',
  left_arm: 'L Arm', right_arm: 'R Arm', left_hand: 'L Hand', right_hand: 'R Hand',
};

// One hue per module, shared by the trend chips and the trend curves so the
// chip IS the legend (iOS `JointGraphs.hue` parity — same values, so a curve
// keeps its colour when an operator moves between the phone and the studio).
// Deliberately warm/paper apart from the neck's blue, matching Theme.Palette.
export const MODULE_HUE = {
  left_arm: '#C25B2A',    // rust
  right_arm: '#2E9E52',   // green
  neck: '#4D9DFF',        // blue
  lift_torso: '#CC8A1E',  // amber
  left_hand: '#9A6A3C',   // clay
  right_hand: '#2AA8A0',  // teal
  base: '#8E6FD8',        // violet
};
