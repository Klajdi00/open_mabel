// RobotLink wire shapes (no network: stub ws) + config URL construction.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RobotLink } from '../../web/js/link.js';
import { teleopUrl, mapUrl, camUrl, monitorUrl, conf } from '../../web/js/config.js';

function stubbed(){
  const link = new RobotLink();
  const sent = [];
  link.ws = { readyState: 1, send: (s) => sent.push(JSON.parse(s)), close(){} };
  return { link, sent };
}

test('every command is a {type, payload} envelope', () => {
  const { link, sent } = stubbed();
  link.frame({ lx: 0, ly: 1, rx: 0, ry: 0 });
  link.joints({ left_arm_1: 0.5 });
  link.controlMode('impedance', 'joint', 'arm', 27);
  link.estop(true);
  link.navGoal(1, 2, 0.5);
  link.record(true, 'pick_cup');
  link.dataCollect(true, 'run1');
  link.externalForce({ torso: [0, 0, 5] });
  for (const m of sent){
    assert.equal(typeof m.type, 'string');
    assert.equal(typeof m.payload, 'object');
  }
  link.close();
});

test('teleop_frame carries navJoystick + sequence', () => {
  const { link, sent } = stubbed();
  link.frame({ lx: -0.2, ly: 0.8, rx: 0.1, ry: 0 });
  link.frame({ lx: 0, ly: 0, rx: 0, ry: 0 });
  assert.equal(sent[0].type, 'teleop_frame');
  assert.deepEqual(sent[0].payload.navJoystick, { lx: -0.2, ly: 0.8, rx: 0.1, ry: 0 });
  assert.equal(sent[1].payload.sequence, sent[0].payload.sequence + 1);
  link.close();
});

test('control_mode / estop / nav_goal payload fields', () => {
  const { link, sent } = stubbed();
  link.controlMode('compliance', 'joint', 'whole_body', 45);
  link.estop(true);
  link.navGoal(1.5, -0.5, 3.1);
  const [cm, es, ng] = sent;
  assert.deepEqual(cm.payload, { method: 'compliance', controlType: 'joint',
                                 region: 'whole_body', stiffness: 45 });
  assert.deepEqual(es.payload, { active: true });
  assert.deepEqual(ng.payload, { x: 1.5, y: -0.5, yaw: 3.1 });
  link.close();
});

test('record select vs active shapes', () => {
  const { link, sent } = stubbed();
  link.selectTask('pour');
  link.record(true, 'pour');
  link.record(false);
  assert.deepEqual(sent[0].payload, { select: 'pour' });
  assert.deepEqual(sent[1].payload, { active: true, task: 'pour' });
  assert.deepEqual(sent[2].payload, { active: false });
  link.close();
});

test('send() returns false with no socket', () => {
  const link = new RobotLink();
  assert.equal(link.send('ping', {}), false);
  link.close();
});

test('config URL builders target the right planes', () => {
  // client tag + per-tab session id (the network panel's "Current window")
  assert.match(teleopUrl(), /^ws:\/\/.*:9090\/teleop\?client=web&sid=[a-z0-9]{4,8}$/);
  assert.match(monitorUrl(), /^ws:\/\/.*:9090\/monitor$/);
  // real robot default: the bridge's own :9092 map plane
  assert.match(mapUrl(), /^ws:\/\/.*:9092\/map$/);
  assert.match(camUrl('main'), /^http:\/\/.*:8080\/camera\/main\/stream\.mjpg$/);
  assert.equal(conf().tempWarn < conf().tempHot, true);
});
