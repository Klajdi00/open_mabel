import os
import sys
import time
import json
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import logging
import zmq
from avp_stream import VisionProStreamer
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

if os.name == 'nt': os.system('')

class ColorFormatter(logging.Formatter):
    COLORS = {logging.INFO: "\033[92m", logging.WARNING: "\033[93m", logging.ERROR: "\033[91m", logging.DEBUG: "\033[96m"}
    RESET = "\033[0m"
    def format(self, record): return f"{self.COLORS.get(record.levelno, self.RESET)}[{record.levelname}] {super().format(record)}{self.RESET}"

logger = logging.getLogger("ORCA_AVP")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorFormatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(handler)

ZMQ_PUB_PORT = 5555                    
XML_FILE = "main.xml"                  
PALM_BODY_NAME = "right_palm"          
WRIST_JOINT_NAME = "right_wrist"       

VR_EULER_OFFSET = [90, 90, -90]       

HAND_JOINTS = [
    "right_index_abd", "right_index_mcp", "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd", "right_ring_mcp", "right_ring_pip",
    "right_pinky_abd", "right_pinky_mcp", "right_pinky_pip",
    "right_thumb_abd", "right_thumb_mcp", "right_thumb_pip", "right_thumb_dip"
]

BODY_MAPPING = {
    3:  "right_thumb_ip",  4:  "right_thumb_dp", 
    6:  "right_index_ip",  8:  "right_index_ip",    
    10: "right_middle_ip", 12: "right_middle_ip", 
    14: "right_ring_ip",   16: "right_ring_ip",    
    18: "right_pinky_ip",  20: "right_pinky_ip"   
}
TIP_INDICES = [4, 8, 12, 16, 20]
RELATIVE_PAIRS = [(4, 8), (4, 12), (4, 16), (4, 20)]

FINGER_SCALES = {"THUMB": 1.0, "INDEX": 1.0, "MIDDLE": 1.0, "RING": 1.0, "PINKY": 1.0}
FINGER_TIP_OFFSETS = {
    "THUMB":  [0.0, 0.0, 0.035],
    "INDEX":  [0.0, 0.0, 0.035],
    "MIDDLE": [0.0, 0.0, 0.035],
    "RING":   [0.0, 0.0, 0.035],
    "PINKY":  [0.0, 0.0, 0.035]
}

W_TIP_POS   = 200.0    
W_MID_POS   = 100.0    
W_COSINE    = 20.0     
W_RELATIVE  = 150.0    
W_SMOOTH    = 0.1      
W_COLLISION = 5.0      
W_FREEZE    = 0.1      

def get_mp_aligned_mats(m):
    return [m[0], m[1], m[2], m[3], m[4], m[6], m[7], m[8], m[9], m[11], m[12], m[13], m[14], m[16], m[17], m[18], m[19], m[21], m[22], m[23], m[24]]

def get_finger_name(idx):
    if idx in [1, 2, 3, 4]: return "THUMB"
    if idx in [5, 6, 7, 8]: return "INDEX"
    if idx in [9, 10, 11, 12]: return "MIDDLE"
    if idx in [13, 14, 15, 16]: return "RING"
    if idx in [17, 18, 19, 20]: return "PINKY"
    return "WRIST"

def draw_retarget_overlay(viewer, wrist_pos, palm_mat, target_local_tips):
    if not hasattr(viewer, 'user_scn') or viewer.user_scn is None: return
    viewer.user_scn.ngeom = 0 
    for avp_idx, local_pos in target_local_tips.items():
        global_target_pos = wrist_pos + (palm_mat @ local_pos)
        rgba = [0.2, 1.0, 0.2, 0.7] if avp_idx in TIP_INDICES else [1.0, 0.8, 0.2, 0.5]
        size = 0.008 if avp_idx in TIP_INDICES else 0.006
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[size, 0, 0], 
            pos=global_target_pos, mat=np.eye(3).flatten(), rgba=rgba 
        )
        viewer.user_scn.ngeom += 1

class OrcaHandOptimizer:
    def __init__(self, model):
        self.model = model
        self.ik_data = mujoco.MjData(model)
        
        self.palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PALM_BODY_NAME)
        self.wrist_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, WRIST_JOINT_NAME)
        
        if self.palm_bid == -1 or self.wrist_jid == -1:
            logger.error("Could not find palm body or wrist joint in XML.")
            sys.exit(1)
            
        self.tracking_bids = {}
        for avp_idx, xml_name in BODY_MAPPING.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, xml_name)
            if bid != -1: self.tracking_bids[avp_idx] = bid

        self.hand_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in HAND_JOINTS]
        if -1 in self.hand_jids:
            logger.error("Missing joints in XML.")
            sys.exit(1)

        self.hand_q_idx = [model.jnt_qposadr[jid] for jid in self.hand_jids]
        self.hand_bounds = [model.jnt_range[jid] for jid in self.hand_jids]
        self.last_q_hand = np.zeros(len(self.hand_jids))
        self.R_offset = R.from_euler('xyz', VR_EULER_OFFSET, degrees=True).as_matrix()

    def get_robot_local_tips(self, q_fingers):
        for i, idx in enumerate(self.hand_q_idx): self.ik_data.qpos[idx] = q_fingers[i]
        mujoco.mj_kinematics(self.model, self.ik_data)
        
        wrist_pos = self.ik_data.xanchor[self.wrist_jid].copy()
        palm_mat = self.ik_data.xmat[self.palm_bid].reshape(3, 3).copy()
        palm_inv = palm_mat.T 
        
        local_robot_tips = {}
        for avp_idx, bid in self.tracking_bids.items():
            global_pos = self.ik_data.xpos[bid].copy()
            if avp_idx in TIP_INDICES: 
                finger_name = get_finger_name(avp_idx)
                global_pos += (self.ik_data.xmat[bid].reshape(3, 3) @ np.array(FINGER_TIP_OFFSETS[finger_name]))
            local_robot_tips[avp_idx] = palm_inv @ (global_pos - wrist_pos)
            
        return wrist_pos, palm_mat, local_robot_tips

    def finger_cost_function(self, q_fingers, target_local_tips, q_prev, is_snapping):
        _, _, robot_local_pts = self.get_robot_local_tips(q_fingers)
        loss = 0.0
        
        common = sorted(list(set(robot_local_pts.keys()) & set(target_local_tips.keys())))
        for idx in common: 
            weight = W_TIP_POS if idx in TIP_INDICES else W_MID_POS
            loss += weight * np.sum((robot_local_pts[idx] - target_local_tips[idx])**2)
            
        if common:
            r_vec = np.array([robot_local_pts[k] for k in common]).flatten()
            h_vec = np.array([target_local_tips[k] for k in common]).flatten()
            loss += W_COSINE * (1.0 - (np.dot(r_vec, h_vec) / ((np.linalg.norm(r_vec) + 1e-6) * (np.linalg.norm(h_vec) + 1e-6))))
            
        for (a, b) in RELATIVE_PAIRS:
            if all(k in robot_local_pts for k in (a, b)) and all(k in target_local_tips for k in (a, b)):
                loss += W_RELATIVE * np.sum(((robot_local_pts[b] - robot_local_pts[a]) - (target_local_tips[b] - target_local_tips[a]))**2)
                
        adjacent_pairs = [(8, 12), (12, 16), (16, 20)]
        for (f1, f2) in adjacent_pairs:
            if f1 in robot_local_pts and f2 in robot_local_pts:
                finger_dist = np.linalg.norm(robot_local_pts[f1] - robot_local_pts[f2])
                if finger_dist < 0.015: loss += W_COLLISION / (finger_dist + 1e-5) 
                
        active_smooth = 0.0 if is_snapping else W_SMOOTH
        loss += active_smooth * np.sum((q_fingers - q_prev)**2)
        loss += W_FREEZE * np.sum((q_fingers - q_prev)**2)
        
        return loss

    def solve(self, h_mats):
        t0 = time.perf_counter()
        
        h_mats_np = [np.array(m).reshape(4, 4) for m in h_mats]
        if np.isnan(h_mats_np).any() or len(h_mats_np) < 21:
            return self.last_q_hand.copy(), 0.0, {}, False
            
        vr_wrist_pos = h_mats_np[0][:3, 3]
        vr_index_tip = h_mats_np[8][:3, 3]
        vr_wrist_rot = h_mats_np[0][:3, :3]
        vr_wrist_inv = vr_wrist_rot.T  

        if np.linalg.norm(vr_index_tip - vr_wrist_pos) < 0.03:
            return self.last_q_hand.copy(), 0.0, {}, False

        target_local_tips = {}
        for avp_idx in self.tracking_bids.keys():
            if avp_idx < len(h_mats_np):
                global_vr_tip = h_mats_np[avp_idx][:3, 3]
                local_vec = global_vr_tip - vr_wrist_pos
                finger_name = get_finger_name(avp_idx)
                scale = FINGER_SCALES.get(finger_name, 1.0)
                target_local_tips[avp_idx] = (self.R_offset @ (vr_wrist_inv @ local_vec)) * scale

        _, _, current_robot_tips = self.get_robot_local_tips(self.last_q_hand)
        max_jump = max([np.linalg.norm(target_local_tips[idx] - current_robot_tips[idx]) for idx in TIP_INDICES if idx in target_local_tips and idx in current_robot_tips] + [0.0])
                
        is_snapping = max_jump > 0.04 
        if is_snapping:
            sys.stdout.write("\r\033[K")
            logger.warning(f"Reconnection/Large movement detected ({max_jump*100:.1f}cm). Snapping to pose...")

        res = minimize(
            self.finger_cost_function, self.last_q_hand, 
            args=(target_local_tips, self.last_q_hand, is_snapping), 
            method='SLSQP', bounds=self.hand_bounds, 
            options={'maxiter': 30 if is_snapping else 10, 'ftol': 1e-2}
        )
        
        if not np.isnan(res.x).any(): 
            self.last_q_hand = np.clip(res.x, [b[0] for b in self.hand_bounds], [b[1] for b in self.hand_bounds])
            
        dt_ms = (time.perf_counter() - t0) * 1000
        return self.last_q_hand.copy(), dt_ms, target_local_tips, True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", type=str, required=True, help="IP address of the Vision Pro")
    args = parser.parse_args()

    try:
        model = mujoco.MjModel.from_xml_path(XML_FILE)
    except Exception as e:
        logger.error(f"Failed to load {XML_FILE}. ({e})")
        sys.exit(1)
        
    vis_data = mujoco.MjData(model)
    optimizer = OrcaHandOptimizer(model)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{ZMQ_PUB_PORT}")

    streamer = VisionProStreamer(ip=args.room)

    logger.info(f"Unified AVP Teleop Online.")
    logger.info(f"Tracking AVP on {args.room} -> Publishing Solved Joints to port {ZMQ_PUB_PORT}")
    
    last_log_time, frames = time.time(), 0

    with mujoco.viewer.launch_passive(model, vis_data) as viewer:
        while viewer.is_running():
            avp_data = streamer.get_latest()
            
            if avp_data and hasattr(avp_data, 'right') and avp_data.right is not None:
                if len(avp_data.right) > 0:
                    mp_mats = get_mp_aligned_mats(avp_data.right)
                    q_hand, dt, target_local_tips, is_valid = optimizer.solve(mp_mats)
                    
                    if not is_valid:
                        viewer.sync()
                        time.sleep(0.002)
                        continue

                    frames += 1
                    bounds_rad = [list(b) for b in optimizer.hand_bounds]

                    target_payload = {
                        "joint_names": HAND_JOINTS,
                        "target_q_rad": q_hand.tolist(),
                        "bounds_rad": bounds_rad
                    }
                    pub.send_string(json.dumps(target_payload))

                    for i, idx in enumerate(optimizer.hand_q_idx): 
                        vis_data.qpos[idx] = q_hand[i]
                    mujoco.mj_forward(model, vis_data)
                    
                    wrist_pos = vis_data.xanchor[optimizer.wrist_jid]
                    palm_mat = vis_data.xmat[optimizer.palm_bid].reshape(3, 3)
                    draw_retarget_overlay(viewer, wrist_pos, palm_mat, target_local_tips)
                    
                    if time.time() - last_log_time > 1.0:
                        sys.stdout.write("\r\033[K") 
                        logger.info(f"Loop Rate: {frames} Hz | IK Solve Time: {dt:.2f}ms")
                        frames, last_log_time = 0, time.time()

            viewer.sync()
            time.sleep(0.002) 

if __name__ == "__main__":
    main()