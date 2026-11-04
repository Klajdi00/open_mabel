"""urdf_gravity.py — static gravity torques for the arm chains straight from the URDF.

An INDEPENDENT theoretical reference beside the MJCF compensator (the URDF is generated
from the MJCF, so the two should agree — running both is the cross-check). Pure
kinematics + statics, no simulator:

    τ_hold(j) = − axis_j · Σ_{links below j} (p_com_i − p_j) × (m_i · g)

i.e. the torque the motor must apply about its axis to hold the subtree against gravity
(sign self-checked against the MJCF compensator at identical settings).

Options mirror the compensator: hand subtrees (left_tower / right_tower) can be
STRIPPED (hands off the robot), and the shoulder-mount tilt (l/r_arm_mount_fixed roll)
can be re-pinned to a target angle to test mount-tilt hypotheses.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

ARM_JOINTS = [f"{s}_arm_{k}" for s in ("left", "right") for k in range(1, 8)]
HAND_ROOTS = ("left_tower", "right_tower")
MOUNT_JOINTS = ("l_arm_mount_fixed", "r_arm_mount_fixed")
G = np.array([0.0, 0.0, -9.81])


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _axis_angle(axis, q):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) or 1.0)
    c, s = math.cos(q), math.sin(q)
    x, y, z = a
    C = 1 - c
    return np.array([[c + x*x*C, x*y*C - z*s, x*z*C + y*s],
                     [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
                     [z*x*C - y*s, z*y*C + x*s, c + z*z*C]])


class UrdfGravity:
    def __init__(self, urdf_path: str, *, hands: bool = False,
                 mount_tilt_deg: "float | None" = None):
        root = ET.parse(urdf_path).getroot()
        self.links = {}       # name -> (mass, com xyz in link frame)
        for link in root.iter("link"):
            inr = link.find("inertial")
            if inr is None:
                continue
            m = float(inr.find("mass").get("value"))
            o = inr.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            self.links[link.get("name")] = (m, np.array(xyz))
        self.joints = {}      # child_link -> dict(name,type,parent,xyz,rpy,axis)
        self.by_name = {}
        for j in root.iter("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = [float(v) for v in (ax.get("xyz", "0 0 1") if ax is not None else "0 0 1").split()]
            d = dict(name=j.get("name"), type=j.get("type"),
                     parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                     xyz=np.array(xyz), rpy=list(rpy), axis=np.array(axis))
            self.joints[d["child"]] = d
            self.by_name[d["name"]] = d
        if mount_tilt_deg is not None:
            for jn in MOUNT_JOINTS:
                jd = self.by_name.get(jn)
                if jd is not None:
                    jd["rpy"] = [math.radians(mount_tilt_deg), jd["rpy"][1], jd["rpy"][2]]
        if not hands:
            drop = set()
            def mark(link):
                drop.add(link)
                for child, jd in self.joints.items():
                    if jd["parent"] == link:
                        mark(child)
            for r in HAND_ROOTS:
                mark(r)
            self._dropped_mass = sum(self.links[l][0] for l in drop if l in self.links)
            for l in drop:
                self.links.pop(l, None)
        else:
            self._dropped_mass = 0.0
        # children index for subtree walks
        self.children = {}
        for child, jd in self.joints.items():
            self.children.setdefault(jd["parent"], []).append(child)

    def _fk(self, q: dict):
        """world (R, p) per link + per-joint (axis_w, origin_w). q keyed by joint name."""
        T = {}
        JW = {}
        def rec(link, R, p):
            T[link] = (R, p)
            for child in self.children.get(link, []):
                jd = self.joints[child]
                Rj = R @ _rpy(*jd["rpy"])
                pj = p + R @ jd["xyz"]
                if jd["type"] == "revolute" or jd["type"] == "continuous":
                    axis_w = Rj @ jd["axis"]
                    JW[jd["name"]] = (axis_w, pj)
                    Rc = Rj @ _axis_angle(jd["axis"], float(q.get(jd["name"], 0.0)))
                elif jd["type"] == "prismatic":
                    Rc = Rj
                    pj = pj + Rj @ (jd["axis"] * float(q.get(jd["name"], 0.0)))
                else:
                    Rc = Rj
                rec(child, Rc, pj)
        roots = set(self.links) | set(self.children)
        child_links = set(self.joints)
        for r in roots - child_links:
            rec(r, np.eye(3), np.zeros(3))
        return T, JW

    def _subtree(self, link):
        out = [link]
        for c in self.children.get(link, []):
            out += self._subtree(c)
        return out

    def hold_tau(self, q: dict) -> dict:
        """{urdf joint name: N·m to HOLD} for the 14 arm joints at pose q
        (q keyed by URDF joint names, model frame)."""
        T, JW = self._fk(q)
        out = {}
        for jn in ARM_JOINTS:
            jd = self.by_name.get(jn)
            if jd is None or jn not in JW:
                continue
            axis_w, pj = JW[jn]
            tau = np.zeros(3)
            for link in self._subtree(jd["child"]):
                if link not in self.links or link not in T:
                    continue
                m, com = self.links[link]
                R, p = T[link]
                pc = p + R @ com
                tau += np.cross(pc - pj, m * G)
            out[jn] = -float(axis_w @ tau)      # motor torque to OPPOSE gravity
        return out
