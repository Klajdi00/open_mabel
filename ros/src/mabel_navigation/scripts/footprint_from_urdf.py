#!/usr/bin/env python3
"""Derive MABEL's navigation SQUARE footprint from the URDF.

The Nav2 costmaps (config/nav2_params.yaml) use a SQUARE footprint polygon -- the
mobile base is a square, not a circle. That polygon is a DERIVED quantity: it must
track the chassis geometry, which comes from the MJCF (mabel_full.xml) -> URDF
pipeline. Run this whenever the base geometry changes and copy the printed
`footprint:` line into BOTH costmaps in nav2_params.yaml:

    python3 src/mabel_navigation/scripts/footprint_from_urdf.py

It walks the URDF base links (mabel_chassis + lift_base by default), loads their
collision meshes, transforms every vertex into base_link, and reports the XY bounding
box (the square footprint) plus the inscribed/circumscribed radii (for inflation).
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np


def _rpy_xyz(origin):
    if origin is None:
        return np.zeros(3), np.zeros(3)
    xyz = np.array([float(v) for v in (origin.get('xyz') or '0 0 0').split()])
    rpy = np.array([float(v) for v in (origin.get('rpy') or '0 0 0').split()])
    return xyz, rpy


def _Rmat(rpy):
    cr, cp, cy = np.cos(rpy)
    sr, sp, sy = np.sin(rpy)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _joints(root):
    j = {}
    for joint in root.findall('joint'):
        child = joint.find('child').get('link')
        xyz, rpy = _rpy_xyz(joint.find('origin'))
        j[child] = (joint.find('parent').get('link'), xyz, _Rmat(rpy))
    return j


def _to_base(link, joints):
    """(R, t) mapping the link's local frame into base_link."""
    chain = []
    cur = link
    while cur in joints:
        parent, xyz, Rj = joints[cur]
        chain.append((xyz, Rj))
        cur = parent
    R = np.eye(3)
    t = np.zeros(3)
    for xyz, Rj in reversed(chain):
        t = t + R @ xyz
        R = R @ Rj
    return R, t


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    default_urdf = os.path.normpath(os.path.join(
        here, '..', '..', 'mabel_description', 'urdf', 'mabel_full.urdf'))
    ap.add_argument('--urdf', default=default_urdf)
    ap.add_argument('--pkg', default=os.path.normpath(os.path.join(
        here, '..', '..', 'mabel_description')),
        help='mabel_description dir that the package:// mesh paths resolve against')
    ap.add_argument('--links', nargs='+', default=['mabel_chassis', 'lift_base'],
                    help='base links whose collision meshes bound the footprint')
    args = ap.parse_args()

    try:
        import trimesh
    except ImportError:
        sys.exit('need trimesh:  pip install trimesh')

    root = ET.parse(args.urdf).getroot()
    joints = _joints(root)

    overall = 0.0
    bbox = [0.0, 0.0, 0.0, 0.0]    # xmin, xmax, ymin, ymax in base_link
    for link in root.findall('link'):
        name = link.get('name')
        if name not in args.links:
            continue
        R, t = _to_base(name, joints)
        for col in link.findall('collision'):
            mesh = col.find('geometry/mesh')
            if mesh is None:
                continue
            scale = float((mesh.get('scale') or '1 1 1').split()[0])
            rel = mesh.get('filename').split('meshes/')[-1]
            path = os.path.join(args.pkg, 'meshes', rel)
            v = trimesh.load(path, force='mesh').vertices * scale
            cxyz, crpy = _rpy_xyz(col.find('origin'))
            v = (_Rmat(crpy) @ v.T).T + cxyz       # apply collision origin
            v = (R @ v.T).T + t                     # into base_link
            rad = float(np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2).max())
            overall = max(overall, rad)
            bbox[0] = min(bbox[0], float(v[:, 0].min()))
            bbox[1] = max(bbox[1], float(v[:, 0].max()))
            bbox[2] = min(bbox[2], float(v[:, 1].min()))
            bbox[3] = max(bbox[3], float(v[:, 1].max()))
            print(f'  {name:16s} x[{v[:,0].min():+.3f},{v[:,0].max():+.3f}] '
                  f'y[{v[:,1].min():+.3f},{v[:,1].max():+.3f}]  R={rad:.3f} m')

    xmin, xmax, ymin, ymax = bbox
    inscribed = min(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
    print(f'\nbounding box: x[{xmin:+.3f},{xmax:+.3f}]  y[{ymin:+.3f},{ymax:+.3f}]')
    print(f'inscribed radius = {inscribed:.3f} m   circumscribed radius = {overall:.3f} m')
    print('-> set in BOTH costmaps in nav2_params.yaml:')
    print(f'   footprint: "[[{xmax:.3f}, {ymax:.3f}], [{xmax:.3f}, {ymin:.3f}], '
          f'[{xmin:.3f}, {ymin:.3f}], [{xmin:.3f}, {ymax:.3f}]]"')
    print(f'   inflation_radius: >= {overall:.3f}   (circumscribed, to keep the '
          f'SE2 potential-field shortcut on)')


if __name__ == '__main__':
    main()
