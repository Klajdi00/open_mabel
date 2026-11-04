#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  Mesh Toolkit  —  Digital Twin Utilities                            ║
║  MuJoCo · Isaac Sim · MJLab                                         ║
╚══════════════════════════════════════════════════════════════════════╝

Usage
─────
  python mesh_toolkit.py                         # interactive launcher
  python mesh_toolkit.py reduce  mesh.stl        # single file, guided
  python mesh_toolkit.py inspect mesh.stl        # detailed analysis
  python mesh_toolkit.py check   ./assets/       # pre-flight check
  python mesh_toolkit.py --help                  # full command list
"""

from __future__ import annotations

import os
import sys
import time
import struct
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import click
import trimesh
from rich.console   import Console
from rich.panel     import Panel
from rich.table     import Table
from rich.progress  import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
from rich.text      import Text
from rich.prompt    import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.padding   import Padding
from rich.rule      import Rule
from rich           import box as rich_box

# ─────────────────────────────────────────────────────────────────────────────
# Console & colour palette  (Apple HIG-inspired)
# ─────────────────────────────────────────────────────────────────────────────
console  = Console()
ACCENT   = "#5AC8FA"   # Apple light-blue
SUCCESS  = "#34C759"   # Apple green
WARN     = "#FF9F0A"   # Apple amber
ERR      = "#FF3B30"   # Apple red
MUTED    = "grey54"
HEADING  = "bold white"
DIM      = "grey42"
TIP_COL  = "#30D158"

# ─────────────────────────────────────────────────────────────────────────────
# Numeric constants
# ─────────────────────────────────────────────────────────────────────────────
MUJOCO_LIMIT   = 200_000
ISAAC_LIMIT    = 500_000   # Isaac Sim is more permissive but let's be sensible

SIZE_PRESETS = {
    # keyword   target faces
    "ultra"    : 8_000,
    "physics"  : 8_000,   # collision hulls
    "low"      : 20_000,
    "medium"   : 50_000,
    "high"     : 80_000,
    "mujoco"   : 80_000,  # MuJoCo visual safe limit
    "isaac"    : 100_000, # Isaac Sim visual
    "mjlab"    : 80_000,
    "full"     : None,    # keep original (used in check)
}

SCALE_FACTORS = {
    ("mm",   "m"): 0.001,
    ("cm",   "m"): 0.01,
    ("inch", "m"): 0.0254,
    ("m",    "m"): 1.0,
    ("mm",  "mm"): 1.0,
}

MATERIAL_DENSITY = {
    "aluminum": 2_700,
    "aluminium":2_700,
    "steel"   : 7_850,
    "abs"     : 1_050,
    "pla"     : 1_250,
    "petg"    : 1_270,
    "nylon"   : 1_150,
    "wood"    : 600,
    "water"   : 1_000,
    "carbon"  : 1_600,
}

# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _logo() -> None:
    console.print()
    console.print(Panel(
        f"[bold {ACCENT}]Mesh Toolkit[/]  [dim]v1.0[/]\n"
        f"[{MUTED}]Digital Twin Utilities  ·  MuJoCo · Isaac Sim · MJLab[/]",
        border_style=ACCENT,
        padding=(0, 2),
        expand=False,
    ))
    console.print()


def _section(title: str) -> None:
    console.print()
    console.print(Rule(f"[{ACCENT}]{title}[/]", style=DIM))
    console.print()


def _ok(msg: str) -> None:
    console.print(f"  [bold {SUCCESS}]✓[/]  {msg}")


def _warn(msg: str) -> None:
    console.print(f"  [bold {WARN}]⚠[/]  [bold]{msg}[/]")


def _err(msg: str) -> None:
    console.print(f"  [bold {ERR}]✗[/]  {msg}")
    sys.exit(1)


def _tip(msg: str) -> None:
    console.print(f"\n  [{TIP_COL}]Tip[/] [{MUTED}]{msg}[/]")


def _info(label: str, value: str) -> None:
    console.print(f"  [dim]{label:<18}[/]  {value}")


def _fmt_faces(n: int) -> str:
    """Coloured face-count string."""
    if n > MUJOCO_LIMIT:
        return f"[bold {ERR}]{n:,}[/]"
    if n > 80_000:
        return f"[bold {WARN}]{n:,}[/]"
    return f"[{SUCCESS}]{n:,}[/]"


def _fmt_size(path: Path) -> str:
    mb = path.stat().st_size / 1_048_576
    if mb > 20:
        return f"[bold {WARN}]{mb:.1f} MB[/]"
    return f"{mb:.1f} MB"


def _fmt_status(ok: bool, label_ok="Ready", label_fail="Issue") -> str:
    if ok:
        return f"[{SUCCESS}]✓  {label_ok}[/]"
    return f"[{ERR}]✗  {label_fail}[/]"


# ─────────────────────────────────────────────────────────────────────────────
# Mesh I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str | Path) -> trimesh.Trimesh:
    p = Path(path)
    if not p.exists():
        _err(f"File not found: {p}")
    try:
        m = trimesh.load_mesh(str(p), process=False)
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(list(m.dump()))
        if not isinstance(m, trimesh.Trimesh):
            _err(f"Could not load {p} as a triangle mesh.")
        return m
    except Exception as e:
        _err(f"Failed to load '{p}': {e}")


def _write_stl(path: Path, mesh: trimesh.Trimesh) -> None:
    """Export as binary STL."""
    mesh.export(str(path), file_type="stl")


def _backup(path: Path) -> Path:
    backup = path.with_name(path.stem + "_original" + path.suffix)
    shutil.copy2(path, backup)
    return backup


# ─────────────────────────────────────────────────────────────────────────────
# Mesh statistics
# ─────────────────────────────────────────────────────────────────────────────

def _stats(mesh: trimesh.Trimesh, path: Path) -> dict:
    bb  = mesh.bounding_box.extents          # [dx, dy, dz]
    ctr = mesh.bounding_box.centroid         # [cx, cy, cz]
    mn  = mesh.vertices.min(axis=0)
    mx  = mesh.vertices.max(axis=0)
    max_dim = float(bb.max())

    # Unit guess from largest dimension
    if max_dim > 500:
        unit_guess, unit_scale = "mm", 0.001
    elif max_dim > 5:
        unit_guess, unit_scale = "mm", 0.001
    elif max_dim > 0.5:
        unit_guess, unit_scale = "m", 1.0
    else:
        unit_guess, unit_scale = "m", 1.0

    try:
        watertight = mesh.is_watertight
    except Exception:
        watertight = False

    file_mb = path.stat().st_size / 1_048_576 if path.exists() else 0.0

    return dict(
        faces      = len(mesh.faces),
        vertices   = len(mesh.vertices),
        bb_extents = bb,
        bb_min     = mn,
        bb_max     = mx,
        centroid   = ctr,
        max_dim    = max_dim,
        unit_guess = unit_guess,
        unit_scale = unit_scale,
        watertight = watertight,
        file_mb    = file_mb,
        mujoco_ok  = len(mesh.faces) <= MUJOCO_LIMIT,
    )


def _parse_size(size_str: str | None, faces_int: int | None) -> int | None:
    """Resolve --size / --faces into a target face count."""
    if faces_int is not None:
        return faces_int
    if size_str is None:
        return None
    key = size_str.strip().lower()
    if key in SIZE_PRESETS:
        return SIZE_PRESETS[key]
    try:
        return int(size_str)
    except ValueError:
        _err(
            f"Unknown size preset '{size_str}'.\n"
            f"  Choose a number or one of: {', '.join(SIZE_PRESETS.keys())}"
        )


def _parse_density(d: str) -> float:
    """Accept a material name or numeric kg/m³ value."""
    key = d.strip().lower()
    if key in MATERIAL_DENSITY:
        return float(MATERIAL_DENSITY[key])
    try:
        return float(d)
    except ValueError:
        _err(
            f"Unknown density '{d}'.\n"
            f"  Use kg/m³ number or one of: {', '.join(MATERIAL_DENSITY.keys())}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core processing functions
# ─────────────────────────────────────────────────────────────────────────────

def _do_reduce(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    """Decimate mesh to target face count."""
    try:
        import fast_simplification as fs
    except ImportError:
        _err("fast-simplification not installed.\n  Run: pip install fast-simplification")

    ratio = max(0.001, 1.0 - (target / len(mesh.faces)))
    pts_out, faces_out = fs.simplify(
        np.array(mesh.vertices, dtype=np.float32),
        np.array(mesh.faces,    dtype=np.int32),
        target_reduction=ratio,
        verbose=False,
    )
    return trimesh.Trimesh(vertices=pts_out, faces=faces_out, process=False)


def _do_center(mesh: trimesh.Trimesh, mode: str, xyz: tuple | None) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Translate mesh so its origin moves to the desired point. Returns (mesh, offset)."""
    if mode == "bbox":
        offset = -mesh.bounding_box.centroid
    elif mode == "geom":
        offset = -mesh.centroid
    elif mode == "bottom":
        # XY bbox-centre, Z at lowest vertex
        cx, cy, _ = mesh.bounding_box.centroid
        offset = np.array([-cx, -cy, -mesh.vertices[:, 2].min()])
    elif mode == "custom" and xyz is not None:
        offset = np.array(xyz)
    else:
        offset = np.zeros(3)
    result = mesh.copy()
    result.vertices += offset
    return result, offset


def _do_hull(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Generate convex hull as a new Trimesh."""
    return mesh.convex_hull


def _do_scale(mesh: trimesh.Trimesh, factor: float) -> trimesh.Trimesh:
    result = mesh.copy()
    result.vertices *= factor
    return result


def _do_inertia(mesh: trimesh.Trimesh, density: float, unit: str) -> dict:
    """
    Compute mass and inertia tensor.
    Mesh should be in metres (pass unit='mm' to auto-convert).
    """
    m = mesh.copy()
    scale = {"mm": 0.001, "cm": 0.01, "inch": 0.0254, "m": 1.0}.get(unit, 1.0)
    m.vertices *= scale          # → metres for physics calcs

    if not m.is_watertight:
        volume = None
        mass   = None
        inertia = None
    else:
        volume  = abs(m.volume)          # m³
        mass    = volume * density       # kg
        inertia = m.moment_inertia * density  # kg·m²

    return dict(
        watertight = m.is_watertight,
        volume     = volume,
        mass       = mass,
        inertia    = inertia,
        density    = density,
        unit_used  = unit,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI — main group
# ─────────────────────────────────────────────────────────────────────────────

CONTEXT_SETTINGS = dict(help_option_names=["--help", "-h"], max_content_width=90)


@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS,
             help=__doc__)
@click.pass_context
def cli(ctx: click.Context) -> None:
    if ctx.invoked_subcommand is None:
        _interactive_launcher()


# ─────────────────────────────────────────────────────────────────────────────
# reduce
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--size",   "-s", default=None,
              metavar="PRESET|N",
              help="Target face count — a number or preset: "
                   "ultra(8K) low(20K) medium(50K) high(80K) "
                   "mujoco(80K) isaac(100K) physics(8K)")
@click.option("--faces",  "-f", type=int,  default=None, help="Explicit target face count (overrides --size)")
@click.option("--output", "-o", default=None,            help="Output path  [default: overwrite]")
@click.option("--backup/--no-backup", default=True,      help="Save _original backup before overwriting")
def reduce(input: str, size: str, faces: int, output: str, backup: bool) -> None:
    """Decimate a mesh to a lower polygon count.

    \b
    Examples
    --------
      python mesh_toolkit.py reduce wheel.stl --size mujoco
      python mesh_toolkit.py reduce wheel.stl --size 50000
      python mesh_toolkit.py reduce wheel.stl --faces 30000 --output wheel_30k.stl
    """
    _logo()
    _section("Reduce  —  Mesh Decimation")

    src   = Path(input)
    mesh  = _load(src)
    s0    = _stats(mesh, src)
    target = _parse_size(size, faces)

    # Interactive fallback if no target given
    if target is None:
        console.print(f"  Current face count:  {_fmt_faces(s0['faces'])}\n")
        console.print(f"  [dim]Presets[/]  ultra {SIZE_PRESETS['ultra']:,}  ·  "
                      f"low {SIZE_PRESETS['low']:,}  ·  "
                      f"medium {SIZE_PRESETS['medium']:,}  ·  "
                      f"high {SIZE_PRESETS['high']:,}  ·  "
                      f"mujoco {SIZE_PRESETS['mujoco']:,}  ·  "
                      f"physics {SIZE_PRESETS['physics']:,}")
        raw = Prompt.ask("  Target faces (number or preset name)",
                         default="mujoco", console=console)
        target = _parse_size(raw, None)

    if target >= s0["faces"]:
        _warn(f"Target {target:,} ≥ current {s0['faces']:,} — nothing to do.")
        return

    dst = Path(output) if output else src

    # Backup
    backup_path = None
    if backup and dst == src:
        backup_path = _backup(src)

    # Reduce with live progress bar
    console.print()
    with Progress(
        SpinnerColumn(spinner_name="dots", style=f"bold {ACCENT}"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=36, style=ACCENT, complete_style=SUCCESS),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as prog:
        task = prog.add_task("  Decimating …", total=100)
        t0 = time.time()
        # Animate progress smoothly while decimation runs (it's not resumable)
        prog.update(task, advance=10)
        result = _do_reduce(mesh, target)
        prog.update(task, advance=90)
        elapsed = time.time() - t0

    _write_stl(dst, result)
    s1 = _stats(result, dst)

    # ── Summary table ──────────────────────────────────────────────────────
    console.print()
    tbl = Table(box=rich_box.ROUNDED, border_style=DIM, show_header=True,
                header_style=f"bold {ACCENT}", padding=(0, 2))
    tbl.add_column("",         style="bold", min_width=12)
    tbl.add_column("Before",   justify="right", min_width=14)
    tbl.add_column("After",    justify="right", min_width=14)
    tbl.add_column("Δ",        justify="right", min_width=10)

    pct_faces = (1 - s1["faces"] / s0["faces"]) * 100
    pct_size  = (1 - s1["file_mb"]  / s0["file_mb"])  * 100 if s0["file_mb"] > 0 else 0

    tbl.add_row("Faces",    f"{s0['faces']:,}",    _fmt_faces(s1["faces"]), f"[{SUCCESS}]−{pct_faces:.1f} %[/]")
    tbl.add_row("Vertices", f"{s0['vertices']:,}", f"{s1['vertices']:,}",   "")
    tbl.add_row("File",     f"{s0['file_mb']:.1f} MB", f"{s1['file_mb']:.1f} MB",
                f"[{SUCCESS}]−{pct_size:.0f} %[/]" if pct_size > 0 else "")
    tbl.add_row("Time",     "",                    f"{elapsed:.1f} s", "")

    console.print(Padding(tbl, (0, 2)))
    console.print()
    _ok(f"Saved → {dst}")
    if backup_path:
        _info("Backup", str(backup_path))
    if s1["faces"] > MUJOCO_LIMIT:
        _warn(f"Still above MuJoCo's {MUJOCO_LIMIT:,} face limit — reduce further.")
    _tip(f"Run  inspect {dst.name}  to verify the result.")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# inspect
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
def inspect(input: str) -> None:
    """Detailed analysis of a single mesh file.

    \b
    Reports bounding box, polygon count, watertight status, unit guess,
    and simulator compatibility.
    """
    _logo()
    src  = Path(input)
    _section(f"Inspect  —  {src.name}")

    with console.status(f"[{ACCENT}]Loading …", spinner="dots"):
        mesh = _load(src)
        st   = _stats(mesh, src)

    # ── Geometry panel ─────────────────────────────────────────────────────
    console.print(f"  [bold]Geometry[/]")
    _info("Faces",     f"{_fmt_faces(st['faces'])}")
    _info("Vertices",  f"{st['vertices']:,}")
    _info("File size", f"{st['file_mb']:.2f} MB  ({src.stat().st_size:,} bytes)")
    _info("Watertight",f"[{SUCCESS}]Yes[/]" if st['watertight'] else f"[{WARN}]No[/]  (inertia may be inaccurate)")

    console.print()
    console.print(f"  [bold]Bounding Box[/]")
    bb = st['bb_extents']
    mn = st['bb_min']
    mx = st['bb_max']
    ctr = st['centroid']
    unit = st['unit_guess']
    _info("Extents",  f"{bb[0]:.2f} × {bb[1]:.2f} × {bb[2]:.2f}  {unit}")
    _info("Min XYZ",  f"[{MUTED}]{mn[0]:.3f}  {mn[1]:.3f}  {mn[2]:.3f}[/]")
    _info("Max XYZ",  f"[{MUTED}]{mx[0]:.3f}  {mx[1]:.3f}  {mx[2]:.3f}[/]")
    _info("Centroid", f"[{MUTED}]{ctr[0]:.3f}  {ctr[1]:.3f}  {ctr[2]:.3f}[/]")
    _info("Origin offset", _fmt_origin_offset(ctr))

    console.print()
    console.print(f"  [bold]Unit Detection[/]")
    _info("Largest dim",   f"{st['max_dim']:.2f}")
    _info("Likely units",  f"[bold {ACCENT}]{unit}[/]"
          + (f"  → scale ×{st['unit_scale']} to convert to metres" if unit != "m" else "  (already SI)"))

    # ── Simulator compatibility ─────────────────────────────────────────────
    console.print()
    console.print(f"  [bold]Simulator Compatibility[/]")
    mj_ok  = st["faces"] <= MUJOCO_LIMIT
    mj_rec = st["faces"] <= 80_000
    is_ok  = st["faces"] <= ISAAC_LIMIT

    _info("MuJoCo limit",    _fmt_status(mj_ok,  f"≤ {MUJOCO_LIMIT:,} faces", f"OVER LIMIT ({st['faces']:,} > {MUJOCO_LIMIT:,})"))
    _info("MuJoCo visual",   _fmt_status(mj_rec, "≤ 80 K  (recommended)", f"⚠ {st['faces']:,} > 80 K — may affect performance"))
    _info("Isaac Sim",       _fmt_status(is_ok,  "OK", f"OVER LIMIT ({st['faces']:,} > {ISAAC_LIMIT:,})"))
    _info("Collision hull",  _fmt_status(st["faces"] <= 10_000, "≤ 10 K  (fast collision)", f"Consider --hull  ({st['faces']:,} faces)"))

    # ── Suggestions ────────────────────────────────────────────────────────
    console.print()
    if not mj_ok:
        _warn(f"Face count exceeds MuJoCo's hard limit of {MUJOCO_LIMIT:,}.")
        _tip(f"python mesh_toolkit.py reduce {src.name} --size mujoco")
    elif not mj_rec:
        _warn("Face count is within limit but above the recommended 80K visual target.")
        _tip(f"python mesh_toolkit.py reduce {src.name} --size mujoco")
    else:
        _ok("Ready for MuJoCo.")

    if unit != "m":
        _tip(f"python mesh_toolkit.py scale {src.name} --from-unit {unit}")
    console.print()


def _fmt_origin_offset(ctr: np.ndarray) -> str:
    d = float(np.linalg.norm(ctr))
    if d < 0.5:
        return f"[{SUCCESS}]{d:.3f}  (origin near centroid — good)[/]"
    return f"[{WARN}]{d:.3f}  (origin offset from centroid — consider centering)[/]"


# ─────────────────────────────────────────────────────────────────────────────
# center
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--mode", "-m",
              type=click.Choice(["bbox", "geom", "bottom", "custom"], case_sensitive=False),
              default="bbox",
              help="Origin placement: bbox=bounding-box centre  geom=mesh centroid  "
                   "bottom=XY-bbox-centre + Z-at-lowest  custom=manual XYZ offset")
@click.option("--xyz", nargs=3, type=float, default=None, metavar="X Y Z",
              help="Shift to apply when --mode custom  (in mesh units)")
@click.option("--output", "-o", default=None, help="Output path  [default: overwrite]")
@click.option("--backup/--no-backup", default=True)
def center(input: str, mode: str, xyz: tuple, output: str, backup: bool) -> None:
    """Reposition mesh origin.

    \b
    Modes
    -----
      bbox    Move origin to bounding-box centre         (default)
      geom    Move origin to volume centroid
      bottom  Move origin to XY-centre, Z=floor of mesh
      custom  Apply a manual XYZ translation

    \b
    Examples
    --------
      python mesh_toolkit.py center part.stl --mode bbox
      python mesh_toolkit.py center part.stl --mode bottom
      python mesh_toolkit.py center part.stl --mode custom --xyz 0 0 -55
    """
    _logo()
    src  = Path(input)
    _section(f"Center  —  {src.name}")

    mesh = _load(src)
    s0   = _stats(mesh, src)

    if mode == "custom" and xyz is None:
        _warn("--mode custom requires --xyz X Y Z")
        raw = Prompt.ask("  Enter offset X Y Z (space-separated, mesh units)",
                         console=console)
        try:
            xyz = tuple(float(v) for v in raw.split())
        except ValueError:
            _err("Invalid XYZ input.")

    result, offset = _do_center(mesh, mode, xyz)
    s1 = _stats(result, src)

    dst = Path(output) if output else src
    if backup and dst == src:
        _backup(src)

    _write_stl(dst, result)

    console.print(f"  Mode   :  [bold {ACCENT}]{mode}[/]")
    console.print(f"  Offset :  [{MUTED}]{offset[0]:+.4f}  {offset[1]:+.4f}  {offset[2]:+.4f}[/]  (mesh units)")
    console.print()
    _info("Old centroid",  f"[{MUTED}]{s0['centroid'][0]:.3f}  {s0['centroid'][1]:.3f}  {s0['centroid'][2]:.3f}[/]")
    _info("New centroid",  f"[{MUTED}]{s1['centroid'][0]:.3f}  {s1['centroid'][1]:.3f}  {s1['centroid'][2]:.3f}[/]")
    console.print()
    _ok(f"Saved → {dst}")
    _tip("Verify in MuJoCo: the green origin site should align with your joint axis.")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# hull
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", default=None,
              help="Output path  [default: <name>_hull.stl]")
def hull(input: str, output: str) -> None:
    """Generate a convex collision hull from a visual mesh.

    The hull is a simpler, convex approximation used for fast collision
    detection in MuJoCo (group=1) while the original is kept for visuals
    (group=2).

    \b
    Example
    -------
      python mesh_toolkit.py hull wheel.stl
      # → wheel_hull.stl  (a few hundred faces, fast collision)
    """
    _logo()
    src = Path(input)
    _section(f"Hull  —  {src.name}")

    with console.status(f"[{ACCENT}]Computing convex hull …", spinner="dots"):
        mesh  = _load(src)
        h     = _do_hull(mesh)

    dst = Path(output) if output else src.with_name(src.stem + "_hull" + src.suffix)
    _write_stl(dst, h)

    s0 = _stats(mesh, src)
    s1 = _stats(h, dst)

    tbl = Table(box=rich_box.ROUNDED, border_style=DIM, show_header=True,
                header_style=f"bold {ACCENT}", padding=(0, 2))
    tbl.add_column("",         style="bold", min_width=12)
    tbl.add_column("Visual",   justify="right", min_width=14)
    tbl.add_column("Hull",     justify="right", min_width=14)

    pct = (1 - s1["faces"] / s0["faces"]) * 100
    tbl.add_row("Faces",    f"{s0['faces']:,}", f"[{SUCCESS}]{s1['faces']:,}[/]",)
    tbl.add_row("Vertices", f"{s0['vertices']:,}", f"{s1['vertices']:,}")
    tbl.add_row("File",     f"{s0['file_mb']:.1f} MB", f"{s1['file_mb']:.2f} MB")
    tbl.add_row("Reduction","", f"[{SUCCESS}]−{pct:.1f} %[/]")

    console.print(Padding(tbl, (0, 2)))
    console.print()
    _ok(f"Hull saved → {dst}")
    _tip(
        "In your MJCF use group='2' for the visual mesh and group='1' for the hull:\n"
        f"  <geom type='mesh' mesh='mypart' group='2' contype='0'/>\n"
        f"  <geom type='mesh' mesh='mypart_hull' group='1' contype='1' material='invisible'/>"
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# scale
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--from-unit", "-u",
              type=click.Choice(["mm", "cm", "inch", "m"], case_sensitive=False),
              default=None,
              help="Source unit  [auto-detected if omitted]")
@click.option("--to-unit", "-t",
              type=click.Choice(["mm", "cm", "m"], case_sensitive=False),
              default="m",
              help="Target unit  [default: m]")
@click.option("--factor", "-f", type=float, default=None,
              help="Custom scale factor (overrides unit conversion)")
@click.option("--output", "-o", default=None,
              help="Output path  [default: overwrite]")
@click.option("--backup/--no-backup", default=True)
def scale(input: str, from_unit: str, to_unit: str, factor: float,
          output: str, backup: bool) -> None:
    """Apply a unit conversion or custom scale factor.

    \b
    Examples
    --------
      python mesh_toolkit.py scale part.stl --from-unit mm
      python mesh_toolkit.py scale part.stl --from-unit mm --to-unit m
      python mesh_toolkit.py scale part.stl --factor 0.001
    """
    _logo()
    src  = Path(input)
    _section(f"Scale  —  {src.name}")

    mesh = _load(src)
    st   = _stats(mesh, src)

    if factor is None:
        if from_unit is None:
            from_unit = st["unit_guess"]
            console.print(f"  [{MUTED}]Auto-detected unit:[/]  [bold {ACCENT}]{from_unit}[/]")
        key = (from_unit.lower(), to_unit.lower())
        factor = SCALE_FACTORS.get(key)
        if factor is None:
            _err(f"No scale factor defined for {from_unit} → {to_unit}.")

    if abs(factor - 1.0) < 1e-9:
        _warn("Scale factor is 1.0 — no change needed.")
        return

    result = _do_scale(mesh, factor)
    s1 = _stats(result, src)

    dst = Path(output) if output else src
    if backup and dst == src:
        _backup(src)
    _write_stl(dst, result)

    _info("Scale factor", f"×{factor}")
    _info("Old extents",  f"{st['bb_extents'][0]:.3f} × {st['bb_extents'][1]:.3f} × {st['bb_extents'][2]:.3f}")
    _info("New extents",  f"{s1['bb_extents'][0]:.3f} × {s1['bb_extents'][1]:.3f} × {s1['bb_extents'][2]:.3f}")
    console.print()
    _ok(f"Saved → {dst}")
    _tip(
        "If you scale meshes individually, set meshdir scale in MJCF instead:\n"
        "  <mesh file='part.stl' scale='0.001 0.001 0.001'/>"
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# inertia
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--density", "-d", default="aluminum",
              metavar="MATERIAL|kg/m³",
              help="Material density — name or kg/m³.  "
                   "Names: " + " | ".join(MATERIAL_DENSITY.keys()))
@click.option("--unit", "-u",
              type=click.Choice(["mm", "cm", "inch", "m", "auto"], case_sensitive=False),
              default="auto",
              help="Mesh unit  [default: auto-detect]")
def inertia(input: str, density: str, unit: str) -> None:
    """Compute mass and inertia tensor assuming uniform density.

    Outputs a ready-to-paste MuJoCo <inertial> element.
    Requires a watertight (manifold) mesh for accurate results.

    \b
    Examples
    --------
      python mesh_toolkit.py inertia motor.stl --density aluminum
      python mesh_toolkit.py inertia bracket.stl --density 1250 --unit mm
    """
    _logo()
    src = Path(input)
    _section(f"Inertia  —  {src.name}")

    mesh  = _load(src)
    st    = _stats(mesh, src)
    rho   = _parse_density(density)

    if unit == "auto":
        unit = st["unit_guess"]
        _info("Auto-detected unit", f"[bold {ACCENT}]{unit}[/]")

    with console.status(f"[{ACCENT}]Computing …", spinner="dots"):
        result = _do_inertia(mesh, rho, unit)

    _info("Material",   f"{density}  ({rho:.0f} kg/m³)")
    _info("Mesh unit",  unit)
    console.print()

    if not result["watertight"]:
        _warn("Mesh is NOT watertight — inertia results will be approximate.")
        console.print(f"  [{MUTED}]Tip: close holes in your CAD tool, or accept the approximation.[/]")
        console.print()

    if result["mass"] is not None:
        vol  = result["volume"]
        mass = result["mass"]
        I    = result["inertia"]

        _info("Volume",  f"{vol*1e6:.2f} cm³  ({vol:.6f} m³)")
        _info("Mass",    f"[bold]{mass:.4f} kg[/]")
        console.print()

        # MuJoCo inertial block
        ctr  = mesh.centroid
        # Convert centroid from mesh units to metres
        scale_f = {"mm": 0.001, "cm": 0.01, "inch": 0.0254, "m": 1.0}[unit]
        ctr_m = ctr * scale_f

        diag = np.diag(I)
        console.print(f"  [bold]MuJoCo [dim]<inertial>[/] snippet[/]")
        console.print()
        console.print(Panel(
            f'[bold {ACCENT}]<inertial[/]\n'
            f'  [bold {ACCENT}]mass=[/]"[{SUCCESS}]{mass:.4f}[/]"\n'
            f'  [bold {ACCENT}]pos=[/]"[{SUCCESS}]{ctr_m[0]:.5f} {ctr_m[1]:.5f} {ctr_m[2]:.5f}[/]"\n'
            f'  [bold {ACCENT}]diaginertia=[/]"[{SUCCESS}]{diag[0]:.6e} {diag[1]:.6e} {diag[2]:.6e}[/]"[bold {ACCENT}]/>[/]',
            border_style=DIM,
            padding=(0, 2),
        ))
        console.print()
        _tip("Paste this inside the matching <body> in your MJCF XML.")
    else:
        _warn("Cannot compute volume on non-watertight mesh.  Fix mesh geometry first.")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# split
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--min-faces", type=int, default=100,
              help="Ignore components with fewer than N faces  [default: 100]")
@click.option("--output-dir", "-o", default=None,
              help="Directory for output files  [default: same as input]")
def split(input: str, min_faces: int, output_dir: str) -> None:
    """Split a multi-body assembly STL into individual component files.

    Useful when a CAD assembly was exported as a single STL but contains
    multiple disconnected components that need separate MJCF body entries.

    \b
    Example
    -------
      python mesh_toolkit.py split assembly.stl
      # → assembly_body_00.stl  assembly_body_01.stl  …
    """
    _logo()
    src = Path(input)
    _section(f"Split  —  {src.name}")

    mesh = _load(src)
    out_dir = Path(output_dir) if output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with console.status(f"[{ACCENT}]Identifying connected components …", spinner="dots"):
        components = mesh.split(only_watertight=False)

    if isinstance(components, trimesh.Trimesh):
        components = [components]

    # Filter tiny noise components
    components = [c for c in components if len(c.faces) >= min_faces]
    components.sort(key=lambda c: len(c.faces), reverse=True)

    if len(components) == 1:
        _warn("Only one component found — mesh is already a single body.")
        return

    console.print(f"  Found [bold {SUCCESS}]{len(components)}[/] components\n")

    tbl = Table(box=rich_box.SIMPLE_HEAD, border_style=DIM,
                header_style=f"bold {ACCENT}", padding=(0, 2))
    tbl.add_column("#",     justify="right")
    tbl.add_column("Faces", justify="right")
    tbl.add_column("Extents (X×Y×Z)",)
    tbl.add_column("Output file")

    for i, comp in enumerate(components):
        out_name = out_dir / f"{src.stem}_body_{i:02d}{src.suffix}"
        _write_stl(out_name, comp)
        bb = comp.bounding_box.extents
        ext_str = f"{bb[0]:.1f}×{bb[1]:.1f}×{bb[2]:.1f}"
        tbl.add_row(str(i), f"{len(comp.faces):,}", ext_str, out_name.name)

    console.print(Padding(tbl, (0, 2)))
    console.print()
    _ok(f"Saved {len(components)} component files → {out_dir}")
    _tip("Each output file maps to one <body> in your MJCF with a <geom mesh='…'/>.")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# check
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, default=False,
              help="Recurse into sub-directories")
def check(path: str, recursive: bool) -> None:
    """Pre-flight digital twin readiness check for a mesh directory.

    Reports face counts, file sizes, unit detection, watertight status,
    and simulator compatibility for all STL files found.

    \b
    Examples
    --------
      python mesh_toolkit.py check ./assets/meshes/base
      python mesh_toolkit.py check ./assets -r
    """
    _logo()
    p = Path(path)
    pattern = "**/*.stl" if recursive else "*.stl"
    files   = sorted(p.glob(pattern)) if p.is_dir() else [p]

    if not files:
        _warn(f"No STL files found in {p}")
        return

    _section(f"Pre-flight Check  —  {p}")

    tbl = Table(box=rich_box.ROUNDED, border_style=DIM, show_header=True,
                header_style=f"bold {ACCENT}", padding=(0, 1))
    tbl.add_column("File",           no_wrap=True, min_width=22)
    tbl.add_column("Faces",          justify="right", min_width=10)
    tbl.add_column("Size",           justify="right", min_width=9)
    tbl.add_column("Units",          justify="center", min_width=7)
    tbl.add_column("Watertight",     justify="center", min_width=11)
    tbl.add_column("MuJoCo",         justify="center", min_width=12)
    tbl.add_column("Isaac Sim",      justify="center", min_width=12)

    issues     = []
    total_mb   = 0.0
    total_face = 0

    with Progress(
        SpinnerColumn(spinner_name="dots", style=f"bold {ACCENT}"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, style=ACCENT, complete_style=SUCCESS),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task("  Analysing meshes …", total=len(files))
        rows = []
        for f in files:
            try:
                mesh = _load(f)
                st   = _stats(mesh, f)
            except SystemExit:
                rows.append((f.name, "[red]error[/]", "", "", "", "", ""))
                prog.advance(task)
                continue

            total_mb   += st["file_mb"]
            total_face += st["faces"]

            mj_ok  = st["faces"] <= MUJOCO_LIMIT
            mj_rec = st["faces"] <= 80_000
            is_ok  = st["faces"] <= ISAAC_LIMIT
            wt     = st["watertight"]

            mj_str  = (f"[{SUCCESS}]✓  Ready[/]" if mj_ok and mj_rec
                       else f"[{WARN}]⚠  Large[/]"  if mj_ok
                       else f"[{ERR}]✗  Over limit[/]")
            is_str  = f"[{SUCCESS}]✓  OK[/]" if is_ok else f"[{ERR}]✗  Over limit[/]"
            wt_str  = f"[{SUCCESS}]✓[/]" if wt else f"[{WARN}]~[/]"
            unit_st = f"[bold {ACCENT}]{st['unit_guess']}[/]"

            rows.append((f.name, _fmt_faces(st["faces"]), f"{st['file_mb']:.1f} MB",
                         unit_st, wt_str, mj_str, is_str))

            if not mj_ok:
                issues.append(f"[bold {ERR}]{f.name}[/]: {st['faces']:,} faces — OVER MuJoCo limit ({MUJOCO_LIMIT:,})")
            elif not mj_rec:
                issues.append(f"[bold {WARN}]{f.name}[/]: {st['faces']:,} faces — recommend reducing to ≤ 80K")

            prog.advance(task)

    for row in rows:
        tbl.add_row(*row)

    console.print(Padding(tbl, (0, 2)))
    console.print()
    _info("Total meshes", str(len(files)))
    _info("Total faces",  f"{total_face:,}")
    _info("Total size",   f"{total_mb:.1f} MB")
    console.print()

    if issues:
        console.print(f"  [bold {WARN}]Issues found:[/]")
        for issue in issues:
            console.print(f"    • {issue}")
        console.print()
        _tip(f"python mesh_toolkit.py batch {p}  --size mujoco")
    else:
        _ok("All meshes pass digital twin readiness checks.")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# batch
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--size",      "-s", default="mujoco", metavar="PRESET|N",
              help="Target face count — number or preset  [default: mujoco=80K]")
@click.option("--faces",     "-f", type=int, default=None,
              help="Explicit target face count (overrides --size)")
@click.option("--pattern",   "-p", default="*.stl",
              help="Glob pattern for input files  [default: *.stl]")
@click.option("--output-dir","-o", default=None,
              help="Output directory  [default: overwrite in-place]")
@click.option("--skip-small/--no-skip-small", default=True,
              help="Skip files already under the target face count  [default: True]")
@click.option("--backup/--no-backup", default=True)
def batch(directory: str, size: str, faces: int, pattern: str,
          output_dir: str, skip_small: bool, backup: bool) -> None:
    """Batch-decimate all meshes in a directory.

    \b
    Examples
    --------
      python mesh_toolkit.py batch ./assets/meshes/base --size mujoco
      python mesh_toolkit.py batch ./assets/meshes     --faces 30000 --output-dir ./reduced
      python mesh_toolkit.py batch ./assets/meshes/base --size physics --pattern "*_visual*"
    """
    _logo()
    src_dir = Path(directory)
    target  = _parse_size(size, faces)
    files   = sorted(src_dir.glob(pattern))

    if not files:
        _warn(f"No files matching '{pattern}' in {src_dir}")
        return

    out_dir = Path(output_dir) if output_dir else src_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _section(f"Batch Reduce  —  {src_dir.name}")
    console.print(f"  Target  : [bold {ACCENT}]{target:,} faces[/]  ({size})\n"
                  f"  Files   : {len(files)}\n"
                  f"  Output  : {out_dir}\n")

    tbl = Table(box=rich_box.SIMPLE_HEAD, border_style=DIM,
                header_style=f"bold {ACCENT}", padding=(0, 1))
    tbl.add_column("File",      no_wrap=True, min_width=24)
    tbl.add_column("Before",    justify="right", min_width=10)
    tbl.add_column("After",     justify="right", min_width=10)
    tbl.add_column("Reduction", justify="right", min_width=10)
    tbl.add_column("Status",    justify="center", min_width=10)

    with Progress(
        SpinnerColumn(spinner_name="dots", style=f"bold {ACCENT}"),
        TextColumn("[progress.description]{task.description:<28}"),
        BarColumn(bar_width=28, style=ACCENT, complete_style=SUCCESS),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task("  Processing …", total=len(files))

        for f in files:
            prog.update(task, description=f"  {f.name[:26]:<26}")
            try:
                mesh = _load(f)
                orig = len(mesh.faces)

                if skip_small and orig <= target:
                    tbl.add_row(f.name, f"{orig:,}", f"[{MUTED}]—[/]",
                                f"[{MUTED}]—[/]", f"[{MUTED}]Skipped[/]")
                    prog.advance(task)
                    continue

                dst_path = out_dir / f.name
                if backup and dst_path.exists() and dst_path == f:
                    _backup(f)

                result = _do_reduce(mesh, target)
                _write_stl(dst_path, result)
                after  = len(result.faces)
                pct    = (1 - after / orig) * 100

                tbl.add_row(f.name, f"{orig:,}", _fmt_faces(after),
                            f"[{SUCCESS}]−{pct:.0f}%[/]",
                            f"[{SUCCESS}]✓ Done[/]")
            except Exception as e:
                tbl.add_row(f.name, "—", "—", "—", f"[{ERR}]Error[/]")

            prog.advance(task)

    console.print(Padding(tbl, (0, 2)))
    console.print()
    _ok(f"Batch complete  →  {out_dir}")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# view
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--show-axes/--no-axes", default=True, help="Show XYZ axis widget")
def view(input: str, show_axes: bool) -> None:
    """Open a mesh in an interactive 3D viewer window.

    Requires a display (not headless).  Controls:
      Mouse drag = rotate  |  Scroll = zoom  |  Right drag = pan  |  Q = quit

    \b
    Example
    -------
      python mesh_toolkit.py view wheel.stl
    """
    _logo()
    src  = Path(input)
    _section(f"View  —  {src.name}")

    mesh = _load(src)
    st   = _stats(mesh, src)

    _info("Faces",   f"{st['faces']:,}")
    _info("Units",   st["unit_guess"])
    _info("Extents", f"{st['bb_extents'][0]:.2f} × {st['bb_extents'][1]:.2f} × {st['bb_extents'][2]:.2f}")
    console.print()
    console.print(f"  [dim]Opening viewer window …  Press Q to quit.[/]\n")

    scene = trimesh.Scene([mesh])
    if show_axes:
        ax = trimesh.creation.axis(axis_length=float(st["max_dim"]) * 0.3)
        scene.add_geometry(ax)

    scene.show(smooth=False)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive launcher  (shown when run with no arguments)
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS = [
    ("reduce",  "Decimate polygon count",               "  Faces ↓   poly budget for simulator"),
    ("inspect", "Analyse geometry & compatibility",      "  Bounding box, watertight, unit guess"),
    ("center",  "Relocate mesh origin",                  "  Align origin to bbox / centroid / floor"),
    ("hull",    "Generate convex collision hull",        "  Fast collision mesh for MuJoCo group=1"),
    ("scale",   "Convert units  (mm → m, etc.)",         "  Apply scale factor or unit conversion"),
    ("inertia", "Compute mass & inertia tensor",         "  Outputs ready-to-paste <inertial> XML"),
    ("check",   "Pre-flight digital twin check",         "  Batch readiness table for a directory"),
    ("batch",   "Batch-process all STLs in a folder",    "  One command, entire mesh library"),
    ("split",   "Split assembly into components",        "  Separate disconnected bodies"),
    ("view",    "Open 3D viewer",                        "  Interactive mesh inspection window"),
]


def _interactive_launcher() -> None:
    _logo()

    tbl = Table(box=rich_box.SIMPLE, border_style=DIM, show_header=False,
                padding=(0, 1))
    tbl.add_column("n",     justify="right",  style=f"bold {ACCENT}", min_width=3)
    tbl.add_column("cmd",   style="bold",     min_width=10)
    tbl.add_column("title", style=HEADING,    min_width=32)
    tbl.add_column("desc",  style=MUTED)

    for i, (cmd, title, desc) in enumerate(_TOOLS, 1):
        tbl.add_row(str(i), cmd, title, desc)

    console.print(Padding(tbl, (0, 4)))
    console.print()

    while True:
        raw = Prompt.ask(
            f"  [bold {ACCENT}]Pick a tool[/]  [dim](1–{len(_TOOLS)}, name, or q to quit)[/]",
            console=console,
        ).strip().lower()

        if raw in ("q", "quit", "exit"):
            console.print(f"\n  [{MUTED}]Goodbye.[/]\n")
            break

        # Resolve number → command name
        cmd_name = None
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(_TOOLS):
                cmd_name = _TOOLS[idx][0]
        else:
            # Try direct name match
            for cmd, _, _ in _TOOLS:
                if raw == cmd:
                    cmd_name = cmd
                    break

        if cmd_name is None:
            _warn(f"'{raw}' not recognised.  Enter a number 1–{len(_TOOLS)} or a command name.")
            continue

        console.print()

        # Ask for the input file interactively
        if cmd_name in ("check", "batch"):
            path = Prompt.ask(
                f"  [{ACCENT}]Directory path[/]  [dim](or . for current dir)[/]",
                default=".", console=console,
            )
            args = [cmd_name, path]
        elif cmd_name == "view":
            path = Prompt.ask(f"  [{ACCENT}]Mesh file path[/]", console=console)
            args = [cmd_name, path]
        else:
            path = Prompt.ask(f"  [{ACCENT}]Mesh file path[/]", console=console)
            args = [cmd_name, path]

        # For reduce: ask size if not given
        if cmd_name == "reduce":
            sz = Prompt.ask(
                f"  [{ACCENT}]Size preset or face count[/]  "
                f"[dim]ultra·low·medium·high·mujoco·isaac·physics or a number[/]",
                default="mujoco", console=console,
            )
            args += ["--size", sz]

        elif cmd_name == "center":
            mode = Prompt.ask(
                f"  [{ACCENT}]Mode[/]  [dim]bbox · geom · bottom · custom[/]",
                default="bbox", console=console,
            )
            args += ["--mode", mode]

        elif cmd_name == "scale":
            fu = Prompt.ask(
                f"  [{ACCENT}]From unit[/]  [dim]mm · cm · inch · m[/]",
                default="mm", console=console,
            )
            args += ["--from-unit", fu]

        elif cmd_name == "inertia":
            mat = Prompt.ask(
                f"  [{ACCENT}]Material[/]  "
                f"[dim]aluminum · steel · abs · pla · petg · wood · or kg/m³[/]",
                default="aluminum", console=console,
            )
            args += ["--density", mat]

        elif cmd_name == "batch":
            sz = Prompt.ask(
                f"  [{ACCENT}]Size preset[/]  "
                f"[dim]ultra·low·medium·high·mujoco·isaac·physics or a number[/]",
                default="mujoco", console=console,
            )
            args += ["--size", sz]

        console.print()
        # Invoke the chosen command via Click's standalone mode
        try:
            cli.main(args=args, standalone_mode=False)
        except SystemExit:
            pass
        except Exception as exc:
            _warn(f"Command failed: {exc}")

        console.print()
        again = Confirm.ask(
            f"  [dim]Run another tool?[/]", default=True, console=console
        )
        if not again:
            console.print(f"\n  [{MUTED}]Goodbye.[/]\n")
            break
        console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
