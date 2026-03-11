#!/usr/bin/env python3
"""Visualize rigid-body motion from a .tak file in Open3D.

This script reads rigid-body definitions (name + local marker layout) from
`Nodes.dat` inside a `.tak` file, then animates selected rigid bodies from a
CSV containing scalar columns:
  <name>.tx, <name>.ty, <name>.tz, <name>.qx, <name>.qy, <name>.qz, <name>.qw
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from read_tak import CompoundFile, extract_utf16_chunks, parse_assets_list, parse_property_pairs


@dataclass(frozen=True)
class RigidBodyDefinition:
    name: str
    color_rgb: tuple[float, float, float]
    local_markers: list[tuple[float, float, float]]


@dataclass
class RigidBodyMotion:
    tx: list[float]
    ty: list[float]
    tz: list[float]
    qx: list[float]
    qy: list[float]
    qz: list[float]
    qw: list[float]

    @property
    def frame_count(self) -> int:
        return len(self.tx)

    def pose_at(self, frame_idx: int) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        return (
            (self.tx[frame_idx], self.ty[frame_idx], self.tz[frame_idx]),
            (self.qx[frame_idx], self.qy[frame_idx], self.qz[frame_idx], self.qw[frame_idx]),
        )


@dataclass
class SceneRigidBody:
    definition: RigidBodyDefinition
    motion: RigidBodyMotion
    point_cloud: Any
    line_set: Any | None
    axis_mesh: Any
    previous_transform: list[list[float]]


def parse_vector3(text: str) -> tuple[float, float, float] | None:
    parts = [item.strip() for item in text.split(",")]
    if len(parts) != 3:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None


def argb_to_rgb01(argb_value: int | None) -> tuple[float, float, float]:
    if argb_value is None:
        return (0.8, 0.8, 0.8)
    r = (argb_value >> 16) & 0xFF
    g = (argb_value >> 8) & 0xFF
    b = argb_value & 0xFF
    return (r / 255.0, g / 255.0, b / 255.0)


def hsv_color(idx: int, total: int) -> tuple[float, float, float]:
    hue = idx / max(1, total)
    return colorsys.hsv_to_rgb(hue, 0.7, 0.95)


def parse_rigidbody_definitions(nodes_text: str) -> list[RigidBodyDefinition]:
    rigidbody_pattern = re.compile(
        r'RigidBody\s+"properties":\s*\[(?P<props>.*?)\](?P<body>.*?)(?=(?:RigidBody\s+"properties":)|\Z)',
        re.S,
    )
    node_name_pattern = re.compile(r'"name":\s*"NodeName"\s*,\s*"value":\s*"([^"]+)"')
    color_pattern = re.compile(r'"name":\s*"Color"\s*,\s*"value":\s*"(\d+)"')
    constraint_pattern = re.compile(r'Constraint\s+"properties":\s*\[(?P<cprops>.*?)\]', re.S)
    constraint_name_pattern = re.compile(r'"name":\s*"NodeName"\s*,\s*"value":\s*"(Marker\d+)"')
    constraint_pos_pattern = re.compile(r'"name":\s*"ConstraintPosition"\s*,\s*"value":\s*"([-+0-9eE., ]+)"')

    definitions: list[RigidBodyDefinition] = []
    for rb_match in rigidbody_pattern.finditer(nodes_text):
        props_text = rb_match.group("props")
        body_text = rb_match.group("body")

        name_match = node_name_pattern.search(props_text)
        if name_match is None:
            continue
        name = name_match.group(1).strip()

        color_match = color_pattern.search(props_text)
        color_value = int(color_match.group(1)) if color_match is not None else None

        local_markers: list[tuple[float, float, float]] = []
        for constraint_match in constraint_pattern.finditer(body_text):
            cprops = constraint_match.group("cprops")
            if constraint_name_pattern.search(cprops) is None:
                continue
            pos_match = constraint_pos_pattern.search(cprops)
            if pos_match is None:
                continue
            marker_pos = parse_vector3(pos_match.group(1))
            if marker_pos is None:
                continue
            local_markers.append(marker_pos)

        if not local_markers:
            continue

        definitions.append(
            RigidBodyDefinition(
                name=name,
                color_rgb=argb_to_rgb01(color_value),
                local_markers=local_markers,
            )
        )

    return definitions


def read_assets_from_metadata(cfb: CompoundFile) -> list[str]:
    meta = cfb.stream_entry("MetaData.dat")
    if meta is None:
        return []
    meta_text = "\n".join(extract_utf16_chunks(cfb.read_stream("MetaData.dat"), min_chars=3))
    props = parse_property_pairs(meta_text)
    return parse_assets_list(props.get("Assets"))


def normalize_quaternion(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    qx, qy, qz, qw = q
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (qx / norm, qy / norm, qz / norm, qw / norm)


def quat_to_rot3(q: tuple[float, float, float, float]) -> list[list[float]]:
    qx, qy, qz, qw = normalize_quaternion(q)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def pose_to_transform(
    translation: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> list[list[float]]:
    rot = quat_to_rot3(quaternion)
    tx, ty, tz = translation
    return [
        [rot[0][0], rot[0][1], rot[0][2], tx],
        [rot[1][0], rot[1][1], rot[1][2], ty],
        [rot[2][0], rot[2][1], rot[2][2], tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rigid_inverse(transform: list[list[float]]) -> list[list[float]]:
    r = [[transform[i][j] for j in range(3)] for i in range(3)]
    t = [transform[0][3], transform[1][3], transform[2][3]]
    rt = [[r[j][i] for j in range(3)] for i in range(3)]
    inv_t = [
        -(rt[0][0] * t[0] + rt[0][1] * t[1] + rt[0][2] * t[2]),
        -(rt[1][0] * t[0] + rt[1][1] * t[1] + rt[1][2] * t[2]),
        -(rt[2][0] * t[0] + rt[2][1] * t[1] + rt[2][2] * t[2]),
    ]
    return [
        [rt[0][0], rt[0][1], rt[0][2], inv_t[0]],
        [rt[1][0], rt[1][1], rt[1][2], inv_t[1]],
        [rt[2][0], rt[2][1], rt[2][2], inv_t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(4))
    return out


def transform_points(
    points: list[tuple[float, float, float]],
    translation: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> list[tuple[float, float, float]]:
    rot = quat_to_rot3(quaternion)
    tx, ty, tz = translation
    world: list[tuple[float, float, float]] = []
    for px, py, pz in points:
        wx = rot[0][0] * px + rot[0][1] * py + rot[0][2] * pz + tx
        wy = rot[1][0] * px + rot[1][1] * py + rot[1][2] * pz + ty
        wz = rot[2][0] * px + rot[2][1] * py + rot[2][2] * pz + tz
        world.append((wx, wy, wz))
    return world


def parse_name_list(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def choose_rigidbodies(
    definitions: list[RigidBodyDefinition],
    requested_names: list[str],
    metadata_assets: list[str],
) -> list[RigidBodyDefinition]:
    by_name = {item.name: item for item in definitions}
    if requested_names:
        missing = [name for name in requested_names if name not in by_name]
        if missing:
            available = ", ".join(sorted(by_name))
            raise ValueError(f"Unknown rigid body name(s): {', '.join(missing)}. Available: {available}")
        return [by_name[name] for name in requested_names]

    ordered: list[RigidBodyDefinition] = []
    for name in metadata_assets:
        if name in by_name:
            ordered.append(by_name[name])
    if ordered:
        return ordered
    return sorted(definitions, key=lambda item: item.name.lower())


def load_motion_csv(csv_path: Path, rigid_body_names: list[str]) -> dict[str, RigidBodyMotion]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Motion CSV not found: {csv_path}")

    required_columns: list[str] = []
    for name in rigid_body_names:
        required_columns.extend(
            [f"{name}.tx", f"{name}.ty", f"{name}.tz", f"{name}.qx", f"{name}.qy", f"{name}.qz", f"{name}.qw"]
        )

    raw_values: dict[str, list[float]] = {column: [] for column in required_columns}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(
                "CSV is missing required columns for selected rigid bodies: "
                + ", ".join(missing)
            )

        for row_idx, row in enumerate(reader, start=2):
            for column in required_columns:
                raw = row.get(column, "")
                try:
                    raw_values[column].append(float(raw))
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid float in column '{column}' at CSV row {row_idx}: {raw!r}") from None

    motions: dict[str, RigidBodyMotion] = {}
    for name in rigid_body_names:
        motions[name] = RigidBodyMotion(
            tx=raw_values[f"{name}.tx"],
            ty=raw_values[f"{name}.ty"],
            tz=raw_values[f"{name}.tz"],
            qx=raw_values[f"{name}.qx"],
            qy=raw_values[f"{name}.qy"],
            qz=raw_values[f"{name}.qz"],
            qw=raw_values[f"{name}.qw"],
        )
    return motions


def build_connection_lines(point_count: int) -> list[list[int]]:
    if point_count < 2:
        return []
    return [[idx, idx + 1] for idx in range(point_count - 1)]


def require_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:
        if sys.version_info >= (3, 13):
            raise SystemExit(
                "open3d does not currently provide wheels for Python 3.13 in this setup.\n"
                "Use Python 3.12 for visualization."
            ) from exc
        raise SystemExit(
            "open3d is required for visualization. Install it first, e.g. `uv add open3d`."
        ) from exc
    return o3d


def build_scene(
    o3d: Any,
    definitions: list[RigidBodyDefinition],
    motions: dict[str, RigidBodyMotion],
    axis_size: float,
    connect_markers: bool,
) -> list[SceneRigidBody]:
    scene_items: list[SceneRigidBody] = []
    total = len(definitions)

    for idx, definition in enumerate(definitions):
        motion = motions[definition.name]
        if motion.frame_count == 0:
            raise ValueError(f"No frames found for rigid body '{definition.name}' in motion CSV")

        initial_translation, initial_quat = motion.pose_at(0)
        initial_points = transform_points(definition.local_markers, initial_translation, initial_quat)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(initial_points)

        color = definition.color_rgb
        if color == (0.8, 0.8, 0.8):
            color = hsv_color(idx, total)
        point_cloud.paint_uniform_color(color)

        line_set = None
        if connect_markers and len(definition.local_markers) >= 2:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(initial_points)
            line_set.lines = o3d.utility.Vector2iVector(build_connection_lines(len(definition.local_markers)))
            line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(definition.local_markers) - 1)])

        axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        initial_transform = pose_to_transform(initial_translation, initial_quat)
        axis_mesh.transform(initial_transform)

        scene_items.append(
            SceneRigidBody(
                definition=definition,
                motion=motion,
                point_cloud=point_cloud,
                line_set=line_set,
                axis_mesh=axis_mesh,
                previous_transform=initial_transform,
            )
        )

    return scene_items


def animate_scene(
    o3d: Any,
    tak_name: str,
    scene_items: list[SceneRigidBody],
    frame_step: int,
    playback_fps: float,
) -> None:
    frame_count = min(item.motion.frame_count for item in scene_items)
    if frame_count <= 0:
        raise ValueError("No animation frames available")

    frame_step = max(1, frame_step)
    playback_fps = max(1e-6, playback_fps)
    sleep_s = 1.0 / playback_fps

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"TAK RigidBody Motion: {tak_name}",
        width=1360,
        height=900,
    )
    for item in scene_items:
        vis.add_geometry(item.point_cloud)
        vis.add_geometry(item.axis_mesh)
        if item.line_set is not None:
            vis.add_geometry(item.line_set)

    world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(world_axis)

    print(f"Animating {frame_count} frames (step={frame_step}, playback_fps={playback_fps:.2f})")
    frame_idx = 0
    while True:
        for item in scene_items:
            translation, quat = item.motion.pose_at(frame_idx)
            world_points = transform_points(item.definition.local_markers, translation, quat)
            item.point_cloud.points = o3d.utility.Vector3dVector(world_points)
            vis.update_geometry(item.point_cloud)

            if item.line_set is not None:
                item.line_set.points = o3d.utility.Vector3dVector(world_points)
                vis.update_geometry(item.line_set)

            transform = pose_to_transform(translation, quat)
            delta = mat4_mul(transform, rigid_inverse(item.previous_transform))
            item.axis_mesh.transform(delta)
            item.previous_transform = transform
            vis.update_geometry(item.axis_mesh)

        if not vis.poll_events():
            break
        vis.update_renderer()
        time.sleep(sleep_s)
        frame_idx = (frame_idx + frame_step) % frame_count

    vis.destroy_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize rigid-body motion from TAK nodes + rigid-body motion CSV columns."
    )
    parser.add_argument("tak_file", type=Path, help="Path to .tak file")
    parser.add_argument(
        "--rigid-bodies",
        type=str,
        default="",
        help="Comma-separated rigid body names to animate (default: metadata Assets order).",
    )
    parser.add_argument(
        "--list-rigid-bodies",
        action="store_true",
        help="List rigid body names parsed from Nodes.dat and exit.",
    )
    parser.add_argument(
        "--motion-csv",
        type=Path,
        default=None,
        help=(
            "CSV with columns <name>.tx/.ty/.tz/.qx/.qy/.qz/.qw for selected rigid bodies. "
            "If omitted, tries <tak_stem>_rigid_body_motion.csv next to the TAK file."
        ),
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Advance this many source frames per rendered frame (default: 1).",
    )
    parser.add_argument(
        "--playback-fps",
        type=float,
        default=120.0,
        help="Playback FPS in the viewer loop (default: 120).",
    )
    parser.add_argument(
        "--axis-size",
        type=float,
        default=0.06,
        help="Per-rigid-body axis size (default: 0.06).",
    )
    parser.add_argument(
        "--connect-markers",
        action="store_true",
        help="Connect rigid-body markers by index order with line segments.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tak_path = args.tak_file.expanduser().resolve()
    if not tak_path.exists():
        print(f"Error: file not found: {tak_path}")
        return 1

    cfb = CompoundFile(tak_path)
    nodes_entry = cfb.stream_entry("Nodes.dat")
    if nodes_entry is None:
        print("Error: Nodes.dat stream not found in TAK file")
        return 1

    nodes_text = "\n".join(extract_utf16_chunks(cfb.read_stream("Nodes.dat"), min_chars=3, max_chunks=300_000))
    definitions = parse_rigidbody_definitions(nodes_text)
    if not definitions:
        print("Error: no rigid body definitions parsed from Nodes.dat")
        return 1

    if args.list_rigid_bodies:
        print("Rigid bodies found:")
        for item in sorted(definitions, key=lambda rb: rb.name.lower()):
            print(f"  - {item.name} (markers: {len(item.local_markers)})")
        return 0

    requested_names = parse_name_list(args.rigid_bodies)
    metadata_assets = read_assets_from_metadata(cfb)
    try:
        selected = choose_rigidbodies(definitions, requested_names, metadata_assets)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    motion_csv = (
        args.motion_csv.expanduser().resolve()
        if args.motion_csv is not None
        else tak_path.with_name(f"{tak_path.stem}_rigid_body_motion.csv")
    )

    try:
        motions = load_motion_csv(motion_csv, [item.name for item in selected])
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading motion data: {exc}")
        print("")
        print("Direct Track*.trk motion decoding from .tak is not implemented in this viewer yet.")
        print("This viewer needs rigid-body scalar time series in CSV columns:")
        print("  <name>.tx, <name>.ty, <name>.tz, <name>.qx, <name>.qy, <name>.qz, <name>.qw")
        print("for each selected rigid body name.")
        return 1

    o3d = require_open3d()
    scene_items = build_scene(
        o3d=o3d,
        definitions=selected,
        motions=motions,
        axis_size=args.axis_size,
        connect_markers=args.connect_markers,
    )

    print(f"Selected rigid bodies: {', '.join(item.definition.name for item in scene_items)}")
    print(f"Motion CSV: {motion_csv}")
    animate_scene(
        o3d=o3d,
        tak_name=tak_path.name,
        scene_items=scene_items,
        frame_step=args.frame_step,
        playback_fps=args.playback_fps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
