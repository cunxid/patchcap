#!/usr/bin/env python3
"""Visualize marker layouts from a .tak file using Open3D.

This viewer extracts marker constraint positions from `Nodes.dat` in the TAK
container and renders them as colored points/spheres.

Note: this script visualizes marker layout constraints (static marker positions
stored in node metadata), not full time-varying trajectories from Track streams.
"""

from __future__ import annotations

import argparse
import colorsys
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from read_tak import CompoundFile, extract_utf16_chunks


@dataclass(frozen=True)
class MarkerConstraint:
    marker_name: str
    marker_index: int
    position: tuple[float, float, float]


def parse_vector3(text: str) -> tuple[float, float, float] | None:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None


def extract_marker_constraints(nodes_text: str) -> list[MarkerConstraint]:
    pattern = re.compile(
        r'"name":\s*"NodeName"\s*,\s*"value":\s*"(?P<name>Marker\d+)"'
        r'(?:(?!"name":\s*"NodeName").)*?'
        r'"name":\s*"ConstraintPosition"\s*,\s*"value":\s*"(?P<pos>[-+0-9eE., ]+)"',
        re.S,
    )

    constraints: list[MarkerConstraint] = []
    for match in pattern.finditer(nodes_text):
        marker_name = match.group("name")
        position = parse_vector3(match.group("pos"))
        if position is None:
            continue
        marker_index = int(marker_name.replace("Marker", ""))
        constraints.append(
            MarkerConstraint(
                marker_name=marker_name,
                marker_index=marker_index,
                position=position,
            )
        )

    return constraints


def split_constraint_groups(constraints: list[MarkerConstraint]) -> list[list[MarkerConstraint]]:
    """Split sequence into groups when marker numbering resets/decreases."""
    if not constraints:
        return []

    groups: list[list[MarkerConstraint]] = []
    current: list[MarkerConstraint] = [constraints[0]]
    prev_index = constraints[0].marker_index

    for marker in constraints[1:]:
        if marker.marker_index <= prev_index:
            groups.append(current)
            current = []
        current.append(marker)
        prev_index = marker.marker_index

    if current:
        groups.append(current)

    return groups


def choose_group(groups: list[list[MarkerConstraint]], selector: str) -> tuple[int, list[MarkerConstraint]]:
    if not groups:
        raise ValueError("No marker groups found")

    if selector == "auto":
        best_idx = max(range(len(groups)), key=lambda i: len({m.marker_name for m in groups[i]}))
        return best_idx, groups[best_idx]

    idx = int(selector)
    if idx < 0 or idx >= len(groups):
        raise ValueError(f"group index out of range: {idx} (available: 0..{len(groups)-1})")
    return idx, groups[idx]


def build_point_cloud(markers: list[MarkerConstraint], o3d: Any) -> Any:
    points = [m.position for m in markers]
    if len(points) == 0:
        return o3d.geometry.PointCloud()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    max_idx = max(m.marker_index for m in markers)
    colors = []
    for marker in markers:
        hue = (marker.marker_index - 1) / max(1, max_idx)
        rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        colors.append(rgb)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def build_spheres(markers: list[MarkerConstraint], radius: float, o3d: Any) -> list[Any]:
    meshes: list[Any] = []
    if radius <= 0:
        return meshes

    max_idx = max(m.marker_index for m in markers)
    for marker in markers:
        hue = (marker.marker_index - 1) / max(1, max_idx)
        rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=16)
        sphere.paint_uniform_color(rgb)
        sphere.translate(marker.position)
        sphere.compute_vertex_normals()
        meshes.append(sphere)

    return meshes


def build_lines(markers: list[MarkerConstraint], o3d: Any) -> Any | None:
    if len(markers) < 2:
        return None

    ordered = sorted(markers, key=lambda m: m.marker_index)
    points = [m.position for m in ordered]
    lines = [[i, i + 1] for i in range(len(ordered) - 1)]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_colors = [[0.6, 0.6, 0.6] for _ in lines]
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    return line_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize marker constraints from a .tak file with Open3D")
    parser.add_argument("tak_file", type=Path, help="Path to .tak file")
    parser.add_argument(
        "--group",
        default="auto",
        help="Marker group to display: 'auto' (default) or integer index from --list-groups",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="Print parsed marker groups and exit",
    )
    parser.add_argument(
        "--sphere-radius",
        type=float,
        default=0.01,
        help="Sphere radius in scene units (default: 0.01)",
    )
    parser.add_argument(
        "--no-spheres",
        action="store_true",
        help="Render only point cloud (no marker spheres)",
    )
    parser.add_argument(
        "--connect-order",
        action="store_true",
        help="Draw lines connecting markers in Marker1..MarkerN order",
    )
    return parser.parse_args()


def require_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:
        if sys.version_info >= (3, 13):
            raise SystemExit(
                "open3d does not currently provide wheels for Python 3.13 in this setup.\n"
                "Use a Python 3.12 environment for visualization, e.g.:\n"
                "  conda create -n patchcap-o3d python=3.12 -y\n"
                "  conda activate patchcap-o3d\n"
                "  pip install open3d\n"
                "  python scripts/visualize_tak_markers_open3d.py <your.tak> --group auto"
            ) from exc
        raise SystemExit(
            "open3d is required for visualization. Install it first, "
            "e.g. `uv add open3d` or `pip install open3d`."
        ) from exc
    return o3d


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

    nodes_blob = cfb.read_stream("Nodes.dat")
    nodes_text = "\n".join(extract_utf16_chunks(nodes_blob, min_chars=3, max_chunks=100_000))

    constraints = extract_marker_constraints(nodes_text)
    if not constraints:
        print("Error: no marker constraints parsed from Nodes.dat")
        return 1

    groups = split_constraint_groups(constraints)

    if args.list_groups:
        print(f"Found {len(groups)} marker groups:")
        for i, group in enumerate(groups):
            names = [m.marker_name for m in group]
            unique = sorted(set(names), key=lambda n: int(n.replace("Marker", "")))
            print(f"  [{i}] constraints={len(group)} unique_markers={len(unique)} names={', '.join(unique)}")
        return 0

    try:
        group_idx, markers = choose_group(groups, args.group)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Using group {group_idx}: {len(markers)} constraints")
    for marker in markers:
        x, y, z = marker.position
        print(f"  {marker.marker_name:>7}  ({x:+.6f}, {y:+.6f}, {z:+.6f})")

    o3d = require_open3d()

    geometries: list[Any] = []
    pcd = build_point_cloud(markers, o3d)
    geometries.append(pcd)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(axis)

    if not args.no_spheres:
        geometries.extend(build_spheres(markers, radius=args.sphere_radius, o3d=o3d))

    if args.connect_order:
        line_set = build_lines(markers, o3d)
        if line_set is not None:
            geometries.append(line_set)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"TAK Markers: {tak_path.name} [group {group_idx}]",
        width=1280,
        height=820,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
