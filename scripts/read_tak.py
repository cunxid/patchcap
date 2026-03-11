#!/usr/bin/env python3
"""Extract metadata from OptiTrack-style .tak motion-capture files.

This script reads .tak files stored as Compound File Binary (OLE/CFB) containers,
parses metadata properties from MetaData.dat, prints metadata-only output, and
writes the same report to a text file for downstream analysis.
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
FREE_SECTOR = 0xFFFFFFFF
END_OF_CHAIN = 0xFFFFFFFE


@dataclass(frozen=True)
class DirectoryEntry:
    index: int
    name: str
    object_type: int
    start_sector: int
    size: int


class CompoundFile:
    """Minimal Compound File Binary (CFB/OLE) reader for stream extraction."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = path.read_bytes()
        self._validate_header()

        self.minor_version = struct.unpack_from("<H", self.data, 0x18)[0]
        self.major_version = struct.unpack_from("<H", self.data, 0x1A)[0]
        self.sector_shift = struct.unpack_from("<H", self.data, 0x1E)[0]
        self.mini_sector_shift = struct.unpack_from("<H", self.data, 0x20)[0]
        self.sector_size = 1 << self.sector_shift
        self.mini_sector_size = 1 << self.mini_sector_shift

        self.first_dir_sector = struct.unpack_from("<I", self.data, 0x30)[0]
        self.mini_stream_cutoff = struct.unpack_from("<I", self.data, 0x38)[0]
        self.first_mini_fat_sector = struct.unpack_from("<I", self.data, 0x3C)[0]
        self.first_difat_sector = struct.unpack_from("<I", self.data, 0x44)[0]
        self.num_difat_sectors = struct.unpack_from("<I", self.data, 0x48)[0]

        self.fat_sector_ids = self._parse_difat()
        self.fat = self._parse_fat(self.fat_sector_ids)
        self.directory_entries = self._parse_directory_entries()

        self._stream_by_name = {
            entry.name: entry
            for entry in self.directory_entries
            if entry.object_type == 2 and entry.name
        }

        self._root_entry = self.directory_entries[0]
        self._mini_fat = self._load_mini_fat()
        self._mini_stream = self._load_root_mini_stream()

    def _validate_header(self) -> None:
        if len(self.data) < 512:
            raise ValueError("File is too small to be a valid compound file")
        if self.data[:8] != MAGIC:
            raise ValueError("Not a Compound File Binary (OLE/CFB) file")

    def _parse_difat(self) -> list[int]:
        difat: list[int] = []

        for i in range(109):
            sector_id = struct.unpack_from("<I", self.data, 0x4C + i * 4)[0]
            if sector_id != FREE_SECTOR:
                difat.append(sector_id)

        next_sector = self.first_difat_sector
        for _ in range(self.num_difat_sectors):
            if next_sector in (FREE_SECTOR, END_OF_CHAIN):
                break
            offset = self._sector_offset(next_sector)
            entries_per_sector = self.sector_size // 4

            for i in range(entries_per_sector - 1):
                sector_id = struct.unpack_from("<I", self.data, offset + i * 4)[0]
                if sector_id != FREE_SECTOR:
                    difat.append(sector_id)

            next_sector = struct.unpack_from("<I", self.data, offset + self.sector_size - 4)[0]

        return difat

    def _parse_fat(self, fat_sector_ids: Iterable[int]) -> list[int]:
        fat: list[int] = []
        entries_per_sector = self.sector_size // 4

        for sector_id in fat_sector_ids:
            offset = self._sector_offset(sector_id)
            fat.extend(struct.unpack_from(f"<{entries_per_sector}I", self.data, offset))

        return fat

    def _parse_directory_entries(self) -> list[DirectoryEntry]:
        directory_bytes = bytearray()
        for sector_id in self._sector_chain(self.first_dir_sector, self.fat):
            offset = self._sector_offset(sector_id)
            directory_bytes.extend(self.data[offset : offset + self.sector_size])

        entries: list[DirectoryEntry] = []
        for i in range(0, len(directory_bytes), 128):
            raw = directory_bytes[i : i + 128]
            if len(raw) < 128:
                break

            name_length = struct.unpack_from("<H", raw, 64)[0]
            if name_length >= 2:
                name = raw[: name_length - 2].decode("utf-16le", errors="ignore")
            else:
                name = ""

            object_type = raw[66]
            start_sector = struct.unpack_from("<I", raw, 116)[0]
            size = struct.unpack_from("<Q", raw, 120)[0]
            entries.append(
                DirectoryEntry(
                    index=i // 128,
                    name=name,
                    object_type=object_type,
                    start_sector=start_sector,
                    size=size,
                )
            )

        return entries

    def _load_mini_fat(self) -> list[int]:
        mini_fat: list[int] = []
        entries_per_sector = self.sector_size // 4

        for sector_id in self._sector_chain(self.first_mini_fat_sector, self.fat):
            offset = self._sector_offset(sector_id)
            mini_fat.extend(struct.unpack_from(f"<{entries_per_sector}I", self.data, offset))

        return mini_fat

    def _load_root_mini_stream(self) -> bytes:
        if self._root_entry.start_sector in (FREE_SECTOR, END_OF_CHAIN) or self._root_entry.size == 0:
            return b""

        out = bytearray()
        for sector_id in self._sector_chain(self._root_entry.start_sector, self.fat):
            offset = self._sector_offset(sector_id)
            out.extend(self.data[offset : offset + self.sector_size])
            if len(out) >= self._root_entry.size:
                break

        return bytes(out[: self._root_entry.size])

    def _sector_chain(self, start_sector: int, table: list[int], limit: int = 2_000_000) -> list[int]:
        if start_sector in (FREE_SECTOR, END_OF_CHAIN):
            return []

        chain: list[int] = []
        seen: set[int] = set()
        current = start_sector

        while current not in (FREE_SECTOR, END_OF_CHAIN):
            if current >= len(table) or current in seen or len(chain) >= limit:
                break
            seen.add(current)
            chain.append(current)
            current = table[current]

        return chain

    def _sector_offset(self, sector_id: int) -> int:
        return (sector_id + 1) * self.sector_size

    def stream_entry(self, name: str) -> DirectoryEntry | None:
        return self._stream_by_name.get(name)

    def read_stream(self, name: str, max_bytes: int | None = None) -> bytes:
        entry = self.stream_entry(name)
        if entry is None:
            raise KeyError(f"Stream not found: {name}")
        if entry.size == 0:
            return b""

        target = entry.size if max_bytes is None else min(entry.size, max_bytes)

        if entry.size < self.mini_stream_cutoff and self._mini_stream:
            return self._read_mini_stream(entry.start_sector, target)
        return self._read_normal_stream(entry.start_sector, target)

    def _read_normal_stream(self, start_sector: int, target_bytes: int) -> bytes:
        out = bytearray()
        for sector_id in self._sector_chain(start_sector, self.fat):
            offset = self._sector_offset(sector_id)
            out.extend(self.data[offset : offset + self.sector_size])
            if len(out) >= target_bytes:
                break
        return bytes(out[:target_bytes])

    def _read_mini_stream(self, start_sector: int, target_bytes: int) -> bytes:
        out = bytearray()
        for mini_sector_id in self._sector_chain(start_sector, self._mini_fat):
            offset = mini_sector_id * self.mini_sector_size
            out.extend(self._mini_stream[offset : offset + self.mini_sector_size])
            if len(out) >= target_bytes:
                break
        return bytes(out[:target_bytes])


def extract_utf16_chunks(blob: bytes, min_chars: int = 3, max_chunks: int = 10000) -> list[str]:
    pattern = re.compile(rb"(?:[\x20-\x7e]\x00){" + str(min_chars).encode("ascii") + rb",}")
    chunks: list[str] = []

    for match in pattern.finditer(blob):
        try:
            text = match.group(0).decode("utf-16le")
        except UnicodeDecodeError:
            continue
        chunks.append(text)
        if len(chunks) >= max_chunks:
            break

    return chunks


def parse_property_pairs(text: str) -> dict[str, str]:
    pairs = re.findall(r'<property\s+name="([^"]+)"\s+value="([^"]*)"', text)
    return {name: value for name, value in pairs}


def parse_epoch_ms(value: str | None) -> str | None:
    if not value:
        return None
    try:
        ms = int(value)
    except ValueError:
        return None

    dt_utc = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    dt_local = dt_utc.astimezone()
    return f"{dt_local.isoformat()} (local), {dt_utc.isoformat()} (UTC)"


def parse_assets_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def infer_frame_count(metadata_props: dict[str, str]) -> int | None:
    frame_key_pairs = [
        ("WorkingStartFrame", "WorkingEndFrame"),
        ("StartFrame", "EndFrame"),
    ]

    for start_key, end_key in frame_key_pairs:
        start_raw = metadata_props.get(start_key)
        end_raw = metadata_props.get(end_key)
        if start_raw is None or end_raw is None:
            continue
        try:
            start = int(start_raw)
            end = int(end_raw)
        except ValueError:
            continue
        if end >= start:
            return end - start + 1
    return None


def infer_rigidbody_names(assets: list[str], inferred_columns: list[str]) -> list[str]:
    rigid_names: list[str] = []
    for name in assets:
        required = {f"{name}.tx", f"{name}.ty", f"{name}.tz", f"{name}.qx", f"{name}.qy", f"{name}.qz", f"{name}.qw"}
        if required.issubset(set(inferred_columns)):
            rigid_names.append(name)

    if rigid_names:
        return rigid_names

    by_prefix: dict[str, set[str]] = {}
    for column in inferred_columns:
        if "." not in column:
            continue
        prefix, suffix = column.rsplit(".", 1)
        by_prefix.setdefault(prefix, set()).add(suffix)

    required_suffixes = {"tx", "ty", "tz", "qx", "qy", "qz", "qw"}
    for prefix in sorted(by_prefix):
        if required_suffixes.issubset(by_prefix[prefix]):
            rigid_names.append(prefix)
    return rigid_names


def write_motion_csv_template(
    output_path: Path,
    rigid_body_names: list[str],
    rows: int,
) -> None:
    headers: list[str] = []
    for name in rigid_body_names:
        headers.extend([f"{name}.tx", f"{name}.ty", f"{name}.tz", f"{name}.qx", f"{name}.qy", f"{name}.qz", f"{name}.qw"])

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for _ in range(max(1, rows)):
            row: list[float] = []
            for _name in rigid_body_names:
                row.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            writer.writerow(row)


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def extract_marker_names(cfb: CompoundFile) -> list[str]:
    nodes = cfb.stream_entry("Nodes.dat")
    if nodes is None:
        return []

    nodes_blob = cfb.read_stream("Nodes.dat")
    nodes_text = "\n".join(extract_utf16_chunks(nodes_blob, min_chars=3, max_chunks=200_000))
    names = re.findall(r'"name":\s*"NodeName"\s*,\s*"value":\s*"(Marker\d+)"', nodes_text)

    try:
        return sorted(set(names), key=lambda name: int(name.replace("Marker", "")))
    except ValueError:
        return sorted(set(names))


def infer_data_columns(
    cfb: CompoundFile,
    assets: list[str],
    marker_names: list[str],
) -> dict[str, object]:
    channels = cfb.stream_entry("Channels.dat")
    if channels is None:
        return {
            "available": False,
            "reason": "Channels.dat not found",
            "columns": [],
            "scan_bytes": 0,
            "rigid_bodies": 0,
            "markers": 0,
        }

    blob = cfb.read_stream("Channels.dat")
    channel_text = extract_utf16_chunks(blob, min_chars=3, max_chunks=300_000)
    semantics = [token for token in channel_text if token in {"Translation", "Rotation", "MarkerError"}]
    if not semantics:
        return {
            "available": False,
            "reason": "No channel semantic labels found in scanned Channels.dat",
            "columns": [],
            "scan_bytes": len(blob),
            "rigid_bodies": 0,
            "markers": 0,
        }

    columns: list[str] = []
    rigid_body_index = 0
    marker_index = 0

    i = 0
    while i < len(semantics):
        # Most OptiTrack TAKs encode rigid body channels as:
        # Translation (xyz), Rotation (quat), MarkerError (float).
        if i + 2 < len(semantics) and semantics[i : i + 3] == ["Translation", "Rotation", "MarkerError"]:
            name = assets[rigid_body_index] if rigid_body_index < len(assets) else f"RigidBody{rigid_body_index + 1}"
            columns.extend(
                [
                    f"{name}.tx",
                    f"{name}.ty",
                    f"{name}.tz",
                    f"{name}.qx",
                    f"{name}.qy",
                    f"{name}.qz",
                    f"{name}.qw",
                    f"{name}.marker_error",
                ]
            )
            rigid_body_index += 1
            i += 3
            continue

        token = semantics[i]
        if token == "Translation":
            marker_name = (
                marker_names[marker_index] if marker_index < len(marker_names) else f"Marker{marker_index + 1:03d}"
            )
            columns.extend([f"{marker_name}.x", f"{marker_name}.y", f"{marker_name}.z"])
            marker_index += 1
        elif token == "Rotation":
            name = assets[rigid_body_index] if rigid_body_index < len(assets) else f"RigidBody{rigid_body_index + 1}"
            columns.extend([f"{name}.qx", f"{name}.qy", f"{name}.qz", f"{name}.qw"])
        elif token == "MarkerError":
            if assets and rigid_body_index > 0:
                name = assets[min(rigid_body_index - 1, len(assets) - 1)]
            else:
                name = f"RigidBody{max(rigid_body_index, 1)}"
            columns.append(f"{name}.marker_error")
        i += 1

    return {
        "available": True,
        "reason": "",
        "columns": columns,
        "scan_bytes": len(blob),
        "rigid_bodies": rigid_body_index,
        "markers": marker_index,
    }


def detect_6dof(cfb: CompoundFile, assets: list[str], channels_scan_bytes: int) -> dict[str, object]:
    channels = cfb.stream_entry("Channels.dat")
    if channels is None:
        return {
            "available": False,
            "reason": "Channels.dat not found",
            "per_asset": {},
            "global_translation": False,
            "global_rotation": False,
            "scan_bytes": 0,
        }

    blob = cfb.read_stream("Channels.dat", max_bytes=channels_scan_bytes)
    channel_text = "\n".join(extract_utf16_chunks(blob, min_chars=3))
    lower = channel_text.lower()

    # Keep broad token matching to support different exporter label variants.
    global_translation = bool(
        re.search(r"\btranslation\b|\bposition\b|\bpos[xyz]\b|\btx\b|\bty\b|\btz\b", lower)
    )
    global_rotation = bool(
        re.search(r"\brotation\b|\borientation\b|\bquat(?:ernion)?\b|\bqw\b|\bqx\b|\bqy\b|\bqz\b", lower)
    )

    per_asset: dict[str, dict[str, bool]] = {}
    for asset in assets:
        n_asset = normalize_text(asset)
        has_match = False
        has_translation = False
        has_rotation = False
        for chunk in channel_text.splitlines():
            n_chunk = normalize_text(chunk)
            if n_asset and n_asset in n_chunk:
                has_match = True
                chunk_lower = chunk.lower()
                if re.search(r"\btranslation\b|\bposition\b|\bpos[xyz]\b|\btx\b|\bty\b|\btz\b", chunk_lower):
                    has_translation = True
                if re.search(
                    r"\brotation\b|\borientation\b|\bquat(?:ernion)?\b|\bqw\b|\bqx\b|\bqy\b|\bqz\b",
                    chunk_lower,
                ):
                    has_rotation = True
                if has_translation and has_rotation:
                    break
        per_asset[asset] = {
            "present_in_channels": has_match,
            "translation": has_translation,
            "rotation": has_rotation,
            "has_6dof": has_translation and has_rotation,
        }

    return {
        "available": True,
        "reason": "",
        "per_asset": per_asset,
        "global_translation": global_translation,
        "global_rotation": global_rotation,
        "scan_bytes": len(blob),
    }


def summarize_metadata(path: Path, channels_scan_bytes: int) -> tuple[str, dict[str, object]]:
    cfb = CompoundFile(path)
    meta = cfb.stream_entry("MetaData.dat")
    if meta is None:
        return "MetaData.dat was not found in this .tak file.", {"rigid_body_names": [], "frame_count": None}

    meta_blob = cfb.read_stream("MetaData.dat")
    meta_text = "\n".join(extract_utf16_chunks(meta_blob, min_chars=3))
    metadata_props = parse_property_pairs(meta_text)
    frame_count = infer_frame_count(metadata_props)

    lines: list[str] = []
    lines.append(f"File: {path}")
    lines.append("MetaData only")
    lines.append("")

    if not metadata_props:
        lines.append("No parseable <property name=... value=...> pairs found in MetaData.dat")
        return "\n".join(lines), {"rigid_body_names": [], "frame_count": frame_count}

    lines.append("Available metadata fields:")
    for key in sorted(metadata_props):
        lines.append(f"  - {key}")

    lines.append("")
    lines.append("Metadata values:")
    for key in sorted(metadata_props):
        lines.append(f"  - {key}: {metadata_props[key]}")

    assets = parse_assets_list(metadata_props.get("Assets"))
    pose_info = detect_6dof(cfb, assets, channels_scan_bytes=channels_scan_bytes)
    marker_names = extract_marker_names(cfb)
    column_info = infer_data_columns(
        cfb,
        assets=assets,
        marker_names=marker_names,
    )

    lines.append("")
    lines.append("6DoF pose check:")
    if not pose_info["available"]:
        lines.append(f"  - Verdict: unknown ({pose_info['reason']})")
    else:
        global_translation = bool(pose_info["global_translation"])
        global_rotation = bool(pose_info["global_rotation"])
        if assets:
            per_asset = pose_info["per_asset"]
            assets_with_6dof = [name for name in assets if per_asset[name]["has_6dof"]]
            if assets_with_6dof:
                lines.append("  - Verdict: yes, 6DoF evidence found for one or more assets")
            elif global_translation and global_rotation:
                lines.append(
                    "  - Verdict: likely yes (rotation+translation labels found globally, "
                    "but not linked to specific asset names)"
                )
            else:
                lines.append("  - Verdict: no clear 6DoF evidence in scanned Channels.dat text")

            lines.append(
                f"  - Channels scan size: {pose_info['scan_bytes']} bytes "
                f"(requested {channels_scan_bytes} bytes)"
            )
            lines.append(f"  - Global translation labels present: {global_translation}")
            lines.append(f"  - Global rotation labels present: {global_rotation}")
            lines.append("  - Per-asset evidence:")
            for asset in assets:
                item = per_asset[asset]
                lines.append(
                    "    - "
                    f"{asset}: present={item['present_in_channels']}, "
                    f"translation={item['translation']}, rotation={item['rotation']}, "
                    f"has_6dof={item['has_6dof']}"
                )
        else:
            if global_translation and global_rotation:
                lines.append("  - Verdict: likely yes (global rotation+translation labels found)")
            else:
                lines.append("  - Verdict: unknown (no assets in metadata and weak channel labels)")
            lines.append(
                f"  - Channels scan size: {pose_info['scan_bytes']} bytes "
                f"(requested {channels_scan_bytes} bytes)"
            )
            lines.append(f"  - Global translation labels present: {global_translation}")
            lines.append(f"  - Global rotation labels present: {global_rotation}")

    lines.append("")
    lines.append("Inferred data columns:")
    if not column_info["available"]:
        lines.append(f"  - unavailable ({column_info['reason']})")
    else:
        lines.append(
            f"  - Channels bytes parsed for column inference: {column_info['scan_bytes']} bytes "
            "(full stream)"
        )
        lines.append(f"  - Inferred rigid-body channel groups: {column_info['rigid_bodies']}")
        lines.append(f"  - Inferred marker translation channels: {column_info['markers']}")
        if marker_names:
            lines.append(f"  - Marker names found in Nodes.dat: {len(marker_names)}")
        if frame_count is not None:
            lines.append(f"  - Values per scalar column: {frame_count}")
        lines.append(f"  - Total scalar columns: {len(column_info['columns'])}")
        for column in column_info["columns"]:
            if frame_count is None:
                lines.append(f"    - {column}")
            else:
                lines.append(f"    - {column} ({frame_count} values)")

    capture_start = parse_epoch_ms(metadata_props.get("CaptureStart"))
    calibration_time = parse_epoch_ms(metadata_props.get("CalibrationTime"))

    if capture_start or calibration_time:
        lines.append("")
        lines.append("Derived timestamps:")
        if capture_start:
            lines.append(f"  - CaptureStart: {capture_start}")
        if calibration_time:
            lines.append(f"  - CalibrationTime: {calibration_time}")

    rigid_body_names = infer_rigidbody_names(assets, column_info["columns"])
    return "\n".join(lines), {"rigid_body_names": rigid_body_names, "frame_count": frame_count}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read a .tak motion capture file, print metadata, and report 6DoF evidence."
    )
    parser.add_argument("tak_file", type=Path, help="Path to .tak file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .txt path (default: <tak_file_stem>_metadata.txt next to input)",
    )
    parser.add_argument(
        "--channels-scan-mb",
        type=int,
        default=8,
        help="How many MB from Channels.dat to scan for 6DoF labels (default: 8)",
    )
    parser.add_argument(
        "--motion-csv-output",
        type=Path,
        default=None,
        help="Path for generated rigid-body motion CSV template (default: <tak_file_stem>_rigid_body_motion.csv)",
    )
    parser.add_argument(
        "--motion-csv-rows",
        type=int,
        default=1,
        help="How many rows to write in generated motion CSV template (default: 1)",
    )
    parser.add_argument(
        "--no-motion-csv",
        action="store_true",
        help="Skip writing the rigid-body motion CSV template.",
    )
    return parser


def default_output_path(tak_path: Path) -> Path:
    return tak_path.with_name(f"{tak_path.stem}_metadata.txt")


def default_motion_csv_path(tak_path: Path) -> Path:
    return tak_path.with_name(f"{tak_path.stem}_rigid_body_motion.csv")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    tak_path = args.tak_file.expanduser().resolve()
    if not tak_path.exists():
        print(f"Error: file not found: {tak_path}")
        return 1
    if tak_path.suffix.lower() != ".tak":
        print(f"Warning: expected .tak file, got: {tak_path.name}")

    output_path = (
        args.output.expanduser().resolve() if args.output is not None else default_output_path(tak_path)
    )
    motion_csv_output = (
        args.motion_csv_output.expanduser().resolve()
        if args.motion_csv_output is not None
        else default_motion_csv_path(tak_path)
    )

    try:
        summary, export_info = summarize_metadata(
            tak_path,
            channels_scan_bytes=max(1, args.channels_scan_mb) * 1024 * 1024,
        )
        output_path.write_text(summary + "\n", encoding="utf-8")

        if not args.no_motion_csv:
            rigid_body_names = list(export_info["rigid_body_names"])
            if rigid_body_names:
                write_motion_csv_template(
                    output_path=motion_csv_output,
                    rigid_body_names=rigid_body_names,
                    rows=max(1, args.motion_csv_rows),
                )
            else:
                print("Warning: no rigid-body channels inferred; skipping motion CSV template generation.")
    except Exception as exc:  # noqa: BLE001 - keep CLI failure friendly
        print(f"Error reading {tak_path}: {exc}")
        return 1

    print(summary)
    print("")
    print(f"Saved metadata report to: {output_path}")
    if not args.no_motion_csv and motion_csv_output.exists():
        print(
            f"Saved rigid-body motion CSV template to: {motion_csv_output} "
            "(placeholder static pose; direct Track*.trk decoding not implemented)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
