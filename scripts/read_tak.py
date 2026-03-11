#!/usr/bin/env python3
"""Summarize OptiTrack-style .tak motion-capture files.

This script reads .tak files stored as Compound File Binary (OLE/CFB) containers
and prints a compact summary of stream contents and key metadata.
"""

from __future__ import annotations

import argparse
import math
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
    """Minimal Compound File Binary (CFB/OLE) reader for stream enumeration."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = path.read_bytes()
        self._validate_header()

        self.minor_version = struct.unpack_from("<H", self.data, 0x18)[0]
        self.major_version = struct.unpack_from("<H", self.data, 0x1A)[0]
        self.byte_order = struct.unpack_from("<H", self.data, 0x1C)[0]
        self.sector_shift = struct.unpack_from("<H", self.data, 0x1E)[0]
        self.mini_sector_shift = struct.unpack_from("<H", self.data, 0x20)[0]
        self.sector_size = 1 << self.sector_shift
        self.mini_sector_size = 1 << self.mini_sector_shift

        self.num_fat_sectors = struct.unpack_from("<I", self.data, 0x2C)[0]
        self.first_dir_sector = struct.unpack_from("<I", self.data, 0x30)[0]
        self.mini_stream_cutoff = struct.unpack_from("<I", self.data, 0x38)[0]
        self.first_mini_fat_sector = struct.unpack_from("<I", self.data, 0x3C)[0]
        self.num_mini_fat_sectors = struct.unpack_from("<I", self.data, 0x40)[0]
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
            if current >= len(table):
                break
            if current in seen:
                break
            if len(chain) >= limit:
                break

            seen.add(current)
            chain.append(current)
            current = table[current]

        return chain

    def _sector_offset(self, sector_id: int) -> int:
        return (sector_id + 1) * self.sector_size

    def stream_entries(self) -> list[DirectoryEntry]:
        return [entry for entry in self.directory_entries if entry.object_type == 2]

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


def human_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def extract_utf16_chunks(blob: bytes, min_chars: int = 4, max_chunks: int = 5000) -> list[str]:
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


def as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def as_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def summarize_tak(path: Path, channels_scan_bytes: int) -> str:
    cfb = CompoundFile(path)
    lines: list[str] = []

    lines.append(f"File: {path}")
    lines.append(f"Size: {human_size(path.stat().st_size)}")
    lines.append(
        "Container: Compound File Binary "
        f"(major={cfb.major_version}, minor={cfb.minor_version}, "
        f"sector={cfb.sector_size}, mini_sector={cfb.mini_sector_size})"
    )

    streams = sorted(cfb.stream_entries(), key=lambda e: e.size, reverse=True)
    lines.append("")
    lines.append("Streams:")
    for entry in streams:
        lines.append(f"  - {entry.name}: {human_size(entry.size)}")

    meta = cfb.stream_entry("MetaData.dat")
    metadata_props: dict[str, str] = {}
    if meta is not None:
        meta_blob = cfb.read_stream("MetaData.dat")
        meta_text = "\n".join(extract_utf16_chunks(meta_blob, min_chars=3))
        metadata_props = parse_property_pairs(meta_text)

    lines.append("")
    lines.append("Take Summary:")

    if metadata_props:
        take_name = metadata_props.get("TakeName")
        frame_rate = as_float(metadata_props.get("FrameRate"))
        start_frame = as_int(metadata_props.get("StartFrame"))
        end_frame = as_int(metadata_props.get("EndFrame"))
        start_time = as_float(metadata_props.get("StartTime"))
        end_time = as_float(metadata_props.get("EndTime"))

        if take_name:
            lines.append(f"  - Name: {take_name}")
        if frame_rate is not None:
            lines.append(f"  - Frame rate: {frame_rate:.3f} Hz")
        if start_frame is not None and end_frame is not None:
            frame_count = max(0, end_frame - start_frame + 1)
            lines.append(f"  - Frame range: {start_frame}..{end_frame} ({frame_count} frames)")
        else:
            frame_count = None

        if start_time is not None and end_time is not None:
            lines.append(f"  - Time range: {start_time:.6f}s..{end_time:.6f}s ({end_time - start_time:.6f}s)")

        capture_start = parse_epoch_ms(metadata_props.get("CaptureStart"))
        if capture_start:
            lines.append(f"  - Capture start: {capture_start}")

        calib_time = parse_epoch_ms(metadata_props.get("CalibrationTime"))
        if calib_time:
            lines.append(f"  - Calibration time: {calib_time}")

        if frame_rate and frame_count:
            derived_duration = frame_count / frame_rate
            if start_time is not None and end_time is not None:
                declared_duration = end_time - start_time
                delta = abs(declared_duration - derived_duration)
                lines.append(
                    "  - Duration check: "
                    f"declared {declared_duration:.6f}s vs frames/fps {derived_duration:.6f}s "
                    f"(delta {delta:.6f}s)"
                )
            else:
                lines.append(f"  - Derived duration: {derived_duration:.6f}s")

        assets = metadata_props.get("Assets")
        if assets:
            lines.append(f"  - Assets: {assets}")
    else:
        lines.append("  - Could not parse MetaData.dat properties")

    nodes = cfb.stream_entry("Nodes.dat")
    if nodes is not None:
        nodes_blob = cfb.read_stream("Nodes.dat")
        nodes_text = "\n".join(extract_utf16_chunks(nodes_blob, min_chars=3))

        marker_names = sorted(set(re.findall(r"\bMarker\d+\b", nodes_text)))
        camera_labels = sorted(set(re.findall(r"PrimeX[^\"\r\n<]*#\d+", nodes_text)))
        if not camera_labels:
            camera_labels = sorted(set(re.findall(r"PrimeX[^\"\r\n<]*", nodes_text)))

        lines.append("")
        lines.append("Node Summary:")
        lines.append(f"  - Markers detected: {len(marker_names)}")
        if marker_names:
            preview = ", ".join(marker_names[:12])
            suffix = " ..." if len(marker_names) > 12 else ""
            lines.append(f"  - Marker names: {preview}{suffix}")

        if camera_labels:
            lines.append(f"  - Cameras detected: {len(camera_labels)}")
            for label in camera_labels[:8]:
                lines.append(f"    - {label}")
        else:
            lines.append("  - Cameras detected: 0 (or not found in Nodes.dat text)")

    channels = cfb.stream_entry("Channels.dat")
    if channels is not None:
        channel_blob = cfb.read_stream("Channels.dat", max_bytes=channels_scan_bytes)
        channel_text = "\n".join(extract_utf16_chunks(channel_blob, min_chars=3))

        channel_tokens = sorted(
            set(
                re.findall(
                    r"\b(?:[A-Za-z][A-Za-z0-9_]*Channel|Rotation|Translation|MarkerError)\b",
                    channel_text,
                )
            )
        )

        lines.append("")
        lines.append("Channel Summary:")
        lines.append(f"  - Scan bytes: {human_size(min(channels.size, channels_scan_bytes))}")
        if channel_tokens:
            lines.append(f"  - Channel/value types: {', '.join(channel_tokens)}")
        else:
            lines.append("  - No recognizable channel labels found in scan")

    track_entries = [entry for entry in streams if entry.name.lower().startswith("track ")]
    if track_entries:
        lines.append("")
        lines.append("Track Streams:")
        for entry in track_entries:
            lines.append(f"  - {entry.name}: {human_size(entry.size)}")

        end_frame = as_int(metadata_props.get("EndFrame"))
        start_frame = as_int(metadata_props.get("StartFrame"))
        if end_frame is not None and start_frame is not None:
            total_frames = max(0, end_frame - start_frame + 1)
            if total_frames > 0:
                largest = max(track_entries, key=lambda e: e.size)
                bytes_per_frame = largest.size / total_frames
                if math.isfinite(bytes_per_frame):
                    lines.append(
                        f"  - Approx bytes/frame in {largest.name}: {bytes_per_frame:.2f}"
                    )

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read a .tak motion capture file and print summarized information."
    )
    parser.add_argument("tak_file", type=Path, help="Path to .tak file")
    parser.add_argument(
        "--channels-scan-mb",
        type=int,
        default=4,
        help="How many MB from Channels.dat to scan for text labels (default: 4)",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    tak_path = args.tak_file.expanduser().resolve()
    if not tak_path.exists():
        print(f"Error: file not found: {tak_path}")
        return 1
    if tak_path.suffix.lower() != ".tak":
        print(f"Warning: expected .tak file, got: {tak_path.name}")

    try:
        summary = summarize_tak(tak_path, channels_scan_bytes=max(1, args.channels_scan_mb) * 1024 * 1024)
    except Exception as exc:  # noqa: BLE001 - keep CLI failure friendly
        print(f"Error reading {tak_path}: {exc}")
        return 1

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
