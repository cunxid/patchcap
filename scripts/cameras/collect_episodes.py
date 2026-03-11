import argparse
import os
import platform
import select
import sys
from pathlib import Path

import cv2


def preferred_backend() -> int | None:
    if platform.system() == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return None


def open_capture(index: int, backend: int | None) -> cv2.VideoCapture:
    if backend is None:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, backend)


def configure_capture(cap: cv2.VideoCapture, width: int, height: int, fps: int) -> None:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)


def next_episode_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in output_root.glob("episode_*"):
        if not path.is_dir():
            continue
        try:
            existing.append(int(path.name.split("_")[-1]))
        except ValueError:
            continue

    episode_id = max(existing, default=0) + 1
    episode_dir = output_root / f"episode_{episode_id:04d}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    return episode_dir


def enable_terminal_single_key_mode() -> tuple[int, list[object]] | None:
    if os.name != "posix" or not sys.stdin.isatty():
        return None

    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        previous_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        return fd, previous_settings
    except Exception:
        return None


def poll_terminal_key(fd: int) -> str | None:
    try:
        readable, _, _ = select.select([fd], [], [], 0)
    except Exception:
        return None

    if not readable:
        return None

    try:
        raw = os.read(fd, 1)
    except Exception:
        return None

    if not raw:
        return None

    return raw.decode(errors="ignore").lower()


def restore_terminal_mode(state: tuple[int, list[object]] | None) -> None:
    if state is None:
        return

    try:
        import termios

        fd, previous_settings = state
        termios.tcsetattr(fd, termios.TCSADRAIN, previous_settings)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Record multi-camera episodes.")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1], help="Camera indices to record.")
    parser.add_argument("--fps", type=int, default=30, help="Recording FPS.")
    parser.add_argument("--width", type=int, default=640, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/episodes"),
        help="Root folder where episode directories are created.",
    )
    args = parser.parse_args()

    backend = preferred_backend()
    captures: dict[int, cv2.VideoCapture] = {}
    latest_frames: dict[int, object] = {}

    for camera_index in args.cameras:
        cap = open_capture(camera_index, backend)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            for opened_cap in captures.values():
                opened_cap.release()
            return 1

        configure_capture(cap, args.width, args.height, args.fps)
        captures[camera_index] = cap
        latest_frames[camera_index] = None
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Camera {camera_index} ready: {actual_w}x{actual_h} @ {actual_fps} FPS")

    recording = False
    episode_dir: Path | None = None
    writers: dict[int, cv2.VideoWriter] = {}
    terminal_state = enable_terminal_single_key_mode()

    print("Controls: press 'r' to start episode, 'q' to stop episode, 'x' or ESC to exit.")
    if terminal_state is not None:
        print("Terminal controls enabled (single key, no Enter required).")
    else:
        print("Terminal controls unavailable. Use keys in an OpenCV window.")

    try:
        while True:
            for camera_index, cap in captures.items():
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: failed to read frame from camera {camera_index}")
                    continue

                latest_frames[camera_index] = frame
                cv2.imshow(f"Camera {camera_index}", frame)

                if recording and camera_index in writers:
                    writers[camera_index].write(frame)

            window_key = cv2.waitKey(1) & 0xFF
            terminal_key = None
            if terminal_state is not None:
                terminal_key = poll_terminal_key(terminal_state[0])

            if terminal_key == "r" or window_key == ord("r"):
                if recording:
                    print("Already recording this episode.")
                    continue

                missing = [idx for idx, frame in latest_frames.items() if frame is None]
                if missing:
                    print(f"Cannot start recording yet. No frame received from cameras: {missing}")
                    continue

                episode_dir = next_episode_dir(args.output_dir)
                new_writers: dict[int, cv2.VideoWriter] = {}
                open_failed = False

                for camera_index, frame in latest_frames.items():
                    frame_height, frame_width = frame.shape[:2]
                    video_path = episode_dir / f"camera_{camera_index}.mp4"
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.fps,
                        (frame_width, frame_height),
                    )
                    if not writer.isOpened():
                        print(f"Error: failed to create video writer for camera {camera_index}")
                        open_failed = True
                        break
                    new_writers[camera_index] = writer

                if open_failed:
                    for writer in new_writers.values():
                        writer.release()
                    episode_dir = None
                    continue

                writers = new_writers
                recording = True
                print(f"Recording episode in: {episode_dir}")

            elif terminal_key == "q" or window_key == ord("q"):
                if not recording:
                    print("Not recording. Press 'r' to start an episode.")
                    continue

                for writer in writers.values():
                    writer.release()
                writers.clear()
                recording = False
                print(f"Episode saved: {episode_dir}")
                episode_dir = None

            elif terminal_key == "x" or window_key in (27, ord("x")):
                break

    finally:
        restore_terminal_mode(terminal_state)
        for writer in writers.values():
            writer.release()
        for cap in captures.values():
            cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
