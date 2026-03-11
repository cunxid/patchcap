import argparse
import platform

import cv2

def preferred_backend() -> int | None:
    if platform.system() == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return None


def open_capture(index: int, backend: int | None) -> cv2.VideoCapture:
    if backend is None:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, backend)


def list_connected_cameras(max_cameras: int, backend: int | None) -> list[int]:
    available = []
    for camera_index in range(max_cameras):
        cap = open_capture(camera_index, backend)
        if cap.isOpened():
            available.append(camera_index)
            cap.release()
    return available


def get_serial_number(index: int) -> str | None:
    if platform.system() != "Linux":
        return None

    try:
        import pyudev

        context = pyudev.Context()
        device = pyudev.Devices.from_device_file(context, f"/dev/video{index}")
    except Exception:
        return None

    info = {item[0]: item[1] for item in device.items()}
    return info.get("ID_SERIAL_SHORT")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test ArduCAMs")
    parser.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=None,
        help="Camera indices to test. If omitted, auto-detect available cameras.",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=8,
        help="Upper bound for auto-detect scan (exclusive).",
    )
    parser.add_argument("--fps", type=int, default=60, help="Requested FPS.")
    parser.add_argument("--width", type=int, default=640, help="Requested width.")
    parser.add_argument("--height", type=int, default=320, help="Requested height.")
    args = parser.parse_args()

    backend = preferred_backend()
    selected = args.cameras or list_connected_cameras(args.max_cameras, backend)

    if not selected:
        print("No cameras detected.")
        return 1

    print(f"Testing camera indices: {selected}")
    cameras = {}

    for index in selected:
        cap = open_capture(index, backend)
        if not cap.isOpened():
            print(f"Warning: Could not open camera {index}")
            continue

        serial = get_serial_number(index)
        if serial:
            print(f"Camera {index} serial number: {serial}")

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        original_fourcc_str = "".join(chr((original_fourcc >> 8 * i) & 0xFF) for i in range(4))

        print(
            f"Camera {index} native: {original_width}x{original_height} "
            f"@ {original_fps} FPS FourCC={original_fourcc_str}"
        )

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc_str = "".join(chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4))

        print(
            f"Camera {index} initialized: {actual_width}x{actual_height} "
            f"@ {actual_fps} FPS FourCC={actual_fourcc_str}"
        )
        cameras[index] = cap

    if not cameras:
        print("No requested cameras could be opened.")
        return 1

    print("Cameras started. Press 'q' to quit.")

    try:
        while True:
            for index, cap in cameras.items():
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to grab frame from camera {index}")
                    continue
                cv2.imshow(f"ArduCAM Feed {index}", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Exiting due to interrupt")
    finally:
        for index, cap in cameras.items():
            cap.release()
            cv2.destroyWindow(f"ArduCAM Feed {index}")
        cv2.destroyAllWindows()
        print("Cameras released and windows closed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
