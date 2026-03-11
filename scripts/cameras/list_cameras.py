"""
Camera Detection Script

This script scans for connected camera devices using OpenCV and lists the indices 
of all available cameras up to a specified maximum number.

Functions:
- list_connected_cameras(max_cameras=30): Returns a list of indices for all available 
  video capture devices. It checks camera indices from 0 up to max_cameras - 1.
- main(): Entry point of the script. Calls the camera detection function and 
  prints out the results.

Usage:
Run this script directly to see a list of connected cameras.

Example:
$ python camera_detection.py
Checking for connected cameras...
Found 2 camera(s): [0, 1]
"""

import cv2

def list_connected_cameras(max_cameras=30):
    available_cameras = []
    for camera_index in range(max_cameras):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            available_cameras.append(camera_index)
            cap.release()  # Release the camera after checking
    return available_cameras

def main():
    print("Checking for connected cameras...")
    cameras = list_connected_cameras()
    if cameras:
        print(f"Found {len(cameras)} camera(s): {cameras}")
    else:
        print("No cameras found.")

if __name__ == "__main__":
    main()