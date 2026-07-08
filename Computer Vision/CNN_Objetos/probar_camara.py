import os
import sys

import cv2


def main():
    device = os.environ.get("CAMERA_DEVICE", "/dev/video0")
    width = int(os.environ.get("CAMERA_WIDTH", "640"))
    height = int(os.environ.get("CAMERA_HEIGHT", "480"))

    print(f"Probando {device} con OpenCV {cv2.__version__}")
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not capture.isOpened():
        print("No se pudo abrir la camara.")
        return 1

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ok, frame = capture.read()
    capture.release()

    if not ok or frame is None:
        print("La camara abrio, pero no entrego frames.")
        return 1

    print(f"Frame leido correctamente: {frame.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
