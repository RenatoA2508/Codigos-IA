import os
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = Path("cifar10_model.h5")
DEFAULT_CAMERA_DEVICE = "/dev/video0"
DEFAULT_CAMERA_SIZE = (640, 480)

CLASS_NAMES = [
    "avion",
    "automovil",
    "ave",
    "gato",
    "ciervo",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camion",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No existe cifar10_model.h5. Primero ejecuta: python train_cifar10.py"
        )
    return tf.keras.models.load_model(MODEL_PATH)


def parse_camera_size():
    width = int(os.environ.get("CAMERA_WIDTH", str(DEFAULT_CAMERA_SIZE[0])))
    height = int(os.environ.get("CAMERA_HEIGHT", str(DEFAULT_CAMERA_SIZE[1])))
    return width, height


def open_camera():
    device = os.environ.get("CAMERA_DEVICE", DEFAULT_CAMERA_DEVICE)
    width, height = parse_camera_size()

    print(f"Abriendo camara {device} con OpenCV {cv2.__version__}")
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara: {device}")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def center_crop_square(frame):
    height, width = frame.shape[:2]
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return frame[top : top + side, left : left + side]


def prepare_frame(frame):
    square = center_crop_square(frame)
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
    image = small.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


def predict(model, frame):
    image = prepare_frame(frame)
    return model(image, training=False).numpy()[0]


def draw_predictions(frame, probabilities):
    panel_width = 260
    height = frame.shape[0]
    panel = np.full((height, panel_width, 3), (242, 238, 235), dtype=np.uint8)

    cv2.putText(
        panel,
        "Clasificacion",
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (45, 35, 30),
        2,
        cv2.LINE_AA,
    )

    if probabilities is None:
        cv2.putText(
            panel,
            "Esperando camara",
            (18, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (110, 96, 90),
            1,
            cv2.LINE_AA,
        )
        return np.hstack((frame, panel))

    best_index = int(np.argmax(probabilities))
    cv2.putText(
        panel,
        CLASS_NAMES[best_index],
        (18, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (45, 35, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"{probabilities[best_index] * 100:.1f}%",
        (18, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (110, 96, 90),
        1,
        cv2.LINE_AA,
    )

    y = 170
    for index in np.argsort(probabilities)[::-1][:5]:
        value = float(probabilities[index])
        cv2.putText(
            panel,
            f"{CLASS_NAMES[int(index)]}: {value * 100:.1f}%",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (45, 35, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(panel, (18, y + 14), (218, y + 25), (220, 211, 205), -1)
        cv2.rectangle(
            panel,
            (18, y + 14),
            (18 + int(200 * value), y + 25),
            (230, 110, 25),
            -1,
        )
        y += 55

    return np.hstack((frame, panel))


def main():
    model = load_model()
    capture = open_camera()
    probabilities = None
    last_prediction_time = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("No se pudo leer un frame de la camara.")

            now = time.monotonic()
            if now - last_prediction_time >= 0.5:
                probabilities = predict(model, frame)
                last_prediction_time = now

            output = draw_predictions(frame, probabilities)
            cv2.imshow("Clasificador CIFAR-10", output)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
