from pathlib import Path
import os
import sys

import numpy as np
import pygame
import pygame.camera
import tensorflow as tf

try:
    import cv2
except ImportError:
    cv2 = None


MODEL_PATH = Path("cifar10_model.h5")
WINDOW_SIZE = (900, 600)
PREVIEW_RECT = pygame.Rect(20, 60, 640, 480)
DEFAULT_CAMERA_SIZE = (640, 480)
FALLBACK_CAMERA_SIZES = ((640, 480), (1280, 720), (320, 240))

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
    width = os.environ.get("CAMERA_WIDTH")
    height = os.environ.get("CAMERA_HEIGHT")
    if not width or not height:
        return DEFAULT_CAMERA_SIZE

    try:
        return int(width), int(height)
    except ValueError as error:
        raise ValueError(
            "CAMERA_WIDTH y CAMERA_HEIGHT deben ser numeros enteros."
        ) from error


class OpenCVCamera:
    def __init__(self, source, size, force_mjpg=False):
        if cv2 is None:
            raise RuntimeError("OpenCV no esta instalado.")

        capture_source = source
        if isinstance(source, str) and source.isdigit():
            capture_source = int(source)

        self.capture = cv2.VideoCapture(capture_source, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError("no se pudo abrir con OpenCV")

        width, height = size
        if force_mjpg:
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.close()
            raise RuntimeError("OpenCV abrio la camara, pero no pudo leer frames")

    def read_surface(self):
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("No se pudo leer un frame de la camara.")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        return pygame.image.frombuffer(frame.tobytes(), (width, height), "RGB").convert()

    def close(self):
        self.capture.release()


class PygameCamera:
    def __init__(self, source, size):
        self.camera = pygame.camera.Camera(source, size, "RGB")
        self.camera.start()

    def read_surface(self):
        return self.camera.get_image()

    def close(self):
        self.camera.stop()


def dev_video_paths():
    camera_paths = sorted(str(path) for path in Path("/dev").glob("video*"))
    return camera_paths


def unique_candidates(candidates):
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def camera_candidates(backend):
    camera_override = os.environ.get("CAMERA_DEVICE")
    candidates = []

    if camera_override:
        return [camera_override]

    if backend == "opencv":
        candidates.extend(dev_video_paths())
        if not candidates:
            candidates.append("0")
        return unique_candidates(candidates)

    cameras = pygame.camera.list_cameras()
    candidates.extend(cameras)
    candidates.extend(dev_video_paths())

    return unique_candidates(candidates)


def open_camera():
    preferred_size = parse_camera_size()
    backend = os.environ.get("CAMERA_BACKEND", "opencv").lower()
    force_mjpg = os.environ.get("CAMERA_FORCE_MJPG", "0") == "1"
    errors = []

    if backend not in {"auto", "opencv", "pygame"}:
        raise ValueError("CAMERA_BACKEND debe ser auto, opencv o pygame.")

    if backend == "opencv" and cv2 is None:
        raise RuntimeError(
            "CAMERA_BACKEND=opencv requiere instalar OpenCV. Ejecuta: "
            "pip install -r requirements.txt"
        )

    if backend in {"auto", "opencv"} and cv2 is not None:
        for candidate in camera_candidates("opencv"):
            for size in (preferred_size, *FALLBACK_CAMERA_SIZES):
                try:
                    print(
                        f"Probando camara con OpenCV: {candidate} {size}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return OpenCVCamera(candidate, size, force_mjpg=force_mjpg)
                except Exception as error:
                    errors.append(f"OpenCV {candidate} {size}: {error}")

    if backend == "pygame":
        pygame.camera.init()
        candidates = camera_candidates("pygame")

        if not candidates:
            raise RuntimeError(
                "No se encontro una camara disponible. "
                "Verifica que la camara este conectada y accesible por el sistema. "
                "En Linux normalmente deberia existir /dev/video0."
            )

        for candidate in candidates:
            for size in (preferred_size, *FALLBACK_CAMERA_SIZES):
                try:
                    print(
                        f"Probando camara con Pygame: {candidate} {size}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return PygameCamera(candidate, size)
                except Exception as error:
                    errors.append(f"Pygame {candidate} {size}: {error}")

    raise RuntimeError(
        "Se encontraron posibles camaras, pero no se pudo abrir ninguna. "
        "Revisa permisos sobre /dev/video* o prueba con: "
        "CAMERA_DEVICE=/dev/video0 CAMERA_BACKEND=opencv python clasificador_pygame.py\n"
        + "\n".join(errors)
    )


def center_crop_square(surface):
    width, height = surface.get_size()
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return surface.subsurface((left, top, side, side)).copy()


def prepare_frame(frame):
    square = center_crop_square(frame)
    small = pygame.transform.smoothscale(square, (32, 32))
    image = pygame.surfarray.array3d(small)
    image = np.transpose(image, (1, 0, 2))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


def predict(model, frame):
    image = prepare_frame(frame)
    return model(image, training=False).numpy()[0]


def fit_surface(surface, target_rect):
    width, height = surface.get_size()
    scale = min(target_rect.width / width, target_rect.height / height)
    scaled_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    scaled = pygame.transform.smoothscale(surface, scaled_size)
    rect = scaled.get_rect(center=target_rect.center)
    return scaled, rect


def draw_predictions(screen, font, small_font, probabilities):
    panel_x = 680
    pygame.draw.rect(screen, (235, 238, 242), (panel_x, 0, 220, 600))

    title = font.render("Clasificacion", True, (30, 35, 45))
    screen.blit(title, (panel_x + 20, 28))

    if probabilities is None:
        waiting = small_font.render("Esperando camara", True, (90, 96, 110))
        screen.blit(waiting, (panel_x + 20, 76))
        return

    best_index = int(np.argmax(probabilities))
    best_label = font.render(CLASS_NAMES[best_index], True, (30, 35, 45))
    confidence = small_font.render(
        f"{probabilities[best_index] * 100:.1f}%", True, (90, 96, 110)
    )

    screen.blit(best_label, (panel_x + 20, 76))
    screen.blit(confidence, (panel_x + 20, 110))

    top_indices = np.argsort(probabilities)[::-1][:5]
    y = 170

    for index in top_indices:
        value = float(probabilities[index])
        label = small_font.render(
            f"{CLASS_NAMES[int(index)]}: {value * 100:.1f}%", True, (30, 35, 45)
        )
        screen.blit(label, (panel_x + 20, y))
        pygame.draw.rect(screen, (205, 211, 220), (panel_x + 20, y + 28, 160, 10))
        pygame.draw.rect(
            screen, (25, 110, 230), (panel_x + 20, y + 28, int(160 * value), 10)
        )
        y += 58


def main():
    camera = None
    pygame.init()
    pygame.display.set_caption("Clasificador CIFAR-10 con camara")

    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 26, bold=True)
    small_font = pygame.font.SysFont("arial", 18)

    try:
        model = load_model()
        camera = open_camera()

        probabilities = None
        last_prediction_time = 0
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            if not running:
                break

            frame = camera.read_surface()

            now = pygame.time.get_ticks()
            if now - last_prediction_time >= 500:
                probabilities = predict(model, frame)
                last_prediction_time = now

            screen.fill((245, 247, 250))
            pygame.draw.rect(screen, (18, 22, 28), PREVIEW_RECT)
            preview, preview_rect = fit_surface(frame, PREVIEW_RECT)
            screen.blit(preview, preview_rect)

            title = font.render("Camara CIFAR-10", True, (30, 35, 45))
            screen.blit(title, (20, 18))

            draw_predictions(screen, font, small_font, probabilities)

            pygame.display.flip()
            clock.tick(30)
    finally:
        if camera is not None:
            try:
                camera.close()
            finally:
                pygame.camera.quit()
                pygame.quit()
        else:
            pygame.camera.quit()
            pygame.quit()


if __name__ == "__main__":
    main()
