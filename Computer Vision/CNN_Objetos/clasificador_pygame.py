from pathlib import Path

import numpy as np
import pygame
import tensorflow as tf


MODEL_PATH = Path("cifar10_model.h5")
WIDTH, HEIGHT = 900, 620
CANVAS_SIZE = 512
CANVAS_POS = (32, 54)
PANEL_X = 590
BACKGROUND = (245, 247, 250)
PANEL = (232, 237, 243)
TEXT = (32, 37, 45)
MUTED = (95, 105, 120)
ACCENT = (28, 115, 232)
BAR_BG = (210, 217, 226)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

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


class Button:
    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect)
        self.text = text

    def draw(self, screen, font):
        pygame.draw.rect(screen, ACCENT, self.rect, border_radius=6)
        label = font.render(self.text, True, WHITE)
        label_rect = label.get_rect(center=self.rect.center)
        screen.blit(label, label_rect)

    def clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontro cifar10_model.h5. Ejecuta primero: python train_cifar10.py"
        )
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(canvas):
    """Convierte el dibujo de Pygame a un tensor compatible con CIFAR-10."""
    small = pygame.transform.smoothscale(canvas, (32, 32))
    image = pygame.surfarray.array3d(small)
    image = np.transpose(image, (1, 0, 2))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


def predict(model, canvas):
    image = prepare_image(canvas)
    probabilities = model.predict(image, verbose=0)[0]
    return probabilities


def draw_prediction_panel(screen, title_font, font, small_font, probabilities):
    pygame.draw.rect(screen, PANEL, (PANEL_X, 54, 278, 512), border_radius=8)

    title = title_font.render("Clasificacion", True, TEXT)
    screen.blit(title, (PANEL_X + 24, 82))

    if probabilities is None:
        waiting = font.render("Esperando dibujo", True, MUTED)
        screen.blit(waiting, (PANEL_X + 24, 132))
        return

    best_index = int(np.argmax(probabilities))
    best_text = font.render(CLASS_NAMES[best_index], True, TEXT)
    confidence = small_font.render(f"{probabilities[best_index] * 100:.1f}%", True, MUTED)
    screen.blit(best_text, (PANEL_X + 24, 126))
    screen.blit(confidence, (PANEL_X + 24, 156))

    top_indices = np.argsort(probabilities)[::-1][:5]
    y = 214
    for index in top_indices:
        name = CLASS_NAMES[int(index)]
        value = float(probabilities[index])
        label = small_font.render(f"{name}  {value * 100:.1f}%", True, TEXT)
        screen.blit(label, (PANEL_X + 24, y))

        bar_width = 220
        pygame.draw.rect(screen, BAR_BG, (PANEL_X + 24, y + 28, bar_width, 10), border_radius=5)
        pygame.draw.rect(
            screen,
            ACCENT,
            (PANEL_X + 24, y + 28, int(bar_width * value), 10),
            border_radius=5,
        )
        y += 62


def draw_canvas_border(screen):
    pygame.draw.rect(screen, WHITE, (*CANVAS_POS, CANVAS_SIZE, CANVAS_SIZE), border_radius=8)
    pygame.draw.rect(screen, (200, 208, 218), (*CANVAS_POS, CANVAS_SIZE, CANVAS_SIZE), 2, 8)


def point_inside_canvas(position):
    x, y = position
    canvas_x, canvas_y = CANVAS_POS
    return (
        canvas_x <= x < canvas_x + CANVAS_SIZE
        and canvas_y <= y < canvas_y + CANVAS_SIZE
    )


def to_canvas_point(position):
    return position[0] - CANVAS_POS[0], position[1] - CANVAS_POS[1]


def main():
    pygame.init()
    pygame.display.set_caption("CNN CIFAR-10")

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("arial", 28, bold=True)
    font = pygame.font.SysFont("arial", 24, bold=True)
    small_font = pygame.font.SysFont("arial", 18)

    model = load_model()
    canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
    canvas.fill(BLACK)

    clear_button = Button((PANEL_X + 24, 508, 110, 36), "Limpiar")
    prediction = None
    drawing = False
    last_point = None
    last_prediction_time = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif clear_button.clicked(event):
                canvas.fill(BLACK)
                prediction = None
            elif event.type == pygame.MOUSEBUTTONDOWN and point_inside_canvas(event.pos):
                drawing = True
                last_point = to_canvas_point(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                last_point = None
            elif event.type == pygame.MOUSEMOTION and drawing and point_inside_canvas(event.pos):
                current_point = to_canvas_point(event.pos)
                if last_point is not None:
                    pygame.draw.line(canvas, WHITE, last_point, current_point, width=18)
                pygame.draw.circle(canvas, WHITE, current_point, 9)
                last_point = current_point

        now = pygame.time.get_ticks()
        if now - last_prediction_time > 300:
            if pygame.surfarray.array3d(canvas).max() > 0:
                prediction = predict(model, canvas)
            last_prediction_time = now

        screen.fill(BACKGROUND)
        title = title_font.render("CNN CIFAR-10", True, TEXT)
        screen.blit(title, (32, 16))
        draw_canvas_border(screen)
        screen.blit(canvas, CANVAS_POS)
        draw_prediction_panel(screen, title_font, font, small_font, prediction)
        clear_button.draw(screen, small_font)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
