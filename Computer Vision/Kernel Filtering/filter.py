import sys

from PIL import Image, ImageDraw, ImageFilter


if len(sys.argv) != 2:
    sys.exit("Usage: python filter.py filename")



def apply_kernel(image, kernel_info):
    kernel = kernel_info["kernel"]
    scale = kernel_info.get("scale", 1)
    offset = kernel_info.get("offset", 0)

    return image.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=kernel,
        scale=scale,
        offset=offset
    ))


def add_title(image, title):
    title_height = 32
    titled_image = Image.new("RGB", (image.width, image.height + title_height), "white")
    titled_image.paste(image, (0, title_height))

    draw = ImageDraw.Draw(titled_image)
    text_width = draw.textlength(title)
    draw.text(((image.width - text_width) / 2, 9), title, fill="black")

    return titled_image


def create_comparison_grid(images, columns=3):
    rows = (len(images) + columns - 1) // columns
    cell_width = images[0].width
    cell_height = images[0].height

    grid = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")

    for index, image in enumerate(images):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        grid.paste(image, (x, y))

    return grid


def show_image(image):
    try:
        from IPython.display import display
        display(image)
    except Exception:
        image.show()


kernels = {
    "Edge Detect": {
        "kernel": [-1, -1, -1,
                   -1,  8, -1,
                   -1, -1, -1],
        "scale": 1
    },
    "Sharpen": {
        "kernel": [0, -1,  0,
                   -1, 5, -1,
                   0, -1,  0],
        "scale": 1
    },
    "Box Blur": {
        "kernel": [1, 1, 1,
                   1, 1, 1,
                   1, 1, 1],
        "scale": 9
    },
    "Emboss": {
        "kernel": [-2, -1, 0,
                   -1,  1, 1,
                   0,  1, 2],
        "scale": 1,
        "offset": 128
    },
    "Sobel Horizontal": {
        "kernel": [-1, -2, -1,
                   0,  0,  0,
                   1,  2,  1],
        "scale": 1,
        "offset": 128
    },
    "Sobel Vertical": {
        "kernel": [-1, 0, 1,
                   -2, 0, 2,
                   -1, 0, 1],
        "scale": 1,
        "offset": 128
    }
}

observations = {
    "Edge Detect": "Resalta los bordes y cambios fuertes de color. Elimina mucha informacion de color y textura, por eso sirve para encontrar contornos.",
    "Sharpen": "Aumenta el contraste entre pixeles vecinos. Hace que los detalles se vean mas definidos, aunque tambien puede exagerar el ruido.",
    "Box Blur": "Promedia los pixeles cercanos. Suaviza la imagen, reduce detalles finos y puede servir para disminuir ruido.",
    "Emboss": "Da un efecto de relieve. Convierte los cambios de luz en sombras y brillos, util para destacar textura o profundidad.",
    "Sobel Horizontal": "Detecta cambios verticales de intensidad y resalta bordes horizontales. Es util para analizar lineas o separaciones en una direccion.",
    "Sobel Vertical": "Detecta cambios horizontales de intensidad y resalta bordes verticales. Es util para encontrar columnas, paredes o limites laterales."
}


image = Image.open(sys.argv[1]).convert("RGB")

display_image = image.copy()
display_image.thumbnail((360, 240))

comparison_images = [add_title(display_image, "Original")]

for name, kernel_info in kernels.items():
    filtered = apply_kernel(display_image, kernel_info)
    comparison_images.append(add_title(filtered, name))

comparison_grid = create_comparison_grid(comparison_images)
comparison_grid.save("filtered_comparison.png")
show_image(comparison_grid)

print("Observaciones:")
for name, observation in observations.items():
    print(f"- {name}: {observation}")

print("\nImagen comparativa guardada como filtered_comparison.png")
