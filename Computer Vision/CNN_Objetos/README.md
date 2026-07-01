# CNN CIFAR-10 con TensorFlow/Keras y Pygame

Este proyecto entrena una red neuronal convolucional con CIFAR-10 y luego usa el modelo guardado en una interfaz Pygame para clasificar dibujos.

## Archivos

- `train_cifar10.py`: carga CIFAR-10, normaliza datos, aplica one-hot encoding, entrena la CNN y guarda `cifar10_model.h5`.
- `clasificador_pygame.py`: abre una ventana para dibujar y clasificar usando el modelo entrenado.
- `requirements.txt`: dependencias del proyecto.

## Crear entorno virtual

En esta carpeta:

```bash
./setup_env.sh
source .venv/bin/activate
```

El script usa por defecto Python 3.12.13, porque TensorFlow no tiene wheel compatible con Python 3.14 en esta maquina. Si quieres indicar otro Python compatible:

```bash
PYTHON_BIN=python3.12 ./setup_env.sh
source .venv/bin/activate
```

## Entrenar el modelo

```bash
python train_cifar10.py
```

Por defecto entrena 10 epocas y guarda:

```text
cifar10_model.h5
```

Tambien puedes cambiar parametros:

```bash
python train_cifar10.py --epochs 10 --batch-size 64 --model-path cifar10_model.h5
```

## Ejecutar la interfaz Pygame

Despues de entrenar:

```bash
python clasificador_pygame.py
```

Clases usadas por CIFAR-10:

```text
avion, automovil, ave, gato, ciervo, perro, rana, caballo, barco, camion
```
