# CNN CIFAR-10 con TensorFlow/Keras y Pygame

Entrena una CNN con CIFAR-10 y usa el modelo guardado para clasificar imagenes de una camara en tiempo real con Pygame.

## 1. Crear entorno virtual

```bash
./setup_env.sh
source .venv/bin/activate
```

## 2. Preparar CIFAR-10

Si ya tienes `cifar-10-python.tar.gz` dentro de `data/`, extraelo asi:

```bash
tar -xzf data/cifar-10-python.tar.gz -C data
```

Debe quedar esta carpeta:

```text
data/cifar-10-batches-py/
```

Si esa carpeta no existe, `train_cifar10.py` intentara descargar CIFAR-10 automaticamente con Keras.

## 3. Entrenar el modelo

```bash
python train_cifar10.py
```

El entrenamiento usa:

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
model.save("cifar10_model.h5")
```

## 4. Clasificar con camara

Despues de entrenar y tener `cifar10_model.h5`:

```bash
python clasificador_pygame.py
```

Necesitas una camara accesible para Pygame. En Linux normalmente debe existir `/dev/video0`. En WSL puede no funcionar si la camara no esta expuesta al entorno Linux.

Si la imagen sale distorsionada despues de hacer `usbipd attach`, prueba OpenCV y una resolucion 16:9:

```bash
CAMERA_DEVICE=/dev/video0 CAMERA_BACKEND=opencv CAMERA_WIDTH=1280 CAMERA_HEIGHT=720 python clasificador_pygame.py
```

Tambien puedes cerrar la ventana con `Esc`.

Clases CIFAR-10:

```text
avion, automovil, ave, gato, ciervo, perro, rana, caballo, barco, camion
```
