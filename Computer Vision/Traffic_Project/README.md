# Proyecto Traffic

Programa de clasificación de señales de tránsito usando TensorFlow y el dataset German Traffic Sign Recognition Benchmark (GTSRB). El dataset debe estar en una carpeta `gtsrb` con subcarpetas numeradas de `0` a `42`, una por cada categoria.

## Preparacion

```bash
chmod +x setup_env.sh
./setup_env.sh
source .venv/bin/activate
```

## Ejecucion

```bash
python traffic.py gtsrb
```

Para guardar el modelo entrenado:

```bash
python traffic.py gtsrb traffic_model.h5
```

## Experimentacion

Primero se implemento una red neuronal convolucional simple, porque el problema consiste en reconocer patrones visuales pequenos dentro de imagenes de senales de transito. La funcion `load_data` recorre las carpetas numeradas del dataset, lee cada imagen con OpenCV, cambia su tamano a 30 x 30 pixeles y guarda la etiqueta segun el nombre de la carpeta. En el modelo se agrego una capa de reescalado para normalizar los pixeles entre 0 y 1 antes de entrenar.

Despues se probo una arquitectura compacta con dos capas convolucionales, dos capas de pooling, una capa densa y dropout. Esta estructura mantiene el codigo sencillo y fiel a las instrucciones, pero aun permite que la red aprenda bordes, formas y combinaciones de colores relevantes para distinguir las 43 categorias. La capa final usa `softmax` con `NUM_CATEGORIES` unidades, y el modelo se compila con `adam`, `categorical_crossentropy` y la metrica `accuracy`.
