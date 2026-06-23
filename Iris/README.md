# Iris MLP

Proyecto de clasificacion de flores Iris con scikit-learn y TensorFlow/Keras.

## Entorno virtual

TensorFlow 2.x no tiene paquete disponible para Python 3.14 en este equipo. El entorno que funciona para este proyecto es `.venv-tf`, creado con Python 3.12.13 mediante `uv`.

```bash
source .venv-tf/bin/activate
```

Si necesitas recrearlo desde cero en un equipo que solo tenga Python 3.14:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install uv
UV_PYTHON_INSTALL_DIR=.uv-python UV_CACHE_DIR=.uv-cache .venv/bin/uv python install 3.12
UV_PYTHON_INSTALL_DIR=.uv-python UV_CACHE_DIR=.uv-cache .venv/bin/uv venv .venv-tf --python 3.12
UV_CACHE_DIR=.uv-cache .venv/bin/uv pip install --python .venv-tf/bin/python -r requirements.txt
```

## Ejecucion

```bash
source .venv-tf/bin/activate
python iris_mlp.py
```

## Documento

El archivo `documentacion_iris_mlp.docx` ya esta generado. Si necesitas regenerarlo:

```bash
python generar_documento.py
```
