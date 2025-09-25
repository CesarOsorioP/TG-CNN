# Estructura del Proyecto DenseNet

## 📁 Organización de Carpetas

### `src/` - Código Fuente Principal
Contiene todo el código fuente organizado en módulos:

#### `src/models/` - Modelos y Entrenamiento
- **`train_model.py`**: Entrenamiento del modelo DenseNet con transfer learning
- **`predict.py`**: Predicción individual de imágenes
- **`freeze_experiments.py`**: Experimentos con diferentes estrategias de freeze

#### `src/analysis/` - Análisis y Visualización
- **`analyze_model.py`**: Análisis detallado del modelo entrenado
- **`analyze_batch_results.py`**: Análisis de resultados de procesamiento en lote

#### `src/utils/` - Utilidades Auxiliares
- Funciones comunes y utilidades del proyecto
- Preparado para futuras expansiones

### `gui/` - Interfaz Gráfica
- **`gui_app.py`**: Aplicación principal de la interfaz gráfica

### `scripts/` - Scripts de Utilidad
- **`batch_predictor.py`**: Procesamiento en lote avanzado
- **`quick_batch.py`**: Procesamiento en lote rápido
- **`quick_start.py`**: Guía interactiva paso a paso
- **`setup_data.py`**: Preparación y análisis de datos

### `data/` - Datos de Entrenamiento
- **`chest_xray/`**: Imágenes de radiografías de tórax
- **`other_images/`**: Otras imágenes (no radiografías)

### `results/` - Resultados y Modelos
- **`models/`**: Modelos entrenados (.pth, .json)
- **`predictions/`**: Resultados de predicción (CSV, TXT)
- **`analysis/`**: Resultados de análisis (gráficos, reportes)

### `docs/` - Documentación
- **`STRUCTURE.md`**: Este archivo
- Documentación adicional del proyecto

## 🔧 Archivos de Configuración

### `main.py` - Script Principal
Punto de entrada único para todas las funcionalidades:
```bash
python main.py [comando] [opciones]
```

### `config.py` - Configuración Centralizada
Configuración de rutas, parámetros y constantes del proyecto.

### `requirements.txt` - Dependencias
Lista de paquetes Python necesarios.

## 📦 Módulos Python

Cada carpeta `src/` contiene un archivo `__init__.py` que:
- Define las importaciones públicas del módulo
- Facilita el uso de `from src.models import ...`
- Documenta el propósito de cada módulo

## 🚀 Flujo de Trabajo Recomendado

1. **Preparación de datos**: `python scripts/setup_data.py`
2. **Entrenamiento**: `python main.py train`
3. **Predicción individual**: `python main.py predict imagen.jpg`
4. **Interfaz gráfica**: `python main.py gui`
5. **Procesamiento en lote**: `python main.py batch directorio/`
6. **Análisis**: `python main.py analyze`

## 🔄 Migración desde Estructura Anterior

Si tenías el proyecto con la estructura anterior:
1. Los archivos se han movido automáticamente
2. Usa `python main.py` en lugar de scripts individuales
3. Los modelos se guardan en `results/models/`
4. Los resultados se guardan en `results/predictions/`

## 📝 Convenciones

- **Nombres de archivos**: snake_case
- **Nombres de clases**: PascalCase
- **Nombres de funciones**: snake_case
- **Constantes**: UPPER_CASE
- **Imports**: Organizados por tipo (estándar, terceros, locales)
