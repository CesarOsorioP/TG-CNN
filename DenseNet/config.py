"""
Configuración del Proyecto DenseNet Chest X-Ray Classifier
==========================================================

Archivo de configuración centralizado para todas las rutas y parámetros del proyecto.
"""

import os
from pathlib import Path

# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent

# Rutas principales
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

# Rutas de datos
CHEST_XRAY_DIR = DATA_DIR / "chest_xray"
OTHER_IMAGES_DIR = DATA_DIR / "other_images"

# Rutas de modelos
DEFAULT_MODEL_PATH = MODELS_DIR / "densenet_chest_xray_model.pth"
TRAINING_CONFIG_PATH = MODELS_DIR / "training_config.json"

# Configuración de entrenamiento
TRAINING_CONFIG = {
    "num_epochs": 15,
    "batch_size": 16,
    "learning_rate": 0.001,
    "freeze_backbone": True,
    "fine_tune_epochs": 5,
    "image_size": (224, 224),
    "num_classes": 2,
    "class_names": ["chest_xray", "other_images"]
}

# Configuración de predicción
PREDICTION_CONFIG = {
    "confidence_threshold": 0.5,
    "batch_size": 32,
    "supported_formats": ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
}

# Configuración de la GUI
GUI_CONFIG = {
    "window_title": "DenseNet Chest X-Ray Classifier",
    "window_size": (1000, 700),
    "image_display_size": (600, 400),
    "theme": "modern"
}

# Configuración de procesamiento en lote
BATCH_CONFIG = {
    "default_batch_size": 32,
    "max_workers": 4,
    "save_detailed_results": True,
    "save_summary": True
}

# Crear directorios si no existen
def ensure_directories():
    """Crear todos los directorios necesarios si no existen."""
    directories = [
        DATA_DIR, RESULTS_DIR, MODELS_DIR, PREDICTIONS_DIR, ANALYSIS_DIR,
        CHEST_XRAY_DIR, OTHER_IMAGES_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Configuración de logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": RESULTS_DIR / "project.log"
}

# Configuración de dispositivos
DEVICE_CONFIG = {
    "auto_detect": True,
    "prefer_cuda": True,
    "fallback_to_cpu": True
}

if __name__ == "__main__":
    # Crear directorios al ejecutar este archivo
    ensure_directories()
    print("✅ Directorios del proyecto creados/verificados")
    print(f"📁 Directorio raíz: {PROJECT_ROOT}")
    print(f"📊 Datos: {DATA_DIR}")
    print(f"🤖 Modelos: {MODELS_DIR}")
    print(f"📈 Resultados: {PREDICTIONS_DIR}")
