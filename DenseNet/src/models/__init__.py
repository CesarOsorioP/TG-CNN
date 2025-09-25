"""
Módulo de Modelos
=================

Contiene las clases y funciones para el entrenamiento y predicción del modelo DenseNet.

Archivos:
- train_model.py: Entrenamiento del modelo con transfer learning
- predict.py: Predicción individual de imágenes
- freeze_experiments.py: Experimentos con diferentes estrategias de freeze
"""

from .train_model import DenseNetClassifier, train_model
from .predict import ChestXrayPredictor

__all__ = ['DenseNetClassifier', 'train_model', 'ChestXrayPredictor']
