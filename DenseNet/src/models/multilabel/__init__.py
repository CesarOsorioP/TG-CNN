"""
Módulo multi-label para clasificación de enfermedades en radiografías de tórax.
Incluye dataset, modelo, entrenamiento, predicción y evaluación multi-label.

"""

from .dataset import MultiLabelChestXrayDataset, create_data_loaders
from .model import DenseNetMultiLabelClassifier, create_model, create_loss_function
from .metrics import MultiLabelMetrics, evaluate_model_multi_label
from .predict_multilabel import MultiLabelDiseasePredictor

__version__ = "1.0.0"
__author__ = "TG-CNN Team"

__all__ = [
    'MultiLabelChestXrayDataset',
    'create_data_loaders',
    'DenseNetMultiLabelClassifier', 
    'create_model',
    'create_loss_function',
    'MultiLabelMetrics',
    'evaluate_model_multi_label',
    'MultiLabelDiseasePredictor'
]
