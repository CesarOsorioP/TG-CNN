"""
Módulo single-label para clasificación de enfermedades en radiografías de tórax.
Proporciona funcionalidad para entrenar y usar modelos single-label con softmax.

"""

from .model import (
    DenseNetSingleLabelClassifier,
    create_model,
    create_loss_function,
    WeightedCrossEntropyLoss,
    WeightedFocalLoss
)

from .dataset import (
    SingleLabelChestXrayDataset,
    create_data_loaders
)

from .metrics import (
    SingleLabelMetrics,
    evaluate_model_single_label
)

from .predict_singlelabel import (
    SingleLabelDiseasePredictor
)

__all__ = [
    'DenseNetSingleLabelClassifier',
    'create_model',
    'create_loss_function',
    'WeightedCrossEntropyLoss',
    'WeightedFocalLoss',
    'SingleLabelChestXrayDataset',
    'create_data_loaders',
    'SingleLabelMetrics',
    'evaluate_model_single_label',
    'SingleLabelDiseasePredictor'
]

__version__ = '1.0.0'

