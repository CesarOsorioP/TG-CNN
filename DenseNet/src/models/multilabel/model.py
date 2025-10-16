"""
Modelo DenseNet para clasificación multi-label de enfermedades en radiografías de tórax.
Basado en la arquitectura del modelo binario existente pero adaptado para multi-label.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

class DenseNetMultiLabelClassifier(nn.Module):
    """
    Clasificador DenseNet para clasificación multi-label de enfermedades.
    
    Arquitectura:
    - Backbone: DenseNet-121 pre-entrenado (congelado opcionalmente)
    - Clasificador: Red personalizada con activación sigmoid
    - Salida: Probabilidades independientes para cada enfermedad
    """
    
    def __init__(self, num_diseases=7, pretrained=True, freeze_backbone=True, 
                 dropout_rate=0.5, hidden_size=512):
        """
        Inicializar modelo multi-label.
        
        Args:
            num_diseases: Número de enfermedades a clasificar
            pretrained: Si usar pesos pre-entrenados
            freeze_backbone: Si congelar el backbone
            dropout_rate: Tasa de dropout
            hidden_size: Tamaño de la capa oculta
        """
        super(DenseNetMultiLabelClassifier, self).__init__()
        
        self.num_diseases = num_diseases
        self.freeze_backbone = freeze_backbone
        
        # Cargar DenseNet-121 pre-entrenado
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Congelar backbone si se especifica
        if freeze_backbone:
            self._freeze_backbone()
        
        # Obtener número de características de la última capa
        num_features = self.backbone.classifier.in_features
        
        # Reemplazar clasificador con red multi-label
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, hidden_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(hidden_size // 2, num_diseases)
        )
        
        # Inicializar pesos del clasificador
        self._initialize_classifier_weights()
        
        print(f"✅ Modelo multi-label inicializado:")
        print(f"   📊 Enfermedades: {num_diseases}")
        print(f"   🔒 Backbone congelado: {freeze_backbone}")
        print(f"   🧠 Características: {num_features}")
        print(f"   🎯 Capa oculta: {hidden_size}")
    
    def _freeze_backbone(self):
        """Congelar todos los parámetros del backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone congelado - solo se entrenará el clasificador")
    
    def unfreeze_backbone(self):
        """Descongelar el backbone para fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔓 Backbone descongelado - fine-tuning habilitado")
    
    def _initialize_classifier_weights(self):
        """
        Inicializar pesos del clasificador con Xavier uniform.
        """
        for module in self.backbone.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass del modelo.
        
        Args:
            x: Tensor de entrada (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Probabilidades para cada enfermedad (batch_size, num_diseases)
        """
        # Obtener características del backbone
        features = self.backbone.features(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        # Clasificación multi-label
        logits = self.backbone.classifier(features)
        
        # Aplicar sigmoid para probabilidades independientes
        probabilities = torch.sigmoid(logits)
        
        return probabilities
    
    def predict_diseases(self, x, threshold=0.5):
        """
        Predecir enfermedades con umbral personalizable.
        
        Args:
            x: Tensor de entrada
            threshold: Umbral para considerar enfermedad presente
            
        Returns:
            dict: Resultados de predicción
        """
        self.eval()
        with torch.no_grad():
            probabilities = self.forward(x)
            predictions = (probabilities > threshold).float()
        
        return {
            'probabilities': probabilities,
            'predictions': predictions,
            'threshold': threshold
        }
    
    def get_model_info(self):
        """
        Obtener información del modelo.
        
        Returns:
            dict: Información del modelo
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'trainable_percentage': (trainable_params / total_params) * 100,
            'num_diseases': self.num_diseases,
            'freeze_backbone': self.freeze_backbone
        }
    
    def print_model_info(self):
        """Imprimir información del modelo."""
        info = self.get_model_info()
        
        print("\n" + "="*50)
        print("🤖 INFORMACIÓN DEL MODELO MULTI-LABEL")
        print("="*50)
        print(f"Total de parámetros: {info['total_parameters']:,}")
        print(f"Parámetros entrenables: {info['trainable_parameters']:,}")
        print(f"Parámetros congelados: {info['frozen_parameters']:,}")
        print(f"Porcentaje entrenable: {info['trainable_percentage']:.2f}%")
        print(f"Número de enfermedades: {info['num_diseases']}")
        print(f"Backbone congelado: {info['freeze_backbone']}")

class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación multi-label.
    Útil para manejar desbalance de clases.
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Inicializar Focal Loss.
        
        Args:
            alpha: Peso de balance de clases
            gamma: Factor de enfoque
            reduction: Tipo de reducción ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Calcular Focal Loss.
        
        Args:
            inputs: Predicciones del modelo (probabilidades)
            targets: Etiquetas verdaderas (0 o 1)
            
        Returns:
            torch.Tensor: Pérdida focal
        """
        # Calcular BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Calcular factor de enfoque
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss con pesos por clase.
    """
    
    def __init__(self, class_weights=None, reduction='mean'):
        """
        Inicializar Weighted BCE Loss.
        
        Args:
            class_weights: Pesos para cada clase
            reduction: Tipo de reducción
        """
        super(WeightedBCELoss, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Calcular Weighted BCE Loss.
        
        Args:
            inputs: Predicciones del modelo
            targets: Etiquetas verdaderas
            
        Returns:
            torch.Tensor: Pérdida ponderada
        """
        # Calcular BCE loss por elemento
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Aplicar pesos si están disponibles
        if self.class_weights is not None:
            weights = self.class_weights.expand_as(targets)
            weighted_loss = bce_loss * weights
        else:
            weighted_loss = bce_loss
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss

def create_model(num_diseases=6, pretrained=True, freeze_backbone=True, 
                dropout_rate=0.5, hidden_size=512):
    """
    Crear modelo multi-label con configuración personalizada.
    
    Args:
        num_diseases: Número de enfermedades
        pretrained: Si usar pesos pre-entrenados
        freeze_backbone: Si congelar backbone
        dropout_rate: Tasa de dropout
        hidden_size: Tamaño de capa oculta
        
    Returns:
        DenseNetMultiLabelClassifier: Modelo configurado
    """
    model = DenseNetMultiLabelClassifier(
        num_diseases=num_diseases,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate,
        hidden_size=hidden_size
    )
    
    return model

def create_loss_function(loss_type='bce', class_weights=None, **kwargs):
    """
    Crear función de pérdida para multi-label.
    
    Args:
        loss_type: Tipo de pérdida ('bce', 'focal', 'weighted_bce')
        class_weights: Pesos de clase para pérdida ponderada
        **kwargs: Argumentos adicionales para la pérdida
        
    Returns:
        nn.Module: Función de pérdida
    """
    if loss_type == 'bce':
        return nn.BCELoss()
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'weighted_bce':
        return WeightedBCELoss(class_weights=class_weights)
    else:
        raise ValueError(f"Tipo de pérdida no soportado: {loss_type}")

if __name__ == "__main__":
    # Ejemplo de uso
    print("🧪 Probando modelo multi-label...")
    
    # Crear modelo
    model = create_model(
        num_diseases=6,
        pretrained=False,  # Para prueba rápida
        freeze_backbone=True
    )
    
    # Imprimir información
    model.print_model_info()
    
    # Probar forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\n🔍 Probando forward pass con batch_size={batch_size}...")
    with torch.no_grad():
        output = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Probar predicción
    print(f"\n🎯 Probando predicción...")
    results = model.predict_diseases(x, threshold=0.5)
    print(f"  Probabilidades shape: {results['probabilities'].shape}")
    print(f"  Predicciones shape: {results['predictions'].shape}")
    
    # Probar diferentes funciones de pérdida
    print(f"\n📊 Probando funciones de pérdida...")
    
    # Crear etiquetas de ejemplo
    targets = torch.randint(0, 2, (batch_size, 6)).float()
    
    # BCE Loss
    bce_loss = create_loss_function('bce')
    bce_value = bce_loss(output, targets)
    print(f"  BCE Loss: {bce_value:.4f}")
    
    # Focal Loss
    focal_loss = create_loss_function('focal', alpha=1.0, gamma=2.0)
    focal_value = focal_loss(output, targets)
    print(f"  Focal Loss: {focal_value:.4f}")
    
    print("\n✅ Pruebas completadas!")
