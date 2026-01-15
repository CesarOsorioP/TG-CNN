"""
Modelo DenseNet para clasificación single-label de enfermedades en radiografías de tórax.
Basado en la arquitectura del modelo multi-label pero adaptado para single-label con softmax.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

class DenseNetSingleLabelClassifier(nn.Module):
    """
    Clasificador DenseNet para clasificación single-label de enfermedades.
    
    Arquitectura:
    - Backbone: DenseNet-121 pre-entrenado (congelado opcionalmente)
    - Clasificador: Red personalizada con activación softmax
    - Salida: Probabilidades que suman 1.0 (una sola enfermedad por imagen)
    """
    
    def __init__(self, num_classes=8, pretrained=True, freeze_backbone=True, 
                 dropout_rate=0.5, hidden_size=512):
        """
        Inicializar modelo single-label.
        
        Args:
            num_classes: Número de clases (enfermedades) a clasificar
            pretrained: Si usar pesos pre-entrenados
            freeze_backbone: Si congelar el backbone
            dropout_rate: Tasa de dropout
            hidden_size: Tamaño de la capa oculta
        """
        super(DenseNetSingleLabelClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # Cargar DenseNet-121 pre-entrenado
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Congelar backbone si se especifica
        if freeze_backbone:
            self._freeze_backbone()
        
        # Obtener número de características de la última capa
        num_features = self.backbone.classifier.in_features
        
        # Reemplazar clasificador con red single-label
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
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Inicializar pesos del clasificador
        self._initialize_classifier_weights()
        
        print(f"✅ Modelo single-label inicializado:")
        print(f"   📊 Clases: {num_classes}")
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
            torch.Tensor: Logits para cada clase (batch_size, num_classes)
            NOTA: No aplica softmax aquí, se aplica en la pérdida o en predicción
        """
        # Obtener características del backbone
        features = self.backbone.features(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        # Clasificación single-label (logits sin softmax)
        logits = self.backbone.classifier(features)
        
        return logits
    
    def predict_class(self, x):
        """
        Predecir clase para una imagen.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            dict: Resultados de predicción con probabilidades softmax
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'predicted_class': predicted_class
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
            'num_classes': self.num_classes,
            'freeze_backbone': self.freeze_backbone
        }
    
    def print_model_info(self):
        """Imprimir información del modelo."""
        info = self.get_model_info()
        
        print("\n" + "="*50)
        print("🤖 INFORMACIÓN DEL MODELO SINGLE-LABEL")
        print("="*50)
        print(f"Total de parámetros: {info['total_parameters']:,}")
        print(f"Parámetros entrenables: {info['trainable_parameters']:,}")
        print(f"Parámetros congelados: {info['frozen_parameters']:,}")
        print(f"Porcentaje entrenable: {info['trainable_percentage']:.2f}%")
        print(f"Número de clases: {info['num_classes']}")
        print(f"Backbone congelado: {info['freeze_backbone']}")

class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss con pesos por clase.
    Útil para manejar desbalance de clases.
    """
    
    def __init__(self, class_weights=None, reduction='mean'):
        """
        Inicializar Weighted Cross Entropy Loss.
        
        Args:
            class_weights: Pesos para cada clase (tensor)
            reduction: Tipo de reducción ('mean', 'sum', 'none')
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Calcular Weighted Cross Entropy Loss.
        
        Args:
            inputs: Logits del modelo (batch_size, num_classes)
            targets: Etiquetas verdaderas (batch_size) - índices de clase
            
        Returns:
            torch.Tensor: Pérdida ponderada
        """
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights, reduction=self.reduction)
        else:
            criterion = nn.CrossEntropyLoss(reduction=self.reduction)
        
        return criterion(inputs, targets)

class WeightedFocalLoss(nn.Module):
    """
    Focal Loss con pesos por clase para clasificación single-label.
    Combina Focal Loss (enfoca en ejemplos difíciles) con pesos por clase (balancea clases).
    """
    
    def __init__(self, class_weights=None, gamma=2.0, alpha=None, reduction='mean'):
        """
        Inicializar Weighted Focal Loss.
        
        Args:
            class_weights: Pesos para cada clase (tensor) - para balancear clases
            gamma: Factor de enfoque (mayor = más enfoque en ejemplos difíciles)
            alpha: Pesos alpha por clase (alternativa a class_weights, opcional)
            reduction: Tipo de reducción ('mean', 'sum', 'none')
        """
        super(WeightedFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Calcular Weighted Focal Loss.
        
        Args:
            inputs: Logits del modelo (batch_size, num_classes)
            targets: Etiquetas verdaderas (batch_size) - índices de clase
            
        Returns:
            torch.Tensor: Pérdida focal ponderada
        """
        # Calcular probabilidades softmax
        probs = F.softmax(inputs, dim=1)
        
        # Obtener probabilidad de la clase correcta (pt)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calcular CrossEntropyLoss base (sin reducción)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        
        # Calcular factor focal: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Aplicar pesos alpha si están disponibles
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = torch.full_like(targets, self.alpha, dtype=torch.float32)
            else:
                # alpha es un tensor con pesos por clase
                alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight
        
        # Calcular focal loss: alpha * (1 - pt)^gamma * CE_loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_model(num_classes=8, pretrained=True, freeze_backbone=True, 
                dropout_rate=0.5, hidden_size=512):
    """
    Crear modelo single-label con configuración personalizada.
    
    Args:
        num_classes: Número de clases
        pretrained: Si usar pesos pre-entrenados
        freeze_backbone: Si congelar backbone
        dropout_rate: Tasa de dropout
        hidden_size: Tamaño de capa oculta
        
    Returns:
        DenseNetSingleLabelClassifier: Modelo configurado
    """
    model = DenseNetSingleLabelClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate,
        hidden_size=hidden_size
    )
    
    return model

def create_loss_function(loss_type='ce', class_weights=None, **kwargs):
    """
    Crear función de pérdida para single-label.
    
    Args:
        loss_type: Tipo de pérdida ('ce', 'weighted_ce', 'focal', 'weighted_focal')
        class_weights: Pesos de clase para pérdida ponderada
        **kwargs: Argumentos adicionales (gamma, alpha para focal loss)
        
    Returns:
        nn.Module: Función de pérdida
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'weighted_ce':
        return WeightedCrossEntropyLoss(class_weights=class_weights)
    elif loss_type == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        alpha = kwargs.get('alpha', None)
        return WeightedFocalLoss(class_weights=None, gamma=gamma, alpha=alpha)
    elif loss_type == 'weighted_focal':
        gamma = kwargs.get('gamma', 2.0)
        alpha = kwargs.get('alpha', None)
        return WeightedFocalLoss(class_weights=class_weights, gamma=gamma, alpha=alpha)
    else:
        raise ValueError(f"Tipo de pérdida no soportado: {loss_type}")

if __name__ == "__main__":
    # Ejemplo de uso
    print("🧪 Probando modelo single-label...")
    
    # Crear modelo
    model = create_model(
        num_classes=8,
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
        logits = model(x)
        probabilities = F.softmax(logits, dim=1)
        print(f"  Input shape: {x.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Probabilities shape: {probabilities.shape}")
        print(f"  Probabilities sum (debe ser 1.0): {probabilities.sum(dim=1)}")
    
    # Probar predicción
    print(f"\n🎯 Probando predicción...")
    results = model.predict_class(x)
    print(f"  Predicted classes: {results['predicted_class']}")
    print(f"  Probabilities shape: {results['probabilities'].shape}")
    
    # Probar función de pérdida
    print(f"\n📊 Probando función de pérdida...")
    
    # Crear etiquetas de ejemplo (índices de clase)
    targets = torch.randint(0, 8, (batch_size,))
    
    # Cross Entropy Loss
    ce_loss = create_loss_function('ce')
    ce_value = ce_loss(logits, targets)
    print(f"  Cross Entropy Loss: {ce_value:.4f}")
    
    # Focal Loss
    focal_loss = create_loss_function('focal', gamma=2.0)
    focal_value = focal_loss(logits, targets)
    print(f"  Focal Loss (gamma=2.0): {focal_value:.4f}")
    
    print("\n✅ Pruebas completadas!")

