"""
Script para experimentar con diferentes estrategias de freeze del backbone.
Útil para encontrar la mejor configuración para tu dataset específico.

"""

import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import json
from datetime import datetime

class DenseNetClassifier(nn.Module):
    """Clasificador DenseNet con diferentes estrategias de freeze."""
    
    def __init__(self, num_classes=2, pretrained=True, freeze_strategy='full_freeze'):
        super(DenseNetClassifier, self).__init__()
        
        # Cargar DenseNet pre-entrenado
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Aplicar estrategia de freeze
        self.freeze_strategy = freeze_strategy
        self._apply_freeze_strategy()
        
        # Obtener número de características
        num_features = self.backbone.classifier.in_features
        
        # Reemplazar clasificador
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def _apply_freeze_strategy(self):
        """Aplicar diferentes estrategias de freeze."""
        
        if self.freeze_strategy == 'no_freeze':
            # No congelar nada
            print("🔓 Sin freeze - todos los parámetros entrenables")
            
        elif self.freeze_strategy == 'full_freeze':
            # Congelar todo el backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Freeze completo - solo clasificador entrenable")
            
        elif self.freeze_strategy == 'partial_freeze':
            # Congelar solo las primeras capas
            for name, param in self.backbone.named_parameters():
                if 'features.denseblock1' in name or 'features.denseblock2' in name:
                    param.requires_grad = False
            print("🔒 Freeze parcial - primeras 2 dense blocks congeladas")
            
        elif self.freeze_strategy == 'last_layers_only':
            # Solo entrenar las últimas capas
            for name, param in self.backbone.named_parameters():
                if not ('denseblock4' in name or 'classifier' in name):
                    param.requires_grad = False
            print("🔒 Solo últimas capas - denseblock4 y clasificador entrenables")
            
        elif self.freeze_strategy == 'gradual_unfreeze':
            # Congelar todo inicialmente, se descongelará gradualmente
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Freeze gradual - se descongelará durante entrenamiento")
    
    def unfreeze_gradually(self, epoch, total_epochs):
        """Descongelar gradualmente durante el entrenamiento."""
        if self.freeze_strategy != 'gradual_unfreeze':
            return
        
        # Descongelar por bloques según el progreso
        unfreeze_ratio = epoch / total_epochs
        
        if unfreeze_ratio >= 0.2:  # 20% del entrenamiento
            # Descongelar denseblock4
            for name, param in self.backbone.named_parameters():
                if 'denseblock4' in name:
                    param.requires_grad = True
        
        if unfreeze_ratio >= 0.5:  # 50% del entrenamiento
            # Descongelar denseblock3
            for name, param in self.backbone.named_parameters():
                if 'denseblock3' in name:
                    param.requires_grad = True
        
        if unfreeze_ratio >= 0.8:  # 80% del entrenamiento
            # Descongelar denseblock2
            for name, param in self.backbone.named_parameters():
                if 'denseblock2' in name:
                    param.requires_grad = True

def compare_freeze_strategies():
    """Comparar diferentes estrategias de freeze."""
    
    strategies = [
        'no_freeze',
        'full_freeze', 
        'partial_freeze',
        'last_layers_only',
        'gradual_unfreeze'
    ]
    
    results = {}
    
    print("="*70)
    print("🔬 COMPARACIÓN DE ESTRATEGIAS DE FREEZE")
    print("="*70)
    
    for strategy in strategies:
        print(f"\n📊 Analizando estrategia: {strategy}")
        print("-" * 50)
        
        # Crear modelo
        model = DenseNetClassifier(num_classes=2, freeze_strategy=strategy)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        results[strategy] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'trainable_percentage': trainable_params / total_params * 100
        }
        
        print(f"Total de parámetros: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")
        print(f"Parámetros congelados: {frozen_params:,}")
        print(f"Porcentaje entrenable: {trainable_params/total_params*100:.2f}%")
    
    # Crear visualización
    plot_freeze_comparison(results)
    
    # Guardar resultados
    with open('freeze_strategies_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Resultados guardados en: freeze_strategies_comparison.json")
    
    return results

def plot_freeze_comparison(results):
    """Visualizar comparación de estrategias de freeze."""
    
    strategies = list(results.keys())
    trainable_params = [results[s]['trainable_params'] for s in strategies]
    frozen_params = [results[s]['frozen_params'] for s in strategies]
    trainable_percentages = [results[s]['trainable_percentage'] for s in strategies]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gráfico 1: Parámetros entrenables vs congelados
    x = range(len(strategies))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], trainable_params, width, label='Entrenables', color='skyblue')
    ax1.bar([i + width/2 for i in x], frozen_params, width, label='Congelados', color='lightcoral')
    ax1.set_xlabel('Estrategia de Freeze')
    ax1.set_ylabel('Número de Parámetros')
    ax1.set_title('Parámetros Entrenables vs Congelados')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Porcentaje de parámetros entrenables
    bars = ax2.bar(strategies, trainable_percentages, color='lightgreen')
    ax2.set_xlabel('Estrategia de Freeze')
    ax2.set_ylabel('Porcentaje Entrenable (%)')
    ax2.set_title('Porcentaje de Parámetros Entrenables')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bar, percentage in zip(bars, trainable_percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{percentage:.1f}%', ha='center', va='bottom')
    
    # Gráfico 3: Comparación logarítmica
    ax3.bar(strategies, trainable_params, color='gold', alpha=0.7)
    ax3.set_xlabel('Estrategia de Freeze')
    ax3.set_ylabel('Parámetros Entrenables (log)')
    ax3.set_title('Parámetros Entrenables (Escala Logarítmica)')
    ax3.set_yscale('log')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Recomendaciones
    recommendations = {
        'no_freeze': 'Riesgo de overfitting alto',
        'full_freeze': 'Recomendado para datasets pequeños',
        'partial_freeze': 'Balance entre flexibilidad y estabilidad',
        'last_layers_only': 'Bueno para fine-tuning',
        'gradual_unfreeze': 'Estrategia avanzada'
    }
    
    colors = ['red', 'green', 'orange', 'blue', 'purple']
    bars = ax4.bar(strategies, [1]*len(strategies), color=colors, alpha=0.7)
    ax4.set_xlabel('Estrategia de Freeze')
    ax4.set_ylabel('Recomendación')
    ax4.set_title('Recomendaciones por Estrategia')
    ax4.set_yticks([])
    ax4.tick_params(axis='x', rotation=45)
    
    # Agregar texto de recomendaciones
    for i, (strategy, rec) in enumerate(recommendations.items()):
        ax4.text(i, 0.5, rec, ha='center', va='center', 
                fontsize=10, fontweight='bold', rotation=90)
    
    plt.tight_layout()
    plt.savefig('freeze_strategies_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_recommendations_for_dataset_size(dataset_size):
    """Obtener recomendaciones basadas en el tamaño del dataset."""
    
    print(f"\n💡 RECOMENDACIONES PARA DATASET DE {dataset_size} IMÁGENES")
    print("="*60)
    
    if dataset_size < 500:
        print("🔴 Dataset muy pequeño:")
        print("   ✅ Estrategia recomendada: full_freeze")
        print("   ✅ Batch size: 8-16")
        print("   ✅ Épocas: 20-30")
        print("   ✅ Learning rate: 0.001")
        print("   ⚠️  Evitar fine-tuning")
        
    elif dataset_size < 1000:
        print("🟡 Dataset pequeño:")
        print("   ✅ Estrategia recomendada: full_freeze o partial_freeze")
        print("   ✅ Batch size: 16-32")
        print("   ✅ Épocas: 15-25")
        print("   ✅ Learning rate: 0.001")
        print("   ⚠️  Fine-tuning opcional con learning rate reducido")
        
    elif dataset_size < 5000:
        print("🟢 Dataset mediano:")
        print("   ✅ Estrategia recomendada: partial_freeze o last_layers_only")
        print("   ✅ Batch size: 32-64")
        print("   ✅ Épocas: 10-20")
        print("   ✅ Learning rate: 0.001")
        print("   ✅ Fine-tuning recomendado")
        
    else:
        print("🟢 Dataset grande:")
        print("   ✅ Estrategia recomendada: no_freeze o gradual_unfreeze")
        print("   ✅ Batch size: 64+")
        print("   ✅ Épocas: 10-15")
        print("   ✅ Learning rate: 0.001")
        print("   ✅ Fine-tuning altamente recomendado")

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Experimentar con estrategias de freeze')
    parser.add_argument('--dataset_size', type=int, default=1200,
                       help='Tamaño del dataset para recomendaciones')
    parser.add_argument('--compare', action='store_true',
                       help='Comparar todas las estrategias')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_freeze_strategies()
    
    get_recommendations_for_dataset_size(args.dataset_size)

if __name__ == "__main__":
    main()
