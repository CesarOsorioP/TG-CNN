"""
Script de entrenamiento para modelo single-label de clasificación de enfermedades.
Basado en el script de entrenamiento multi-label pero adaptado para single-label con softmax.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import argparse
from pathlib import Path
import torch.nn.functional as F

# Intentar importar DirectML para GPUs AMD
try:
    import torch_directml
    DML_AVAILABLE = True
except ImportError:
    DML_AVAILABLE = False

# Importar módulos del proyecto
try:
    from .dataset import create_data_loaders
    from .model import create_model, create_loss_function
    from .metrics import SingleLabelMetrics, evaluate_model_single_label
except ImportError:
    from dataset import create_data_loaders
    from model import create_model, create_loss_function
    from metrics import SingleLabelMetrics, evaluate_model_single_label


def get_transforms():
    """
    Obtener transformaciones de datos para entrenamiento y validación.
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Entrenar una época del modelo.
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        criterion: Función de pérdida
        optimizer: Optimizador
        device: Dispositivo a usar
        epoch: Número de época
        
    Returns:
        dict: Métricas de la época
    """
    model.train()
    train_loss = 0.0
    all_predictions = []
    all_targets = []
    correct = 0
    total = 0
    
    train_pbar = tqdm(train_loader, desc=f"Época {epoch+1} - Entrenando")
    
    for batch_idx, (data, target) in enumerate(train_pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Calcular predicciones (argmax para single-label)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        # Guardar para métricas
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        # Calcular accuracy
        correct += (predictions == target).sum().item()
        total += target.size(0)
        
        train_pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    # Calcular métricas de entrenamiento
    train_loss /= len(train_loader)
    train_accuracy = 100. * correct / total
    
    # Calcular F1-Score
    from sklearn.metrics import f1_score, accuracy_score
    f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
    
    return {
        'loss': train_loss,
        'accuracy': train_accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def validate_epoch(model, val_loader, criterion, device, class_names, epoch):
    """
    Validar una época del modelo.
    
    Args:
        model: Modelo a validar
        val_loader: DataLoader de validación
        criterion: Función de pérdida
        device: Dispositivo a usar
        class_names: Lista de nombres de clases
        epoch: Número de época
        
    Returns:
        dict: Métricas de validación
    """
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    correct = 0
    total = 0
    
    val_pbar = tqdm(val_loader, desc=f"Época {epoch+1} - Validando")
    
    with torch.no_grad():
        for data, target in val_pbar:
            data, target = data.to(device), target.to(device)
            
            logits = model(data)
            loss = criterion(logits, target)
            
            val_loss += loss.item()
            
            # Calcular predicciones
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Guardar para métricas
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Calcular accuracy
            correct += (predictions == target).sum().item()
            total += target.size(0)
            
            val_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    # Calcular métricas de validación
    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total
    
    # Calcular métricas single-label
    metrics_calculator = SingleLabelMetrics(class_names)
    metrics = metrics_calculator.calculate_metrics(
        np.array(all_targets), 
        np.array(all_predictions), 
        np.array(all_probabilities)
    )
    
    return {
        'loss': val_loss,
        'accuracy': val_accuracy,
        'f1_macro': metrics['f1_macro'],
        'f1_weighted': metrics['f1_weighted'],
        'precision_macro': metrics['precision_macro'],
        'recall_macro': metrics['recall_macro']
    }

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, 
                class_names, device, loss_type='ce', class_weights=None, **kwargs):
    """
    Entrenar el modelo single-label.
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        num_epochs: Número de épocas
        learning_rate: Tasa de aprendizaje
        class_names: Lista de nombres de clases
        device: Dispositivo a usar
        loss_type: Tipo de función de pérdida
        class_weights: Pesos de clase (opcional)
        **kwargs: Argumentos adicionales (gamma, alpha para focal loss)
        
    Returns:
        dict: Historial de entrenamiento
    """
    # Configurar función de pérdida
    criterion = create_loss_function(loss_type, class_weights, **kwargs)
    criterion = criterion.to(device)
    
    # Configurar optimizador
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Configurar scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # Monitorea accuracy (maximizar)
        factor=0.5,           # Reduce a la mitad
        patience=5,           # Espera 5 épocas sin mejora
        min_lr=1e-6           # LR mínimo
    )
    
    # Historial de entrenamiento
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_f1_macro': [],
        'val_f1_macro': [],
        'train_f1_weighted': [],
        'val_f1_weighted': [],
        'val_precision_macro': [],
        'val_recall_macro': []
    }
    
    best_val_accuracy = 0.0
    best_model_state = None
    patience = 10
    patience_counter = 0
    min_delta = 0.1  # Mejora mínima del 0.1% en accuracy
    
    print(f"Iniciando entrenamiento single-label...")
    print(f"Épocas: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Dispositivo: {device}")
    print(f"Función de pérdida: {loss_type}")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"ÉPOCA {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Entrenar
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validar
        val_metrics = validate_epoch(model, val_loader, criterion, device, class_names, epoch)
        
        # Actualizar historial
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['train_f1_macro'].append(train_metrics['f1_macro'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['train_f1_weighted'].append(train_metrics['f1_weighted'])
        history['val_f1_weighted'].append(val_metrics['f1_weighted'])
        history['val_precision_macro'].append(val_metrics['precision_macro'])
        history['val_recall_macro'].append(val_metrics['recall_macro'])
        
        # Imprimir métricas
        print(f"\n📊 MÉTRICAS DE LA ÉPOCA:")
        print(f"  Entrenamiento - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.2f}%, F1-Macro: {train_metrics['f1_macro']:.4f}")
        print(f"  Validación - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.2f}%, F1-Macro: {val_metrics['f1_macro']:.4f}")
        print(f"  Precision (Macro): {val_metrics['precision_macro']:.4f}")
        print(f"  Recall (Macro): {val_metrics['recall_macro']:.4f}")
        
        # Guardar mejor modelo (con mejora mínima)
        if val_metrics['accuracy'] > best_val_accuracy + min_delta:
            best_val_accuracy = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✅ ¡Nuevo mejor modelo! Accuracy: {best_val_accuracy:.2f}%")
        else:
            patience_counter += 1
            print(f"  ⏳ Sin mejora significativa ({patience_counter}/{patience})")
        
        # Actualizar scheduler
        scheduler.step(val_metrics['accuracy'])
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping activado después de {patience} épocas sin mejora")
            break
    
    # Cargar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n🏆 Mejor modelo cargado - Accuracy: {best_val_accuracy:.2f}%")
    
    return history

def plot_training_history(history, save_path=None):
    """
    Visualizar el historial de entrenamiento.
    
    Args:
        history: Diccionario con historial de entrenamiento
        save_path: Ruta para guardar la figura
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Entrenamiento', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validación', linewidth=2)
    axes[0, 0].set_title('Pérdida durante el entrenamiento')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Pérdida')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Entrenamiento', linewidth=2)
    axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Validación', linewidth=2)
    axes[0, 1].set_title('Accuracy durante el entrenamiento')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1-Score Macro
    axes[0, 2].plot(epochs, history['train_f1_macro'], 'b-', label='Entrenamiento', linewidth=2)
    axes[0, 2].plot(epochs, history['val_f1_macro'], 'r-', label='Validación', linewidth=2)
    axes[0, 2].set_title('F1-Score (Macro) durante el entrenamiento')
    axes[0, 2].set_xlabel('Época')
    axes[0, 2].set_ylabel('F1-Score (Macro)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # F1-Score Weighted
    axes[1, 0].plot(epochs, history['train_f1_weighted'], 'b-', label='Entrenamiento', linewidth=2)
    axes[1, 0].plot(epochs, history['val_f1_weighted'], 'r-', label='Validación', linewidth=2)
    axes[1, 0].set_title('F1-Score (Weighted) durante el entrenamiento')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('F1-Score (Weighted)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Precision
    axes[1, 1].plot(epochs, history['val_precision_macro'], 'g-', label='Validación', linewidth=2)
    axes[1, 1].set_title('Precision (Macro) durante el entrenamiento')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Precision (Macro)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Recall
    axes[1, 2].plot(epochs, history['val_recall_macro'], 'm-', label='Validación', linewidth=2)
    axes[1, 2].set_title('Recall (Macro) durante el entrenamiento')
    axes[1, 2].set_xlabel('Época')
    axes[1, 2].set_ylabel('Recall (Macro)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Gráfico de entrenamiento guardado en: {save_path}")
    
    plt.show()

def main():
    """
    Función principal para entrenar el modelo single-label.
    """
    parser = argparse.ArgumentParser(description='Entrenar modelo single-label de enfermedades')
    parser.add_argument('--data_dir', type=str, default='data_diseases',
                       help='Directorio con datos single-label')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Tamaño del lote')
    parser.add_argument('--num_epochs', type=int, default=15,
                       help='Número de épocas')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                       help='Tasa de aprendizaje para backbone congelado')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                       help='Congelar backbone durante entrenamiento')
    parser.add_argument('--fine_tune_epochs', type=int, default=15,
                       help='Épocas de fine-tuning')
    parser.add_argument('--fine_tune_lr', type=float, default=0.00001,
                       help='Tasa de aprendizaje para fine-tuning')
    parser.add_argument('--loss_type', type=str, default='weighted_focal',
                       choices=['ce', 'weighted_ce', 'focal', 'weighted_focal'],
                       help='Tipo de función de pérdida')
    parser.add_argument('--gamma', type=float, default=2.5,
                       help='Factor gamma para Focal Loss (mayor = más enfoque en ejemplos difíciles)')
    parser.add_argument('--alpha', type=float, default=None,
                       help='Peso alpha para Focal Loss (opcional, para balance adicional)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu', 'dml'],
                       help='Dispositivo a usar (auto, cuda, cpu o dml para AMD)')
    parser.add_argument('--output_dir', type=str, default='results/models/singlelabel',
                       help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Usando NVIDIA CUDA")
        elif DML_AVAILABLE:
            device = torch_directml.device()
            print("Usando AMD DirectML")
        else:
            device = torch.device('cpu')
            print("Usando CPU")
    elif args.device == 'dml':
        if DML_AVAILABLE:
            device = torch_directml.device()
            print("Usando AMD DirectML forzado")
        else:
            print("Error: torch_directml no está instalado. Usando CPU...")
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print("ENTRENAMIENTO DE MODELO SINGLE-LABEL")
    print("="*60)
    print(f"Directorio de datos: {args.data_dir}")
    print(f"Tamaño de lote: {args.batch_size}")
    print(f"Épocas: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Backbone congelado: {args.freeze_backbone}")
    print(f"Dispositivo: {device}")
    print(f"Función de pérdida: {args.loss_type}")
    if args.loss_type in ['focal', 'weighted_focal']:
        print(f"Gamma (Focal Loss): {args.gamma}")
        if args.alpha is not None:
            print(f"Alpha (Focal Loss): {args.alpha}")
    
    # Verificar directorio de datos
    if not os.path.exists(args.data_dir):
        print(f"Error: Directorio {args.data_dir} no encontrado")
        print("Ejecuta primero: python src/models/multilabel/prepare_data.py")
        return
    
    # Obtener transformaciones
    train_transform, val_transform = get_transforms()
    
    # Crear DataLoaders
    print(f"\nPreparando datos...")
    train_loader, val_loader, test_loader, dataset_stats = create_data_loaders(
        data_dir=args.data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=args.batch_size,
        include_normal=True,
        num_workers=0
    )
    
    # Definir nombres de clases (incluyendo Normal)
    class_names = [
        'Neumonía', 'Atelectasia', 'Edema', 
        'Tuberculosis', 'COVID-19', 'Normal', 'Nodules', 'Mass'
    ]
    
    # Crear modelo
    print(f"\nCreando modelo...")
    model = create_model(
        num_classes=len(class_names),
        pretrained=True,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    
    # Imprimir información del modelo
    model.print_model_info()
    
    # Calcular pesos de clase si es necesario
    class_weights = None
    if args.loss_type == 'weighted_ce':
        # Crear dataset temporal para calcular pesos
        try:
            from .dataset import SingleLabelChestXrayDataset
        except ImportError:
            from dataset import SingleLabelChestXrayDataset
        temp_dataset = SingleLabelChestXrayDataset(args.data_dir, transform=None)
        class_weights = temp_dataset.get_class_weights()
        class_weights = class_weights.to(device)
        print(f"Pesos de clase calculados: {class_weights}")
    
    # Entrenar modelo
    print(f"\nIniciando entrenamiento...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        class_names=class_names,
        device=device,
        loss_type=args.loss_type,
        class_weights=class_weights,
        gamma=args.gamma,
        alpha=args.alpha
    )
    
    # Guardar checkpoint PRE fine-tuning
    os.makedirs(args.output_dir, exist_ok=True)
    pre_ft_path = os.path.join(args.output_dir, 'densenet_singlelabel_pre_ft.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': len(class_names),
        'history': history,
        'config': vars(args)
    }, pre_ft_path)
    print(f"\n💾 Checkpoint pre fine-tuning guardado en: {pre_ft_path}")
    
    # LIMPIEZA EXPLÍCITA DE MEMORIA ANTES DEL FINE-TUNING
    import gc
    del train_loader
    del val_loader
    gc.collect()
    
    # Vaciar caché de GPU si es posible
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif 'privateuseone' in str(device): # DirectML usa este nombre a veces
        pass

    # Fine-tuning (opcional)
    if args.fine_tune_epochs > 0 and args.freeze_backbone:
        try:
            print(f"\n🔓 Iniciando fine-tuning...")
            model.unfreeze_backbone()
            
            fine_tune_lr = args.fine_tune_lr
            print(f"📈 Learning rate para fine-tuning: {fine_tune_lr}")
            
            # Re-crear DataLoaders para fine-tuning con batch_size reducido
            print(f"🔁 Re-creando DataLoaders para fine-tuning con batch_size=12...")
            train_loader_ft, val_loader_ft, test_loader_ft, _ = create_data_loaders(
                data_dir=args.data_dir,
                train_transform=train_transform,
                val_transform=val_transform,
                batch_size=8,
                include_normal=True,
                num_workers=0
            )

            fine_tune_history = train_model(
                model=model,
                train_loader=train_loader_ft,
                val_loader=val_loader_ft,
                num_epochs=args.fine_tune_epochs,
                learning_rate=fine_tune_lr,
                class_names=class_names,
                device=device,
                loss_type=args.loss_type,
                class_weights=class_weights,
                gamma=args.gamma,
                alpha=args.alpha
            )
            
            # Combinar historiales
            for key in history:
                history[key].extend(fine_tune_history[key])
            
            print(f"✅ Fine-tuning completado")
        except Exception as e:
            print(f"\n⚠️ Fine-tuning detenido por error: {e}")
            print(f"Se conserva el checkpoint pre fine-tuning en: {pre_ft_path}")
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Guardar modelo final
    model_path = os.path.join(args.output_dir, 'densenet_singlelabel_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': len(class_names),
        'history': history,
        'config': vars(args)
    }, model_path)
    print(f"\n💾 Modelo guardado en: {model_path}")
    
    # Visualizar historial de entrenamiento
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Evaluar en conjunto de prueba
    print(f"\n📊 Evaluando en conjunto de prueba...")
    test_metrics, test_predictions, test_targets, test_probabilities = evaluate_model_single_label(
        model, test_loader, class_names, device
    )
    
    # Imprimir métricas de prueba
    metrics_calculator = SingleLabelMetrics(class_names)
    metrics_calculator.print_metrics(test_metrics)
    
    # Guardar configuración y métricas
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'freeze_backbone': args.freeze_backbone,
        'fine_tune_epochs': args.fine_tune_epochs,
        'loss_type': args.loss_type,
        'gamma': args.gamma if args.loss_type in ['focal', 'weighted_focal'] else None,
        'alpha': args.alpha if args.loss_type in ['focal', 'weighted_focal'] else None,
        'device': str(device),
        'class_names': class_names,
        'dataset_stats': dataset_stats,
        'test_metrics': test_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Entrenamiento completado!")
    print(f"📁 Resultados guardados en: {args.output_dir}")
    print(f"📄 Configuración guardada en: {config_path}")

if __name__ == "__main__":
    main()

