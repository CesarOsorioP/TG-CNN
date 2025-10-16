"""
Script de entrenamiento para modelo multi-label de clasificación de enfermedades.
Basado en el script de entrenamiento binario pero adaptado para multi-label.

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

# Importar módulos del proyecto
from dataset import create_data_loaders
from model import create_model, create_loss_function
from metrics import MultiLabelMetrics, evaluate_model_multi_label

def get_transforms():
    """
    Obtener transformaciones de datos para entrenamiento y validación.
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
    
    train_pbar = tqdm(train_loader, desc=f"Época {epoch+1} - Entrenando")
    
    for batch_idx, (data, target) in enumerate(train_pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Guardar predicciones para métricas
        predictions = (output > 0.5).float()
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Calcular métricas de entrenamiento
    train_loss /= len(train_loader)
    
    # Calcular F1-Score
    from sklearn.metrics import f1_score
    f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    f1_micro = f1_score(all_targets, all_predictions, average='micro', zero_division=0)
    
    return {
        'loss': train_loss,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro
    }

def validate_epoch(model, val_loader, criterion, device, disease_names, epoch):
    """
    Validar una época del modelo.
    
    Args:
        model: Modelo a validar
        val_loader: DataLoader de validación
        criterion: Función de pérdida
        device: Dispositivo a usar
        disease_names: Lista de nombres de enfermedades
        epoch: Número de época
        
    Returns:
        dict: Métricas de validación
    """
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    val_pbar = tqdm(val_loader, desc=f"Época {epoch+1} - Validando")
    
    with torch.no_grad():
        for data, target in val_pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            val_loss += loss.item()
            
            # Guardar predicciones para métricas
            predictions = (output > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(output.cpu().numpy())
            
            val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Calcular métricas de validación
    val_loss /= len(val_loader)
    
    # Calcular métricas multi-label
    metrics_calculator = MultiLabelMetrics(disease_names)
    metrics = metrics_calculator.calculate_metrics(
        np.array(all_targets), 
        np.array(all_predictions), 
        np.array(all_probabilities)
    )
    
    return {
        'loss': val_loss,
        'f1_macro': metrics['f1_macro'],
        'f1_micro': metrics['f1_micro'],
        'hamming_loss': metrics['hamming_loss'],
        'exact_match': metrics['exact_match'],
        'jaccard_macro': metrics['jaccard_macro']
    }

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, 
                disease_names, device, loss_type='bce', class_weights=None):
    """
    Entrenar el modelo multi-label.
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        num_epochs: Número de épocas
        learning_rate: Tasa de aprendizaje
        disease_names: Lista de nombres de enfermedades
        device: Dispositivo a usar
        loss_type: Tipo de función de pérdida
        class_weights: Pesos de clase (opcional)
        
    Returns:
        dict: Historial de entrenamiento
    """
    # Configurar función de pérdida
    criterion = create_loss_function(loss_type, class_weights)
    criterion = criterion.to(device)
    
    # Configurar optimizador
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Historial de entrenamiento
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1_macro': [],
        'val_f1_macro': [],
        'train_f1_micro': [],
        'val_f1_micro': [],
        'val_hamming_loss': [],
        'val_exact_match': [],
        'val_jaccard_macro': []
    }
    
    best_val_f1 = 0.0
    best_model_state = None
    patience = 3
    patience_counter = 0
    
    print(f"🚀 Iniciando entrenamiento multi-label...")
    print(f"📊 Épocas: {num_epochs}")
    print(f"📈 Learning rate: {learning_rate}")
    print(f"💻 Dispositivo: {device}")
    print(f"🎯 Función de pérdida: {loss_type}")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"ÉPOCA {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Entrenar
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validar
        val_metrics = validate_epoch(model, val_loader, criterion, device, disease_names, epoch)
        
        # Actualizar historial
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_f1_macro'].append(train_metrics['f1_macro'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['train_f1_micro'].append(train_metrics['f1_micro'])
        history['val_f1_micro'].append(val_metrics['f1_micro'])
        history['val_hamming_loss'].append(val_metrics['hamming_loss'])
        history['val_exact_match'].append(val_metrics['exact_match'])
        history['val_jaccard_macro'].append(val_metrics['jaccard_macro'])
        
        # Imprimir métricas
        print(f"\n📊 MÉTRICAS DE LA ÉPOCA:")
        print(f"  Entrenamiento - Loss: {train_metrics['loss']:.4f}, F1-Macro: {train_metrics['f1_macro']:.4f}")
        print(f"  Validación - Loss: {val_metrics['loss']:.4f}, F1-Macro: {val_metrics['f1_macro']:.4f}")
        print(f"  Hamming Loss: {val_metrics['hamming_loss']:.4f}")
        print(f"  Exact Match: {val_metrics['exact_match']:.4f}")
        
        # Guardar mejor modelo
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✅ ¡Nuevo mejor modelo! F1-Macro: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  ⏳ Sin mejora ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping activado después de {patience} épocas sin mejora")
            break
        
        scheduler.step()
    
    # Cargar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n🏆 Mejor modelo cargado - F1-Macro: {best_val_f1:.4f}")
    
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
    
    # F1-Score Macro
    axes[0, 1].plot(epochs, history['train_f1_macro'], 'b-', label='Entrenamiento', linewidth=2)
    axes[0, 1].plot(epochs, history['val_f1_macro'], 'r-', label='Validación', linewidth=2)
    axes[0, 1].set_title('F1-Score (Macro) durante el entrenamiento')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('F1-Score (Macro)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1-Score Micro
    axes[0, 2].plot(epochs, history['train_f1_micro'], 'b-', label='Entrenamiento', linewidth=2)
    axes[0, 2].plot(epochs, history['val_f1_micro'], 'r-', label='Validación', linewidth=2)
    axes[0, 2].set_title('F1-Score (Micro) durante el entrenamiento')
    axes[0, 2].set_xlabel('Época')
    axes[0, 2].set_ylabel('F1-Score (Micro)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Hamming Loss
    axes[1, 0].plot(epochs, history['val_hamming_loss'], 'g-', label='Validación', linewidth=2)
    axes[1, 0].set_title('Hamming Loss durante el entrenamiento')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Hamming Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Exact Match
    axes[1, 1].plot(epochs, history['val_exact_match'], 'm-', label='Validación', linewidth=2)
    axes[1, 1].set_title('Exact Match Ratio durante el entrenamiento')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Exact Match Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Jaccard Score
    axes[1, 2].plot(epochs, history['val_jaccard_macro'], 'c-', label='Validación', linewidth=2)
    axes[1, 2].set_title('Jaccard Score (Macro) durante el entrenamiento')
    axes[1, 2].set_xlabel('Época')
    axes[1, 2].set_ylabel('Jaccard Score (Macro)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Gráfico de entrenamiento guardado en: {save_path}")
    
    plt.show()

def main():
    """
    Función principal para entrenar el modelo multi-label.
    """
    parser = argparse.ArgumentParser(description='Entrenar modelo multi-label de enfermedades')
    parser.add_argument('--data_dir', type=str, default='data_diseases',
                       help='Directorio con datos multi-label')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Tamaño del lote')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Número de épocas')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Tasa de aprendizaje')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                       help='Congelar backbone durante entrenamiento')
    parser.add_argument('--fine_tune_epochs', type=int, default=5,
                       help='Épocas de fine-tuning')
    parser.add_argument('--loss_type', type=str, default='bce',
                       choices=['bce', 'focal', 'weighted_bce'],
                       help='Tipo de función de pérdida')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Dispositivo a usar')
    parser.add_argument('--output_dir', type=str, default='results/models/multilabel',
                       help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🚀 ENTRENAMIENTO DE MODELO MULTI-LABEL")
    print("="*60)
    print(f"📁 Directorio de datos: {args.data_dir}")
    print(f"📊 Tamaño de lote: {args.batch_size}")
    print(f"🔄 Épocas: {args.num_epochs}")
    print(f"📈 Learning rate: {args.learning_rate}")
    print(f"🔒 Backbone congelado: {args.freeze_backbone}")
    print(f"💻 Dispositivo: {device}")
    print(f"🎯 Función de pérdida: {args.loss_type}")
    
    # Verificar directorio de datos
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Directorio {args.data_dir} no encontrado")
        print("💡 Ejecuta primero: python src/models/multilabel/prepare_data.py")
        return
    
    # Obtener transformaciones
    train_transform, val_transform = get_transforms()
    
    # Crear DataLoaders
    print(f"\n📊 Preparando datos...")
    train_loader, val_loader, test_loader, dataset_stats = create_data_loaders(
        data_dir=args.data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=args.batch_size,
        include_normal=True,
        num_workers=0
    )
    
    # Definir nombres de enfermedades (incluyendo Normal)
    disease_names = [
        'Neumonía', 'Cáncer', 'Atelectasia', 
        'Edema', 'Tuberculosis', 'COVID-19', 'Normal'
    ]
    
    # Crear modelo
    print(f"\n🤖 Creando modelo...")
    model = create_model(
        num_diseases=len(disease_names),
        pretrained=True,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    
    # Imprimir información del modelo
    model.print_model_info()
    
    # Calcular pesos de clase si es necesario
    class_weights = None
    if args.loss_type == 'weighted_bce':
        # Crear dataset temporal para calcular pesos
        from dataset import MultiLabelChestXrayDataset
        temp_dataset = MultiLabelChestXrayDataset(args.data_dir, transform=None)
        class_weights = temp_dataset.get_class_weights()
        print(f"📊 Pesos de clase calculados: {class_weights}")
    
    # Entrenar modelo
    print(f"\n🏋️ Iniciando entrenamiento...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        disease_names=disease_names,
        device=device,
        loss_type=args.loss_type,
        class_weights=class_weights
    )
    
    # Guardar checkpoint PRE fine-tuning para no perder el mejor modelo de la primera fase
    os.makedirs(args.output_dir, exist_ok=True)
    pre_ft_path = os.path.join(args.output_dir, 'densenet_multilabel_pre_ft.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'disease_names': disease_names,
        'num_diseases': len(disease_names),
        'history': history,
        'config': vars(args)
    }, pre_ft_path)
    print(f"\n💾 Checkpoint pre fine-tuning guardado en: {pre_ft_path}")
    
    # Fine-tuning (opcional) con manejo de errores para conservar el pre-FT
    if args.fine_tune_epochs > 0 and args.freeze_backbone:
        try:
            print(f"\n🔓 Iniciando fine-tuning...")
            model.unfreeze_backbone()
            
            # Reducir learning rate para fine-tuning
            fine_tune_lr = args.learning_rate * 0.1
            print(f"📈 Learning rate reducido a: {fine_tune_lr}")
            
            # Re-crear DataLoaders para fine-tuning con batch_size fijo de 8
            print(f"🔁 Re-creando DataLoaders para fine-tuning con batch_size=8 ...")
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
                disease_names=disease_names,
                device=device,
                loss_type=args.loss_type,
                class_weights=class_weights
            )
            
            # Combinar historiales
            for key in history:
                history[key].extend(fine_tune_history[key])
            
            # Guardar checkpoint POST fine-tuning adicional
            post_ft_path = os.path.join(args.output_dir, 'densenet_multilabel_post_ft.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'disease_names': disease_names,
                'num_diseases': len(disease_names),
                'history': history,
                'config': vars(args)
            }, post_ft_path)
            print(f"💾 Checkpoint post fine-tuning guardado en: {post_ft_path}")
        except Exception as e:
            print(f"\n⚠️ Fine-tuning detenido por error: {e}")
            print(f"Se conserva el checkpoint pre fine-tuning en: {pre_ft_path}")
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Guardar modelo
    model_path = os.path.join(args.output_dir, 'densenet_multilabel_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'disease_names': disease_names,
        'num_diseases': len(disease_names),
        'history': history,
        'config': vars(args)
    }, model_path)
    print(f"\n💾 Modelo guardado en: {model_path}")
    
    # Visualizar historial de entrenamiento
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Evaluar en conjunto de prueba
    print(f"\n📊 Evaluando en conjunto de prueba...")
    test_metrics, test_predictions, test_targets, test_probabilities = evaluate_model_multi_label(
        model, test_loader, disease_names, device
    )
    
    # Imprimir métricas de prueba
    metrics_calculator = MultiLabelMetrics(disease_names)
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
        'device': str(device),
        'disease_names': disease_names,
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
