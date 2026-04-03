"""
Script de entrenamiento para clasificación de radiografías de tórax usando DenseNet
con transfer learning.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime

def get_device():
    """Detectar el mejor dispositivo disponible (CUDA, DirectML para AMD, o CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    
    # Intentar usar DirectML (Soporte nativo para AMD Radeon RX 580 en Windows)
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device('cpu')

class ChestXrayDataset(Dataset):
    """
    Dataset personalizado para radiografías de tórax.
    
    Estructura esperada:
    data/
    ├── chest_xray/     # Imágenes de radiografías de tórax
    └── other_images/   # Otras imágenes (no radiografías)
    """
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Mapeo de clases
        self.class_to_idx = {'chest_xray': 0, 'other_images': 1}
        self.idx_to_class = {0: 'chest_xray', 1: 'other_images'}
        
        # Cargar imágenes y etiquetas
        self._load_data()
    
    def _load_data(self):
        """Cargar todas las imágenes y sus etiquetas."""
        for class_name in self.class_to_idx.keys():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Advertencia: Directorio {class_dir} no encontrado")
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"Total de imágenes cargadas: {len(self.images)}")
        print(f"Radiografías de tórax: {self.labels.count(0)}")
        print(f"Otras imágenes: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Cargar imagen
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class DenseNetClassifier(nn.Module):
    """
    Clasificador DenseNet con transfer learning y freeze del backbone.
    """
    
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
        super(DenseNetClassifier, self).__init__()
        
        # Cargar DenseNet pre-entrenado
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Congelar parámetros del backbone para evitar overfitting
        if freeze_backbone:
            self._freeze_backbone()
        
        # Obtener número de características de la última capa
        num_features = self.backbone.classifier.in_features
        
        # Reemplazar la capa clasificadora
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def _freeze_backbone(self):
        """Congelar todos los parámetros del backbone (features)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone congelado - solo se entrenará el clasificador")
    
    def unfreeze_backbone(self):
        """Descongelar el backbone para fine-tuning opcional."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔓 Backbone descongelado - fine-tuning habilitado")
    
    def forward(self, x):
        return self.backbone(x)

def get_transforms():
    """Obtener transformaciones de datos para entrenamiento y validación."""
    
    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
    """
    Entrenar el modelo.
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader para datos de entrenamiento
        val_loader: DataLoader para datos de validación
        num_epochs: Número de épocas
        learning_rate: Tasa de aprendizaje
    """
    
    device = get_device()
    model = model.to(device)
    
    # Función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Historial de entrenamiento
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"Entrenando en dispositivo: {device}")
    
    # Mostrar información de parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Total de parámetros: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    print(f"Parámetros congelados: {frozen_params:,}")
    print(f"Porcentaje entrenable: {trainable_params/total_params*100:.2f}%")
    
    for epoch in range(num_epochs):
        print(f"\nÉpoca {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc="Entrenando")
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validando")
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Calcular métricas
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f"Pérdida de entrenamiento: {train_loss:.4f}")
        print(f"Precisión de entrenamiento: {train_acc:.2f}%")
        print(f"Pérdida de validación: {val_loss:.4f}")
        print(f"Precisión de validación: {val_acc:.2f}%")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"¡Nuevo mejor modelo! Precisión: {val_acc:.2f}%")
        
        scheduler.step()
    
    # Cargar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nMejor precisión de validación: {best_val_acc:.2f}%")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }

def plot_training_history(history, save_path=None):
    """Visualizar el historial de entrenamiento."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de pérdidas
    ax1.plot(history['train_losses'], label='Entrenamiento', color='blue')
    ax1.plot(history['val_losses'], label='Validación', color='red')
    ax1.set_title('Pérdida durante el entrenamiento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.legend()
    ax1.grid(True)
    
    # Gráfico de precisiones
    ax2.plot(history['train_accuracies'], label='Entrenamiento', color='blue')
    ax2.plot(history['val_accuracies'], label='Validación', color='red')
    ax2.set_title('Precisión durante el entrenamiento')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Precisión (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()

def evaluate_model(model, test_loader, class_names):
    """Evaluar el modelo en el conjunto de prueba."""
    
    device = get_device()
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluando"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Reporte de clasificación
    print("\n" + "="*50)
    print("REPORTE DE CLASIFICACIÓN")
    print("="*50)
    print(classification_report(all_targets, all_predictions, 
                              target_names=class_names))
    
    # Matriz de confusión
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.show()
    
    return all_predictions, all_targets

def organize_data_splits(data_dir, train_split=0.7, val_split=0.15, test_split=0.15):
    """Organiza automáticamente las carpetas en subdirectorios train, val, test."""
    import shutil
    import random
    
    classes = ['chest_xray', 'other_images']
    
    # Comprobar si ya están organizadas
    if os.path.exists(os.path.join(data_dir, 'train', classes[0])):
        return True
        
    print("📁 Organizando automáticamente las imágenes en directorios train/val/test...")
    
    # Verificar carpetas maestras
    for cls in classes:
        src = os.path.join(data_dir, cls)
        if not os.path.exists(src):
            print(f"❌ Error crítico: Falta la carpeta base {src}")
            return False
            
    # Crear estructura física
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(data_dir, split, cls), exist_ok=True)
            
    # Distribuir archivos de manera aleatoria pero reproducible
    for cls in classes:
        src = os.path.join(data_dir, cls)
        files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        random.seed(42)
        random.shuffle(files)
        
        n_total = len(files)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_f = files[:n_train]
        val_f = files[n_train:n_train+n_val]
        test_f = files[n_train+n_val:]
        
        print(f"  ➜ {cls} -> {len(train_f)} a train, {len(val_f)} a val, {len(test_f)} a test")
        
        for f in train_f: shutil.move(os.path.join(src, f), os.path.join(data_dir, 'train', cls, f))
        for f in val_f: shutil.move(os.path.join(src, f), os.path.join(data_dir, 'val', cls, f))
        for f in test_f: shutil.move(os.path.join(src, f), os.path.join(data_dir, 'test', cls, f))
        
        try:
            os.rmdir(src)
        except OSError:
            pass
            
    print("✅ Separación completada correctamente.")
    return True

def main():
    """Función principal para entrenar el modelo."""
    
    print("\n" + "="*50)
    print("🔍 VERIFICACIÓN DE HARDWARE")
    device = get_device()
    if device.type == 'cpu':
        print("⚠️ ATENCIÓN: Se usará el procesador (CPU) lo cual es más lento.")
        print("   Para activar la aceleración de tu AMD Radeon RX 580, necesitas")
        print("   utilizar Python 3.10 e instalar el motor 'torch-directml':")
        print("   ejecuta -> pip install torch-directml")
    else:
        print("✅ Aceleración por tarjeta gráfica activada exitosamente.")
    print("="*50 + "\n")
    
    # Configuración optimizada para datasets pequeños (~1200 imágenes)
    DATA_DIR = "data"  # Directorio con las imágenes
    BATCH_SIZE = 16    # Reducido para datasets pequeños
    NUM_EPOCHS = 5    # Aumentado para compensar el freeze
    LEARNING_RATE = 0.0001
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    FREEZE_BACKBONE = True  # Congelar backbone para evitar overfitting
    FINE_TUNE_EPOCHS = 3    # Épocas adicionales con fine-tuning (opcional)
    
    # Verificar que existe el directorio base de datos
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directorio {DATA_DIR} no encontrado.")
        print("Por favor, crea la siguiente estructura inicial:")
        print("data/")
        print("├── chest_xray/     # Imágenes de radiografías de tórax")
        print("└── other_images/   # Otras imágenes (no radiografías)")
        return
        
    # Organizar directorios si no se ha hecho
    if not organize_data_splits(DATA_DIR, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT):
        return
    
    # Obtener transformaciones
    train_transform, val_transform = get_transforms()
    
    # Cargar datasets en memoria usando las nuevas carpetas individuales
    train_dataset = ChestXrayDataset(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = ChestXrayDataset(os.path.join(DATA_DIR, 'val'), transform=val_transform)
    test_dataset = ChestXrayDataset(os.path.join(DATA_DIR, 'test'), transform=val_transform)
    
    if len(train_dataset) == 0:
        print("No se encontraron imágenes en el directorio de entrenamiento.")
        return
    
    # Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Tamaño del conjunto de entrenamiento: {len(train_dataset)}")
    print(f"Tamaño del conjunto de validación: {len(val_dataset)}")
    print(f"Tamaño del conjunto de prueba: {len(test_dataset)}")
    
    # Crear modelo con backbone congelado
    model = DenseNetClassifier(num_classes=2, pretrained=True, freeze_backbone=FREEZE_BACKBONE)
    
    # Entrenar modelo con backbone congelado
    print("\n🏋️  Iniciando entrenamiento con backbone congelado...")
    print("="*60)
    history = train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE)
    
    # Opcional: Fine-tuning con backbone descongelado
    if FINE_TUNE_EPOCHS > 0:
        print(f"\n🔓 Iniciando fine-tuning con backbone descongelado...")
        print("="*60)
        print("⚠️  Advertencia: Fine-tuning puede causar overfitting en datasets pequeños")
        
        # Descongelar backbone
        model.unfreeze_backbone()
        
        # Reducir learning rate para fine-tuning
        fine_tune_lr = LEARNING_RATE * 0.1
        print(f"Learning rate reducido a: {fine_tune_lr}")
        
        # Entrenar con fine-tuning
        fine_tune_history = train_model(model, train_loader, val_loader, FINE_TUNE_EPOCHS, fine_tune_lr)
        
        # Combinar historiales
        history['train_losses'].extend(fine_tune_history['train_losses'])
        history['val_losses'].extend(fine_tune_history['val_losses'])
        history['train_accuracies'].extend(fine_tune_history['train_accuracies'])
        history['val_accuracies'].extend(fine_tune_history['val_accuracies'])
        history['best_val_acc'] = max(history['best_val_acc'], fine_tune_history['best_val_acc'])
    
    # Guardar modelo
    model_path = "densenet_chest_xray_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx,
        'idx_to_class': train_dataset.idx_to_class,
        'history': history
    }, model_path)
    print(f"\nModelo guardado en: {model_path}")
    
    # Visualizar historial de entrenamiento
    plot_training_history(history, "training_history.png")
    
    # Evaluar en conjunto de prueba
    class_names = ['chest_xray', 'other_images']
    predictions, targets = evaluate_model(model, test_loader, class_names)
    
    # Guardar configuración de entrenamiento
    config = {
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'train_split': TRAIN_SPLIT,
        'val_split': VAL_SPLIT,
        'test_split': TEST_SPLIT,
        'freeze_backbone': FREEZE_BACKBONE,
        'fine_tune_epochs': FINE_TUNE_EPOCHS,
        'best_val_acc': history['best_val_acc'],
        'total_epochs': NUM_EPOCHS + FINE_TUNE_EPOCHS,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguración guardada en: training_config.json")
    print("\n¡Entrenamiento completado!")

if __name__ == "__main__":
    main()
