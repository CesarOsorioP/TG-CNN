"""
Dataset personalizado para clasificación multi-label de enfermedades en radiografías de tórax.
Basado en la estructura del dataset binario existente pero adaptado para multi-label.

"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from collections import Counter

class MultiLabelChestXrayDataset(Dataset):
    """
    Dataset para clasificación multi-label de enfermedades en radiografías de tórax.
    
    Estructura esperada:
    data_diseases/
    ├── Neumonía/        # Imágenes con neumonía
    ├── Cáncer/          # Imágenes con cáncer
    ├── Atelectasia/     # Imágenes con atelectasia
    ├── Edema/           # Imágenes con edema
    ├── Tuberculosis/    # Imágenes con tuberculosis
    └── COVID-19/        # Imágenes con COVID-19
    """
    
    def __init__(self, data_dir, transform=None, include_normal=True, split='train'):
        """
        Inicializar dataset multi-label.
        
        Args:
            data_dir: Directorio con carpetas de enfermedades
            transform: Transformaciones a aplicar a las imágenes
            include_normal: Si incluir clase "Normal" (default: True)
            split: División de datos a usar ('train', 'val', 'test')
        """
        self.data_dir = data_dir
        self.transform = transform
        self.include_normal = include_normal
        self.split = split
        
        # Definir enfermedades (incluyendo Normal)
        self.disease_names = [
            'Neumonía',
            'Cáncer', 
            'Atelectasia',
            'Edema',
            'Tuberculosis',
            'COVID-19',
            'Normal'
        ]
        
        if include_normal:
            self.disease_names.append('Normal')
        
        # Crear mapeo de enfermedades
        self.disease_to_idx = {disease: i for i, disease in enumerate(self.disease_names)}
        self.idx_to_disease = {i: disease for i, disease in enumerate(self.disease_names)}
        
        # Cargar datos
        self.images = []
        self.labels = []
        self._load_data()
    
    def _load_data(self):
        """
        Cargar todas las imágenes y crear etiquetas multi-label desde la estructura train/val/test.
        """
        print(f"🔄 Cargando datos multi-label para split '{self.split}'...")
        
        for disease in self.disease_names:
            # Usar la nueva estructura con subcarpetas
            disease_dir = os.path.join(self.data_dir, disease, self.split)
            
            if not os.path.exists(disease_dir):
                print(f"⚠️  Advertencia: Directorio {disease_dir} no encontrado")
                continue
            
            # Obtener archivos de imagen
            image_files = []
            for file in os.listdir(disease_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                    image_files.append(file)
            
            print(f"📁 {disease} ({self.split}): {len(image_files)} imágenes")
            
            # Crear etiquetas multi-label para cada imagen
            for img_file in image_files:
                img_path = os.path.join(disease_dir, img_file)
                self.images.append(img_path)
                
                # Crear etiqueta multi-label (one-hot para esta enfermedad)
                label = [0] * len(self.disease_names)
                label[self.disease_to_idx[disease]] = 1
                self.labels.append(label)
        
        print(f"✅ Total de imágenes cargadas: {len(self.images)}")
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """
        Imprimir distribución de clases.
        """
        print("\n📊 Distribución de clases:")
        for i, disease in enumerate(self.disease_names):
            count = sum(1 for label in self.labels if label[i] == 1)
            percentage = (count / len(self.labels)) * 100 if self.labels else 0
            print(f"  {disease}: {count:,} imágenes ({percentage:.1f}%)")
    
    def __len__(self):
        """Retornar número total de imágenes."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Obtener imagen y etiqueta en el índice dado.
        
        Args:
            idx: Índice de la imagen
            
        Returns:
            tuple: (imagen_tensor, etiqueta_tensor)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Cargar imagen
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ Error cargando imagen {img_path}: {e}")
            # Crear imagen en blanco como fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        # Convertir etiqueta a tensor
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return image, label_tensor
    
    def get_class_weights(self):
        """
        Calcular pesos de clase para balancear el dataset.
        
        Returns:
            torch.Tensor: Pesos de clase
        """
        # Contar ocurrencias de cada clase
        class_counts = [0] * len(self.disease_names)
        for label in self.labels:
            for i, is_present in enumerate(label):
                if is_present:
                    class_counts[i] += 1
        
        # Calcular pesos (inversamente proporcional a la frecuencia)
        total_samples = len(self.labels)
        weights = []
        
        for count in class_counts:
            if count > 0:
                weight = total_samples / (len(self.disease_names) * count)
                weights.append(weight)
            else:
                weights.append(0.0)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_disease_statistics(self):
        """
        Obtener estadísticas detalladas del dataset.
        
        Returns:
            dict: Estadísticas del dataset
        """
        stats = {
            'total_images': len(self.images),
            'num_diseases': len(self.disease_names),
            'disease_names': self.disease_names,
            'class_distribution': {},
            'multi_label_stats': {}
        }
        
        # Distribución por clase
        for i, disease in enumerate(self.disease_names):
            count = sum(1 for label in self.labels if label[i] == 1)
            stats['class_distribution'][disease] = {
                'count': count,
                'percentage': (count / len(self.labels)) * 100 if self.labels else 0
            }
        
        # Estadísticas multi-label
        labels_per_image = [sum(label) for label in self.labels]
        stats['multi_label_stats'] = {
            'avg_labels_per_image': np.mean(labels_per_image),
            'max_labels_per_image': max(labels_per_image),
            'min_labels_per_image': min(labels_per_image),
            'single_label_images': sum(1 for count in labels_per_image if count == 1),
            'multi_label_images': sum(1 for count in labels_per_image if count > 1),
            'normal_images': sum(1 for count in labels_per_image if count == 0) if self.include_normal else 0
        }
        
        return stats
    
    def print_statistics(self):
        """
        Imprimir estadísticas detalladas del dataset.
        """
        stats = self.get_disease_statistics()
        
        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS DEL DATASET MULTI-LABEL")
        print("="*60)
        print(f"Total de imágenes: {stats['total_images']:,}")
        print(f"Número de enfermedades: {stats['num_diseases']}")
        
        print(f"\n📈 Distribución por enfermedad:")
        for disease, info in stats['class_distribution'].items():
            print(f"  {disease}: {info['count']:,} ({info['percentage']:.1f}%)")
        
        print(f"\n🏷️  Estadísticas multi-label:")
        ml_stats = stats['multi_label_stats']
        print(f"  Promedio de etiquetas por imagen: {ml_stats['avg_labels_per_image']:.2f}")
        print(f"  Máximo de etiquetas por imagen: {ml_stats['max_labels_per_image']}")
        print(f"  Mínimo de etiquetas por imagen: {ml_stats['min_labels_per_image']}")
        print(f"  Imágenes con una etiqueta: {ml_stats['single_label_images']:,}")
        print(f"  Imágenes con múltiples etiquetas: {ml_stats['multi_label_images']:,}")
        if self.include_normal:
            print(f"  Imágenes normales: {ml_stats['normal_images']:,}")

def create_data_loaders(data_dir, train_transform, val_transform, 
                       batch_size=16, include_normal=True, num_workers=4):
    """
    Crear DataLoaders para entrenamiento, validación y prueba usando la estructura train/val/test.
    
    Args:
        data_dir: Directorio con datos multi-label (con subcarpetas train/val/test)
        train_transform: Transformaciones para entrenamiento
        val_transform: Transformaciones para validación
        batch_size: Tamaño del lote
        include_normal: Si incluir clase normal (default: True)
        num_workers: Número de workers para DataLoader
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, dataset_stats)
    """
    # Crear datasets para cada split
    train_dataset = MultiLabelChestXrayDataset(
        data_dir=data_dir, 
        transform=train_transform,
        include_normal=include_normal,
        split='train'
    )
    
    val_dataset = MultiLabelChestXrayDataset(
        data_dir=data_dir, 
        transform=val_transform,
        include_normal=include_normal,
        split='val'
    )
    
    test_dataset = MultiLabelChestXrayDataset(
        data_dir=data_dir, 
        transform=val_transform,
        include_normal=include_normal,
        split='test'
    )
    
    # Obtener estadísticas del dataset de entrenamiento
    dataset_stats = train_dataset.get_disease_statistics()
    
    # Crear DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n📊 División del dataset:")
    print(f"  Entrenamiento: {len(train_dataset):,} imágenes")
    print(f"  Validación: {len(val_dataset):,} imágenes")
    print(f"  Prueba: {len(test_dataset):,} imágenes")
    
    return train_loader, val_loader, test_loader, dataset_stats

if __name__ == "__main__":
    # Ejemplo de uso
    from torchvision import transforms
    
    # Transformaciones de ejemplo
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Crear dataset
    dataset = MultiLabelChestXrayDataset(
        data_dir="data_diseases",
        transform=transform,
        include_normal=False
    )
    
    # Imprimir estadísticas
    dataset.print_statistics()
    
    # Ejemplo de acceso a datos
    if len(dataset) > 0:
        image, label = dataset[0]
        print(f"\n🔍 Ejemplo de datos:")
        print(f"  Imagen shape: {image.shape}")
        print(f"  Etiqueta: {label}")
        print(f"  Enfermedades detectadas: {[dataset.idx_to_disease[i] for i, present in enumerate(label) if present]}")
