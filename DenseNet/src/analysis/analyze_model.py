"""
Script de análisis y visualización para el modelo DenseNet entrenado.
Incluye análisis de características, visualización de activaciones y métricas detalladas.

Autor: Asistente AI
Fecha: 2024
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import os
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
import cv2
from pathlib import Path

class DenseNetClassifier(nn.Module):
    """Clasificador DenseNet con freeze del backbone (misma arquitectura que en train_model.py)"""
    
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
        super(DenseNetClassifier, self).__init__()
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Congelar parámetros del backbone para evitar overfitting
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class ModelAnalyzer:
    """
    Clase para analizar y visualizar el modelo entrenado.
    """
    
    def __init__(self, model_path, data_dir=None):
        """
        Inicializar el analizador.
        
        Args:
            model_path: Ruta al modelo entrenado
            data_dir: Directorio con datos de prueba (opcional)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.data_dir = data_dir
        
        # Cargar modelo
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Cargar el modelo entrenado."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        print(f"Cargando modelo desde: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = DenseNetClassifier(num_classes=2, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = checkpoint['idx_to_class']
        
        print(f"Modelo cargado exitosamente en dispositivo: {self.device}")
    
    def analyze_model_architecture(self):
        """Analizar y visualizar la arquitectura del modelo."""
        print("\n" + "="*60)
        print("ANÁLISIS DE ARQUITECTURA DEL MODELO")
        print("="*60)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total de parámetros: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")
        print(f"Parámetros congelados: {total_params - trainable_params:,}")
        print(f"Porcentaje entrenable: {trainable_params/total_params*100:.2f}%")
        
        # Mostrar arquitectura
        print("\nArquitectura del modelo:")
        print("-" * 40)
        print(self.model)
        
        # Análisis de capas
        print("\nAnálisis por capas:")
        print("-" * 40)
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Solo hojas del árbol
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    print(f"{name}: {num_params:,} parámetros")
    
    def visualize_feature_maps(self, image_path, layer_name='backbone.features.denseblock4.denselayer16.conv2'):
        """
        Visualizar mapas de características de una capa específica.
        
        Args:
            image_path: Ruta a la imagen
            layer_name: Nombre de la capa a visualizar
        """
        if not os.path.exists(image_path):
            print(f"Imagen no encontrada: {image_path}")
            return
        
        # Cargar y preprocesar imagen
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Hook para capturar activaciones
        activations = {}
        def hook_fn(module, input, output):
            activations['output'] = output.detach().cpu()
        
        # Registrar hook
        for name, module in self.model.named_modules():
            if name == layer_name:
                hook = module.register_forward_hook(hook_fn)
                break
        else:
            print(f"Capa '{layer_name}' no encontrada")
            return
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Remover hook
        hook.remove()
        
        # Visualizar mapas de características
        feature_maps = activations['output'].squeeze(0)  # Remover batch dimension
        
        # Seleccionar algunos mapas para visualizar
        num_maps = min(16, feature_maps.shape[0])
        selected_maps = feature_maps[:num_maps]
        
        # Crear grid de visualización
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_maps):
            feature_map = selected_maps[i].numpy()
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'Canal {i}')
            axes[i].axis('off')
        
        # Ocultar ejes vacíos
        for i in range(num_maps, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Mapas de Características - {layer_name}', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        return feature_maps
    
    def generate_gradcam(self, image_path, target_class=None, layer_name='backbone.features.denseblock4.denselayer16.conv2'):
        """
        Generar mapa de activación Grad-CAM para una imagen.
        
        Args:
            image_path: Ruta a la imagen
            target_class: Clase objetivo (None para usar la predicción del modelo)
            layer_name: Nombre de la capa para Grad-CAM
        """
        if not os.path.exists(image_path):
            print(f"Imagen no encontrada: {image_path}")
            return
        
        # Cargar y preprocesar imagen
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_()
        
        # Hook para capturar activaciones
        activations = {}
        gradients = {}
        
        def forward_hook(module, input, output):
            activations['output'] = output
        
        def backward_hook(module, grad_input, grad_output):
            gradients['output'] = grad_output[0]
        
        # Registrar hooks
        for name, module in self.model.named_modules():
            if name == layer_name:
                forward_hook_handle = module.register_forward_hook(forward_hook)
                backward_hook_handle = module.register_backward_hook(backward_hook)
                break
        else:
            print(f"Capa '{layer_name}' no encontrada")
            return
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Remover hooks
        forward_hook_handle.remove()
        backward_hook_handle.remove()
        
        # Calcular Grad-CAM
        activations = activations['output'].squeeze(0)  # [C, H, W]
        gradients = gradients['output'].squeeze(0)     # [C, H, W]
        
        # Peso global promedio del gradiente
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Combinar mapas de características con pesos
        cam = torch.zeros(activations.shape[1:])  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Normalizar
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        # Convertir a numpy
        cam = cam.detach().cpu().numpy()
        
        # Visualizar
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Imagen original
        ax1.imshow(image)
        ax1.set_title('Imagen Original')
        ax1.axis('off')
        
        # Grad-CAM
        im2 = ax2.imshow(cam, cmap='jet', alpha=0.8)
        ax2.set_title('Grad-CAM')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)
        
        # Superposición
        ax3.imshow(image)
        ax3.imshow(cam, cmap='jet', alpha=0.4)
        ax3.set_title('Superposición')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return cam
    
    def analyze_predictions(self, test_loader):
        """
        Analizar predicciones del modelo en un conjunto de prueba.
        
        Args:
            test_loader: DataLoader con datos de prueba
        """
        print("\n" + "="*60)
        print("ANÁLISIS DE PREDICCIONES")
        print("="*60)
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Reporte de clasificación
        class_names = list(self.idx_to_class.values())
        print("\nReporte de Clasificación:")
        print("-" * 40)
        print(classification_report(all_targets, all_predictions, 
                                  target_names=class_names))
        
        # Matriz de confusión
        cm = confusion_matrix(all_targets, all_predictions)
        
        plt.figure(figsize=(10, 8))
        
        # Subplot 1: Matriz de confusión
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Matriz de Confusión')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        
        # Subplot 2: Distribución de confianza
        plt.subplot(2, 2, 2)
        max_probs = np.max(all_probabilities, axis=1)
        plt.hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribución de Confianza')
        plt.xlabel('Confianza Máxima')
        plt.ylabel('Frecuencia')
        
        # Subplot 3: ROC Curve
        plt.subplot(2, 2, 3)
        fpr, tpr, _ = roc_curve(all_targets, all_probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        
        # Subplot 4: Errores por clase
        plt.subplot(2, 2, 4)
        errors_by_class = []
        for i in range(len(class_names)):
            class_mask = all_targets == i
            class_errors = np.sum((all_predictions[class_mask] != all_targets[class_mask]))
            errors_by_class.append(class_errors)
        
        plt.bar(class_names, errors_by_class, alpha=0.7, edgecolor='black')
        plt.title('Errores por Clase')
        plt.ylabel('Número de Errores')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'confusion_matrix': cm,
            'roc_auc': roc_auc
        }
    
    def visualize_feature_embeddings(self, test_loader, n_samples=500):
        """
        Visualizar embeddings de características usando t-SNE.
        
        Args:
            test_loader: DataLoader con datos de prueba
            n_samples: Número máximo de muestras para t-SNE
        """
        print("\n" + "="*60)
        print("VISUALIZACIÓN DE EMBEDDINGS")
        print("="*60)
        
        # Hook para extraer características
        features = []
        labels = []
        
        def hook_fn(module, input, output):
            features.append(output.detach().cpu().numpy())
        
        # Registrar hook en la última capa antes del clasificador
        for name, module in self.model.named_modules():
            if name == 'backbone.classifier.0':  # Primera capa del clasificador
                hook = module.register_forward_hook(hook_fn)
                break
        
        self.model.eval()
        with torch.no_grad():
            count = 0
            for data, target in test_loader:
                if count >= n_samples:
                    break
                
                data = data.to(self.device)
                _ = self.model(data)
                
                labels.extend(target.numpy())
                count += len(target)
        
        hook.remove()
        
        # Concatenar características
        features = np.concatenate(features, axis=0)
        labels = np.array(labels)
        
        # Aplicar t-SNE
        print("Aplicando t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(features)
        
        # Visualizar
        plt.figure(figsize=(12, 8))
        class_names = list(self.idx_to_class.values())
        colors = ['red', 'blue']
        
        for i, class_name in enumerate(class_names):
            mask = labels == i
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=colors[i], label=class_name, alpha=0.6, s=20)
        
        plt.title('Visualización t-SNE de Características')
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return embeddings_2d, labels
    
    def generate_report(self, test_loader=None, save_path='model_analysis_report.json'):
        """
        Generar reporte completo del análisis del modelo.
        
        Args:
            test_loader: DataLoader con datos de prueba (opcional)
            save_path: Ruta para guardar el reporte
        """
        print("\n" + "="*60)
        print("GENERANDO REPORTE COMPLETO")
        print("="*60)
        
        report = {
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'device': str(self.device),
                'classes': self.idx_to_class
            }
        }
        
        # Análisis de arquitectura
        self.analyze_model_architecture()
        
        # Análisis de predicciones si hay datos de prueba
        if test_loader is not None:
            analysis_results = self.analyze_predictions(test_loader)
            report['performance'] = {
                'confusion_matrix': analysis_results['confusion_matrix'].tolist(),
                'roc_auc': float(analysis_results['roc_auc'])
            }
            
            # Visualización de embeddings
            try:
                embeddings, labels = self.visualize_feature_embeddings(test_loader)
                report['embeddings'] = {
                    'embeddings_2d': embeddings.tolist(),
                    'labels': labels.tolist()
                }
            except Exception as e:
                print(f"Error en visualización de embeddings: {e}")
        
        # Guardar reporte
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReporte guardado en: {save_path}")
        return report

def main():
    """Función principal para ejecutar análisis desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizar modelo DenseNet entrenado')
    parser.add_argument('--model', type=str, default='densenet_chest_xray_model.pth',
                       help='Ruta al modelo entrenado')
    parser.add_argument('--data_dir', type=str, help='Directorio con datos de prueba')
    parser.add_argument('--image', type=str, help='Imagen para análisis individual')
    parser.add_argument('--report', action='store_true', help='Generar reporte completo')
    
    args = parser.parse_args()
    
    # Inicializar analizador
    try:
        analyzer = ModelAnalyzer(args.model, args.data_dir)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return
    
    # Análisis de arquitectura
    analyzer.analyze_model_architecture()
    
    # Análisis de imagen individual
    if args.image:
        if os.path.exists(args.image):
            print(f"\nAnalizando imagen: {args.image}")
            analyzer.visualize_feature_maps(args.image)
            analyzer.generate_gradcam(args.image)
        else:
            print(f"Imagen no encontrada: {args.image}")
    
    # Generar reporte completo
    if args.report:
        analyzer.generate_report()

if __name__ == "__main__":
    main()
