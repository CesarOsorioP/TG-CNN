"""
Script de predicción para el modelo DenseNet entrenado.
Permite clasificar imágenes individuales o lotes de imágenes.

"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class DenseNetClassifier(nn.Module):
    """
    Clasificador DenseNet con transfer learning y freeze del backbone.
    (Misma arquitectura que en train_model.py)
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
    
    def forward(self, x):
        return self.backbone(x)

class ChestXrayPredictor:
    """
    Clase para realizar predicciones con el modelo entrenado.
    """
    
    def __init__(self, model_path, device=None):
        """
        Inicializar el predictor.
        
        Args:
            model_path: Ruta al archivo del modelo guardado (.pth)
            device: Dispositivo a usar ('cuda', 'cpu', o None para auto-detectar)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.transform = None
        
        # Cargar modelo
        self._load_model(model_path)
        
        # Configurar transformaciones
        self._setup_transforms()
    
    def _load_model(self, model_path):
        """Cargar el modelo entrenado."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        print(f"Cargando modelo desde: {model_path}")
        
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Crear modelo
        self.model = DenseNetClassifier(num_classes=2, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Cargar mapeo de clases
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = checkpoint['idx_to_class']
        
        print(f"Modelo cargado exitosamente en dispositivo: {self.device}")
        print(f"Clases disponibles: {list(self.idx_to_class.values())}")
    
    def _setup_transforms(self):
        """Configurar transformaciones para las imágenes."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_single_image(self, image_path, return_probabilities=False):
        """
        Predecir una sola imagen.
        
        Args:
            image_path: Ruta a la imagen
            return_probabilities: Si devolver probabilidades además de la clase
            
        Returns:
            dict: Resultado de la predicción
        """
        # Cargar y preprocesar imagen
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            return {"error": f"Error al cargar la imagen: {str(e)}"}
        
        # Realizar predicción
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        # Preparar resultado
        predicted_class = self.idx_to_class[predicted_class_idx]
        result = {
            "image_path": image_path,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_chest_xray": predicted_class == "chest_xray"
        }
        
        if return_probabilities:
            result["probabilities"] = {
                self.idx_to_class[i]: probabilities[0][i].item() 
                for i in range(len(self.idx_to_class))
            }
        
        return result
    
    def predict_batch(self, image_paths, return_probabilities=False):
        """
        Predecir múltiples imágenes.
        
        Args:
            image_paths: Lista de rutas a las imágenes
            return_probabilities: Si devolver probabilidades además de las clases
            
        Returns:
            list: Lista de resultados de predicción
        """
        results = []
        
        for image_path in image_paths:
            result = self.predict_single_image(image_path, return_probabilities)
            results.append(result)
        
        return results
    
    def predict_directory(self, directory_path, extensions=('.jpg', '.jpeg', '.png'), 
                         return_probabilities=False):
        """
        Predecir todas las imágenes en un directorio.
        
        Args:
            directory_path: Ruta al directorio
            extensions: Extensiones de archivo a procesar
            return_probabilities: Si devolver probabilidades además de las clases
            
        Returns:
            list: Lista de resultados de predicción
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directorio no encontrado: {directory_path}")
        
        # Encontrar todas las imágenes
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(directory_path).glob(f"**/*{ext}"))
            image_paths.extend(Path(directory_path).glob(f"**/*{ext.upper()}"))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            print(f"No se encontraron imágenes con extensiones {extensions} en {directory_path}")
            return []
        
        print(f"Procesando {len(image_paths)} imágenes...")
        return self.predict_batch(image_paths, return_probabilities)
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualizar una predicción con la imagen y el resultado.
        
        Args:
            image_path: Ruta a la imagen
            save_path: Ruta para guardar la visualización (opcional)
        """
        result = self.predict_single_image(image_path, return_probabilities=True)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        # Cargar imagen original
        image = Image.open(image_path).convert('RGB')
        
        # Crear visualización
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mostrar imagen
        ax1.imshow(image)
        ax1.set_title(f"Imagen: {os.path.basename(image_path)}")
        ax1.axis('off')
        
        # Mostrar predicción
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        is_chest_xray = result['is_chest_xray']
        
        # Color basado en la predicción
        color = 'green' if is_chest_xray else 'red'
        
        ax2.text(0.1, 0.8, f"Predicción: {predicted_class}", 
                fontsize=16, fontweight='bold', color=color)
        ax2.text(0.1, 0.6, f"Confianza: {confidence:.3f}", 
                fontsize=14)
        ax2.text(0.1, 0.4, f"Es radiografía de tórax: {'Sí' if is_chest_xray else 'No'}", 
                fontsize=14, color=color, fontweight='bold')
        
        # Mostrar probabilidades
        if "probabilities" in result:
            ax2.text(0.1, 0.2, "Probabilidades:", fontsize=12, fontweight='bold')
            y_pos = 0.1
            for class_name, prob in result["probabilities"].items():
                ax2.text(0.1, y_pos, f"  {class_name}: {prob:.3f}", fontsize=10)
                y_pos -= 0.05
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualización guardada en: {save_path}")
        
        plt.show()
        
        return result

def main():
    """Función principal para ejecutar predicciones desde línea de comandos."""
    
    parser = argparse.ArgumentParser(description='Clasificar radiografías de tórax con DenseNet')
    parser.add_argument('--model', type=str, default='densenet_chest_xray_model.pth',
                       help='Ruta al modelo entrenado')
    parser.add_argument('--image', type=str, help='Ruta a una imagen individual')
    parser.add_argument('--directory', type=str, help='Directorio con imágenes para procesar')
    parser.add_argument('--output', type=str, help='Archivo de salida para guardar resultados')
    parser.add_argument('--visualize', action='store_true', 
                       help='Mostrar visualización de la predicción')
    parser.add_argument('--probabilities', action='store_true',
                       help='Incluir probabilidades en los resultados')
    
    args = parser.parse_args()
    
    # Verificar que se proporcionó al menos una opción
    if not args.image and not args.directory:
        print("Error: Debes proporcionar --image o --directory")
        parser.print_help()
        return
    
    # Inicializar predictor
    try:
        predictor = ChestXrayPredictor(args.model)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return
    
    results = []
    
    # Procesar imagen individual
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Imagen no encontrada: {args.image}")
            return
        
        print(f"Procesando imagen: {args.image}")
        result = predictor.predict_single_image(args.image, args.probabilities)
        results.append(result)
        
        # Mostrar resultado
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Predicción: {result['predicted_class']}")
            print(f"Confianza: {result['confidence']:.3f}")
            print(f"Es radiografía de tórax: {'Sí' if result['is_chest_xray'] else 'No'}")
            
            if args.probabilities and "probabilities" in result:
                print("Probabilidades:")
                for class_name, prob in result["probabilities"].items():
                    print(f"  {class_name}: {prob:.3f}")
        
        # Visualizar si se solicita
        if args.visualize:
            predictor.visualize_prediction(args.image)
    
    # Procesar directorio
    if args.directory:
        print(f"Procesando directorio: {args.directory}")
        results = predictor.predict_directory(args.directory, 
                                            return_probabilities=args.probabilities)
        
        if not results:
            return
        
        # Mostrar resumen
        chest_xray_count = sum(1 for r in results if r.get('is_chest_xray', False))
        other_count = len(results) - chest_xray_count
        
        print(f"\nResumen de predicciones:")
        print(f"Total de imágenes procesadas: {len(results)}")
        print(f"Radiografías de tórax detectadas: {chest_xray_count}")
        print(f"Otras imágenes: {other_count}")
        print(f"Porcentaje de radiografías: {chest_xray_count/len(results)*100:.1f}%")
    
    # Guardar resultados si se especifica archivo de salida
    if args.output and results:
        output_data = {
            'model_path': args.model,
            'timestamp': str(torch.datetime.now()),
            'results': results
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Resultados guardados en: {args.output}")

if __name__ == "__main__":
    main()
