"""
Script de predicción para modelo single-label de clasificación de enfermedades.
Permite clasificar imágenes individuales o lotes con una sola enfermedad por imagen.

"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Importar módulos del proyecto
try:
    from .model import DenseNetSingleLabelClassifier
except ImportError:
    from model import DenseNetSingleLabelClassifier

class SingleLabelDiseasePredictor:
    """
    Clase para realizar predicciones con el modelo single-label entrenado.
    """
    
    def __init__(self, model_path, device=None):
        """
        Inicializar el predictor single-label.
        
        Args:
            model_path: Ruta al archivo del modelo guardado (.pth)
            device: Dispositivo a usar ('cuda', 'cpu', o None para auto-detectar)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = None
        self.num_classes = None
        self.transform = None
        
        # Cargar modelo
        self._load_model(model_path)
        
        # Configurar transformaciones
        self._setup_transforms()
    
    def _load_model(self, model_path):
        """Cargar el modelo single-label entrenado."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        print(f"🔄 Cargando modelo single-label desde: {model_path}")
        
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extraer información del modelo
        self.class_names = checkpoint['class_names']
        self.num_classes = checkpoint['num_classes']
        
        # Crear modelo
        self.model = DenseNetSingleLabelClassifier(
            num_classes=self.num_classes,
            pretrained=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Modelo cargado exitosamente en dispositivo: {self.device}")
        print(f"📊 Clases disponibles: {self.class_names}")
    
    def _setup_transforms(self):
        """Configurar transformaciones para las imágenes."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_single_image(self, image_path, return_probabilities=True):
        """
        Predecir clase en una sola imagen.
        
        Args:
            image_path: Ruta a la imagen
            return_probabilities: Si devolver probabilidades detalladas
            
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
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        
        # Convertir probabilidades a numpy
        probabilities = probabilities.cpu().numpy()[0]
        confidence = float(probabilities[predicted_class_idx])
        predicted_class = self.class_names[predicted_class_idx]
        
        # Preparar resultado
        result = {
            "image_path": image_path,
            "predicted_class": predicted_class,
            "predicted_class_idx": int(predicted_class_idx),
            "confidence": confidence,
            "is_normal": predicted_class == "Normal",
            "summary": f"Clase detectada: {predicted_class} ({confidence:.1%})"
        }
        
        if return_probabilities:
            result["all_probabilities"] = {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, probabilities)
            }
        
        return result
    
    def predict_batch(self, image_paths, return_probabilities=True):
        """
        Predecir clases en múltiples imágenes.
        
        Args:
            image_paths: Lista de rutas a las imágenes
            return_probabilities: Si devolver probabilidades detalladas
            
        Returns:
            list: Lista de resultados de predicción
        """
        results = []
        
        print(f"🔄 Procesando {len(image_paths)} imágenes...")
        
        for i, image_path in enumerate(image_paths):
            if i % 100 == 0:
                print(f"  Procesadas: {i}/{len(image_paths)}")
            
            result = self.predict_single_image(image_path, return_probabilities)
            results.append(result)
        
        return results
    
    def predict_directory(self, directory_path, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'),
                         return_probabilities=True):
        """
        Predecir todas las imágenes en un directorio.
        
        Args:
            directory_path: Ruta al directorio
            extensions: Extensiones de archivo a procesar
            return_probabilities: Si devolver probabilidades detalladas
            
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
            print(f"❌ No se encontraron imágenes con extensiones {extensions} en {directory_path}")
            return []
        
        print(f"📁 Encontradas {len(image_paths)} imágenes en {directory_path}")
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
            print(f"❌ Error: {result['error']}")
            return result
        
        # Cargar imagen original
        image = Image.open(image_path).convert('RGB')
        
        # Crear visualización
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Mostrar imagen
        ax1.imshow(image)
        ax1.set_title(f"Imagen: {os.path.basename(image_path)}", fontsize=14)
        ax1.axis('off')
        
        # Mostrar predicción
        ax2.set_title("Resultado del Análisis", fontsize=16, fontweight='bold')
        
        color = 'green' if result['is_normal'] else 'red'
        ax2.text(0.1, 0.9, f"CLASE DETECTADA: {result['predicted_class']}", 
                fontsize=18, fontweight='bold', color=color)
        ax2.text(0.1, 0.7, f"Confianza: {result['confidence']:.1%}", 
                fontsize=14, color=color)
        
        # Mostrar todas las probabilidades
        if 'all_probabilities' in result:
            ax2.text(0.1, 0.5, "Probabilidades Detalladas:", 
                    fontsize=12, fontweight='bold')
            y_pos = 0.45
            
            # Ordenar por probabilidad
            sorted_probs = sorted(result['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)
            
            for class_name, prob in sorted_probs:
                color_prob = 'red' if class_name == result['predicted_class'] else 'gray'
                ax2.text(0.1, y_pos, f"{class_name}: {prob:.1%}", 
                        fontsize=10, color=color_prob)
                y_pos -= 0.03
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualización guardada en: {save_path}")
        
        plt.show()
        
        return result
    
    def analyze_batch_results(self, results):
        """
        Analizar resultados de un lote de predicciones.
        
        Args:
            results: Lista de resultados de predicción
            
        Returns:
            dict: Análisis de los resultados
        """
        total_images = len(results)
        successful_predictions = [r for r in results if 'error' not in r]
        failed_predictions = [r for r in results if 'error' in r]
        
        # Contar por clase
        class_counts = {}
        for class_name in self.class_names:
            class_counts[class_name] = sum(1 for r in successful_predictions 
                                         if r['predicted_class'] == class_name)
        
        # Estadísticas de confianza
        confidences = [r['confidence'] for r in successful_predictions]
        avg_confidence = np.mean(confidences) if confidences else 0
        high_confidence = sum(1 for c in confidences if c > 0.8)
        medium_confidence = sum(1 for c in confidences if 0.6 <= c <= 0.8)
        low_confidence = sum(1 for c in confidences if c < 0.6)
        
        return {
            'total_images': total_images,
            'successful_predictions': len(successful_predictions),
            'failed_predictions': len(failed_predictions),
            'class_counts': class_counts,
            'avg_confidence': avg_confidence,
            'high_confidence_count': high_confidence,
            'medium_confidence_count': medium_confidence,
            'low_confidence_count': low_confidence
        }
    
    def print_batch_analysis(self, analysis):
        """Imprimir análisis de resultados en lote."""
        print("\n" + "="*70)
        print("📊 ANÁLISIS DE RESULTADOS SINGLE-LABEL")
        print("="*70)
        print(f"Total de imágenes procesadas: {analysis['total_images']:,}")
        print(f"Predicciones exitosas: {analysis['successful_predictions']:,}")
        print(f"Predicciones fallidas: {analysis['failed_predictions']:,}")
        
        print(f"\n🔍 DISTRIBUCIÓN POR CLASE:")
        for class_name, count in analysis['class_counts'].items():
            percentage = (count / analysis['successful_predictions'] * 100) if analysis['successful_predictions'] > 0 else 0
            print(f"  {class_name}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n📈 ESTADÍSTICAS DE CONFIANZA:")
        print(f"  Confianza promedio: {analysis['avg_confidence']:.3f}")
        print(f"  Alta confianza (>80%): {analysis['high_confidence_count']:,}")
        print(f"  Confianza media (60-80%): {analysis['medium_confidence_count']:,}")
        print(f"  Baja confianza (<60%): {analysis['low_confidence_count']:,}")

def main():
    """Función principal para ejecutar predicciones desde línea de comandos."""
    
    parser = argparse.ArgumentParser(description='Clasificar enfermedades en radiografías con modelo single-label')
    parser.add_argument('--model', type=str, default='results/models/singlelabel/densenet_singlelabel_model.pth',
                       help='Ruta al modelo entrenado')
    parser.add_argument('--image', type=str, help='Ruta a una imagen individual')
    parser.add_argument('--directory', type=str, help='Directorio con imágenes para procesar')
    parser.add_argument('--output', type=str, help='Archivo de salida para guardar resultados')
    parser.add_argument('--visualize', action='store_true', 
                       help='Mostrar visualización de la predicción')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                       default='auto', help='Dispositivo a usar')
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Verificar que se proporcionó al menos una opción
    if not args.image and not args.directory:
        print("❌ Error: Debes proporcionar --image o --directory")
        parser.print_help()
        return
    
    # Inicializar predictor
    try:
        predictor = SingleLabelDiseasePredictor(args.model, device=device)
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return
    
    results = []
    
    # Procesar imagen individual
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Error: Imagen no encontrada: {args.image}")
            return
        
        print(f"🔄 Procesando imagen: {args.image}")
        result = predictor.predict_single_image(args.image, return_probabilities=True)
        results.append(result)
        
        # Mostrar resultado
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"\n📊 RESULTADO:")
            print(f"  {result['summary']}")
            print(f"  Clase: {result['predicted_class']}")
            print(f"  Confianza: {result['confidence']:.1%}")
        
        # Visualizar si se solicita
        if args.visualize:
            predictor.visualize_prediction(args.image)
    
    # Procesar directorio
    if args.directory:
        print(f"🔄 Procesando directorio: {args.directory}")
        results = predictor.predict_directory(args.directory, return_probabilities=True)
        
        if not results:
            return
        
        # Analizar resultados
        analysis = predictor.analyze_batch_results(results)
        predictor.print_batch_analysis(analysis)
    
    # Guardar resultados si se especifica archivo de salida
    if args.output and results:
        output_data = {
            'model_path': args.model,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Resultados guardados en: {args.output}")

if __name__ == "__main__":
    main()

