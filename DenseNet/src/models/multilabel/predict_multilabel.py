"""
Script de predicción para modelo multi-label de clasificación de enfermedades.
Permite clasificar imágenes individuales o lotes con detección de múltiples enfermedades.

"""

import torch
import torch.nn as nn
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
from model import DenseNetMultiLabelClassifier

class MultiLabelDiseasePredictor:
    """
    Clase para realizar predicciones con el modelo multi-label entrenado.
    """
    
    def __init__(self, model_path, device=None, threshold=None, use_adaptive_thresholds=True):
        """
        Inicializar el predictor multi-label.
        
        Args:
            model_path: Ruta al archivo del modelo guardado (.pth)
            device: Dispositivo a usar ('cuda', 'cpu', o None para auto-detectar)
            threshold: Umbral para considerar enfermedad presente (si es None, usa umbrales adaptativos)
            use_adaptive_thresholds: Si usar umbrales adaptativos cargados del modelo
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.model = None
        self.disease_names = None
        self.num_diseases = None
        self.optimal_thresholds = None
        self.transform = None
        
        # Cargar modelo
        self._load_model(model_path)
        
        # Configurar transformaciones
        self._setup_transforms()
        
        # Mostrar configuración de umbrales
        if self.optimal_thresholds:
            print("\n📊 Umbrales adaptativos cargados:")
            for disease, threshold in self.optimal_thresholds.items():
                print(f"   {disease}: {threshold:.3f}")
        else:
            print(f"\n⚠️  Usando threshold fijo: {self.threshold}")
    
    def _load_model(self, model_path):
        """Cargar el modelo multi-label entrenado."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        print(f"🔄 Cargando modelo multi-label desde: {model_path}")
        
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extraer información del modelo
        self.disease_names = checkpoint['disease_names']
        self.num_diseases = checkpoint['num_diseases']
        
        # Cargar umbrales óptimos si existen
        if 'optimal_thresholds' in checkpoint and self.use_adaptive_thresholds:
            self.optimal_thresholds = checkpoint['optimal_thresholds']
            self.threshold = None  # Se usará umbral adaptativo
            print(f"📊 Umbrales adaptativos disponibles para {len(self.optimal_thresholds)} enfermedades")
        else:
            if self.threshold is None:
                self.threshold = 0.5  # Valor por defecto
            print(f"⚠️  Umbrales adaptativos no encontrados, usando threshold fijo: {self.threshold}")
        
        # Crear modelo
        self.model = DenseNetMultiLabelClassifier(
            num_diseases=self.num_diseases,
            pretrained=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Modelo cargado exitosamente en dispositivo: {self.device}")
        print(f"📊 Enfermedades disponibles: {self.disease_names}")
    
    def _setup_transforms(self):
        """Configurar transformaciones para las imágenes."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _apply_thresholds(self, probabilities):
        """
        Aplicar umbrales adaptativos o fijo a las probabilidades.
        
        Args:
            probabilities: Tensor o array con probabilidades
            
        Returns:
            numpy array: Predicciones binarias
        """
        # Convertir a numpy si es tensor
        if isinstance(probabilities, torch.Tensor):
            probs = probabilities.cpu().numpy()
            if probs.ndim == 2:
                probs = probs[0]
        else:
            probs = probabilities
            if probs.ndim == 2:
                probs = probs[0]
        
        # Aplicar umbrales adaptativos si están disponibles
        if self.optimal_thresholds is not None:
            predictions = np.zeros_like(probs, dtype=int)
            for i, disease in enumerate(self.disease_names):
                threshold = self.optimal_thresholds.get(disease, 0.5)
                predictions[i] = 1 if probs[i] > threshold else 0
            return predictions
        else:
            # Usar threshold fijo
            return (probs > self.threshold).astype(int)
    
    def predict_single_image(self, image_path, return_probabilities=True):
        """
        Predecir enfermedades en una sola imagen.
        
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
            probabilities = self.model(input_tensor)
        
        # Aplicar umbrales adaptativos o fijo
        predictions = self._apply_thresholds(probabilities)
        
        # Convertir probabilidades a numpy
        probabilities = probabilities.cpu().numpy()[0]
        
        # Identificar enfermedades detectadas
        detected_diseases = []
        for i, (disease, prob, pred) in enumerate(zip(self.disease_names, probabilities, predictions)):
            if pred == 1:
                confidence_level = self._get_confidence_level(prob)
                detected_diseases.append({
                    'disease': disease,
                    'probability': float(prob),
                    'confidence': confidence_level
                })
        
        # Preparar resultado
        result = {
            "image_path": image_path,
            "detected_diseases": detected_diseases,
            "is_normal": len(detected_diseases) == 0,
            "num_diseases": len(detected_diseases),
            "summary": self._generate_summary(detected_diseases)
        }
        
        if return_probabilities:
            result["all_probabilities"] = {
                disease: float(prob) for disease, prob in zip(self.disease_names, probabilities)
            }
        
        return result
    
    def predict_batch(self, image_paths, return_probabilities=True):
        """
        Predecir enfermedades en múltiples imágenes.
        
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
    
    def _get_confidence_level(self, probability):
        """Determinar nivel de confianza basado en probabilidad."""
        if probability > 0.8:
            return "Alta"
        elif probability > 0.6:
            return "Media"
        else:
            return "Baja"
    
    def _generate_summary(self, detected_diseases):
        """Generar resumen de enfermedades detectadas."""
        if not detected_diseases:
            return "No se detectaron enfermedades - Radiografía normal"
        
        if len(detected_diseases) == 1:
            disease = detected_diseases[0]
            return f"Enfermedad detectada: {disease['disease']} ({disease['confidence']})"
        else:
            diseases = [d['disease'] for d in detected_diseases]
            return f"Múltiples enfermedades detectadas: {', '.join(diseases)}"
    
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
        ax2.set_title("Resultados del Análisis", fontsize=16, fontweight='bold')
        
        if result['is_normal']:
            ax2.text(0.1, 0.9, "✅ RADIOGRAFÍA NORMAL", 
                    fontsize=18, fontweight='bold', color='green')
            ax2.text(0.1, 0.7, "No se detectaron enfermedades en la imagen.", 
                    fontsize=14, color='green')
        else:
            # Mostrar enfermedades detectadas
            y_pos = 0.9
            ax2.text(0.1, y_pos, f"🔍 ENFERMEDADES DETECTADAS ({result['num_diseases']}):", 
                    fontsize=16, fontweight='bold', color='red')
            y_pos -= 0.1
            
            for disease_info in result['detected_diseases']:
                color = 'red' if disease_info['confidence'] == 'Alta' else 'orange'
                ax2.text(0.1, y_pos, f"• {disease_info['disease']}", 
                        fontsize=14, fontweight='bold', color=color)
                y_pos -= 0.05
                ax2.text(0.2, y_pos, f"  Probabilidad: {disease_info['probability']:.1%}", 
                        fontsize=12, color=color)
                ax2.text(0.2, y_pos-0.03, f"  Confianza: {disease_info['confidence']}", 
                        fontsize=12, color=color)
                y_pos -= 0.08
        
        # Mostrar todas las probabilidades
        if 'all_probabilities' in result:
            ax2.text(0.1, 0.3, "Probabilidades Detalladas:", 
                    fontsize=12, fontweight='bold')
            y_pos = 0.25
            
            for disease, prob in result['all_probabilities'].items():
                color = 'red' if prob > self.threshold else 'gray'
                ax2.text(0.1, y_pos, f"{disease}: {prob:.1%}", 
                        fontsize=10, color=color)
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
        
        # Contar por tipo de resultado
        normal_images = sum(1 for r in successful_predictions if r['is_normal'])
        single_disease_images = sum(1 for r in successful_predictions if r['num_diseases'] == 1)
        multi_disease_images = sum(1 for r in successful_predictions if r['num_diseases'] > 1)
        
        # Contar enfermedades detectadas
        disease_counts = {}
        for disease in self.disease_names:
            disease_counts[disease] = sum(1 for r in successful_predictions 
                                        for detected in r['detected_diseases'] 
                                        if detected['disease'] == disease)
        
        # Estadísticas de confianza
        all_probabilities = []
        for r in successful_predictions:
            if 'all_probabilities' in r:
                all_probabilities.extend(r['all_probabilities'].values())
        
        avg_confidence = np.mean(all_probabilities) if all_probabilities else 0
        high_confidence = sum(1 for p in all_probabilities if p > 0.8)
        medium_confidence = sum(1 for p in all_probabilities if 0.6 <= p <= 0.8)
        low_confidence = sum(1 for p in all_probabilities if p < 0.6)
        
        return {
            'total_images': total_images,
            'successful_predictions': len(successful_predictions),
            'failed_predictions': len(failed_predictions),
            'normal_images': normal_images,
            'single_disease_images': single_disease_images,
            'multi_disease_images': multi_disease_images,
            'disease_counts': disease_counts,
            'avg_confidence': avg_confidence,
            'high_confidence_count': high_confidence,
            'medium_confidence_count': medium_confidence,
            'low_confidence_count': low_confidence
        }
    
    def print_batch_analysis(self, analysis):
        """Imprimir análisis de resultados en lote."""
        print("\n" + "="*70)
        print("📊 ANÁLISIS DE RESULTADOS MULTI-LABEL")
        print("="*70)
        print(f"Total de imágenes procesadas: {analysis['total_images']:,}")
        print(f"Predicciones exitosas: {analysis['successful_predictions']:,}")
        print(f"Predicciones fallidas: {analysis['failed_predictions']:,}")
        
        print(f"\n🏥 TIPOS DE RESULTADOS:")
        print(f"  Radiografías normales: {analysis['normal_images']:,}")
        print(f"  Con una enfermedad: {analysis['single_disease_images']:,}")
        print(f"  Con múltiples enfermedades: {analysis['multi_disease_images']:,}")
        
        print(f"\n🔍 ENFERMEDADES DETECTADAS:")
        for disease, count in analysis['disease_counts'].items():
            percentage = (count / analysis['successful_predictions'] * 100) if analysis['successful_predictions'] > 0 else 0
            print(f"  {disease}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n📈 ESTADÍSTICAS DE CONFIANZA:")
        print(f"  Confianza promedio: {analysis['avg_confidence']:.3f}")
        print(f"  Alta confianza (>80%): {analysis['high_confidence_count']:,}")
        print(f"  Confianza media (60-80%): {analysis['medium_confidence_count']:,}")
        print(f"  Baja confianza (<60%): {analysis['low_confidence_count']:,}")

def main():
    """Función principal para ejecutar predicciones desde línea de comandos."""
    
    parser = argparse.ArgumentParser(description='Clasificar enfermedades en radiografías con modelo multi-label')
    parser.add_argument('--model', type=str, default='results/models/multilabel/densenet_multilabel_model.pth',
                       help='Ruta al modelo entrenado')
    parser.add_argument('--image', type=str, help='Ruta a una imagen individual')
    parser.add_argument('--directory', type=str, help='Directorio con imágenes para procesar')
    parser.add_argument('--output', type=str, help='Archivo de salida para guardar resultados')
    parser.add_argument('--visualize', action='store_true', 
                       help='Mostrar visualización de la predicción')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Umbral para considerar enfermedad presente')
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
        predictor = MultiLabelDiseasePredictor(args.model, device=device, threshold=args.threshold)
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
            if result['detected_diseases']:
                print(f"  Enfermedades detectadas:")
                for disease in result['detected_diseases']:
                    print(f"    • {disease['disease']}: {disease['probability']:.1%} ({disease['confidence']})")
        
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
            'threshold': args.threshold,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Resultados guardados en: {args.output}")

if __name__ == "__main__":
    main()
