"""
Script para predicción en lote de grandes cantidades de imágenes.
Procesa miles de imágenes automáticamente y genera reportes detallados.

"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
import time
from datetime import datetime
from pathlib import Path
import argparse
from tqdm import tqdm
import multiprocessing as mp
from collections import Counter
import csv

class DenseNetClassifier(nn.Module):
    """Clasificador DenseNet con freeze del backbone."""
    
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
        super(DenseNetClassifier, self).__init__()
        self.backbone = models.densenet121(pretrained=pretrained)
        
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

class BatchPredictor:
    """
    Clasificador en lote para procesar grandes cantidades de imágenes.
    """
    
    def __init__(self, model_path, device=None, batch_size=32):
        """
        Inicializar el predictor en lote.
        
        Args:
            model_path: Ruta al modelo entrenado
            device: Dispositivo a usar ('cuda', 'cpu', o None para auto-detectar)
            batch_size: Tamaño del lote para procesamiento
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
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
        
        print(f"🔄 Cargando modelo desde: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = DenseNetClassifier(num_classes=2, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = checkpoint['idx_to_class']
        
        print(f"✅ Modelo cargado exitosamente en dispositivo: {self.device}")
        print(f"📊 Clases disponibles: {list(self.idx_to_class.values())}")
    
    def _setup_transforms(self):
        """Configurar transformaciones para las imágenes."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_single_image(self, image_path):
        """
        Predecir una sola imagen.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            dict: Resultado de la predicción
        """
        try:
            # Cargar y preprocesar imagen
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Realizar predicción
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()
            
            predicted_class = self.idx_to_class[predicted_class_idx]
            
            return {
                "image_path": image_path,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "is_chest_xray": predicted_class == "chest_xray",
                "probabilities": {
                    self.idx_to_class[i]: probabilities[0][i].item() 
                    for i in range(len(self.idx_to_class))
                }
            }
            
        except Exception as e:
            return {
                "image_path": image_path,
                "error": str(e),
                "predicted_class": None,
                "confidence": 0.0,
                "is_chest_xray": False
            }
    
    def predict_batch(self, image_paths):
        """
        Predecir un lote de imágenes.
        
        Args:
            image_paths: Lista de rutas a las imágenes
            
        Returns:
            list: Lista de resultados de predicción
        """
        results = []
        
        # Procesar en lotes
        for i in tqdm(range(0, len(image_paths), self.batch_size), 
                     desc="Procesando lotes", unit="lote"):
            batch_paths = image_paths[i:i + self.batch_size]
            
            # Procesar cada imagen en el lote
            for image_path in batch_paths:
                result = self.predict_single_image(image_path)
                results.append(result)
        
        return results
    
    def find_images(self, directory, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
        """
        Encontrar todas las imágenes en un directorio.
        
        Args:
            directory: Directorio a buscar
            extensions: Extensiones de archivo a procesar
            
        Returns:
            list: Lista de rutas a las imágenes encontradas
        """
        image_paths = []
        
        print(f"🔍 Buscando imágenes en: {directory}")
        
        for ext in extensions:
            # Buscar con extensión en minúsculas
            pattern = f"**/*{ext}"
            image_paths.extend(Path(directory).glob(pattern))
            
            # Buscar con extensión en mayúsculas
            pattern = f"**/*{ext.upper()}"
            image_paths.extend(Path(directory).glob(pattern))
        
        # Convertir a strings y eliminar duplicados
        image_paths = list(set(str(p) for p in image_paths))
        
        print(f"📁 Encontradas {len(image_paths)} imágenes")
        return image_paths
    
    def process_directory(self, input_dir, output_dir=None, save_results=True):
        """
        Procesar todas las imágenes en un directorio.
        
        Args:
            input_dir: Directorio con imágenes a procesar
            output_dir: Directorio para guardar resultados (opcional)
            save_results: Si guardar resultados en archivos
            
        Returns:
            dict: Resumen de resultados
        """
        start_time = time.time()
        
        # Encontrar imágenes
        image_paths = self.find_images(input_dir)
        
        if not image_paths:
            print("❌ No se encontraron imágenes para procesar")
            return None
        
        print(f"\n🚀 Iniciando procesamiento de {len(image_paths)} imágenes...")
        print(f"📊 Tamaño de lote: {self.batch_size}")
        print(f"💻 Dispositivo: {self.device}")
        
        # Procesar imágenes
        results = self.predict_batch(image_paths)
        
        # Analizar resultados
        summary = self.analyze_results(results)
        
        # Mostrar resumen
        self.print_summary(summary)
        
        # Guardar resultados si se solicita
        if save_results and output_dir:
            self.save_results(results, summary, output_dir)
        
        # Tiempo total
        total_time = time.time() - start_time
        print(f"\n⏱️  Tiempo total: {total_time:.2f} segundos")
        print(f"⚡ Velocidad: {len(image_paths)/total_time:.2f} imágenes/segundo")
        
        return summary
    
    def analyze_results(self, results):
        """Analizar resultados y generar resumen."""
        total_images = len(results)
        successful_predictions = [r for r in results if 'error' not in r]
        failed_predictions = [r for r in results if 'error' in r]
        
        # Contar por clase
        class_counts = Counter(r['predicted_class'] for r in successful_predictions)
        chest_xray_count = class_counts.get('chest_xray', 0)
        other_count = class_counts.get('other_images', 0)
        
        # Estadísticas de confianza
        confidences = [r['confidence'] for r in successful_predictions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Distribución de confianza
        high_confidence = len([c for c in confidences if c > 0.9])
        medium_confidence = len([c for c in confidences if 0.7 <= c <= 0.9])
        low_confidence = len([c for c in confidences if c < 0.7])
        
        return {
            'total_images': total_images,
            'successful_predictions': len(successful_predictions),
            'failed_predictions': len(failed_predictions),
            'chest_xray_count': chest_xray_count,
            'other_count': other_count,
            'chest_xray_percentage': (chest_xray_count / len(successful_predictions) * 100) if successful_predictions else 0,
            'other_percentage': (other_count / len(successful_predictions) * 100) if successful_predictions else 0,
            'average_confidence': avg_confidence,
            'high_confidence_count': high_confidence,
            'medium_confidence_count': medium_confidence,
            'low_confidence_count': low_confidence,
            'results': results
        }
    
    def print_summary(self, summary):
        """Imprimir resumen de resultados."""
        print("\n" + "="*60)
        print("📊 RESUMEN DE RESULTADOS")
        print("="*60)
        print(f"Total de imágenes procesadas: {summary['total_images']:,}")
        print(f"Predicciones exitosas: {summary['successful_predictions']:,}")
        print(f"Predicciones fallidas: {summary['failed_predictions']:,}")
        
        print(f"\n🏥 RADIOGRAFÍAS DE TÓRAX:")
        print(f"  Cantidad: {summary['chest_xray_count']:,}")
        print(f"  Porcentaje: {summary['chest_xray_percentage']:.1f}%")
        
        print(f"\n🖼️  OTRAS IMÁGENES:")
        print(f"  Cantidad: {summary['other_count']:,}")
        print(f"  Porcentaje: {summary['other_percentage']:.1f}%")
        
        print(f"\n📈 ESTADÍSTICAS DE CONFIANZA:")
        print(f"  Confianza promedio: {summary['average_confidence']:.3f}")
        print(f"  Alta confianza (>90%): {summary['high_confidence_count']:,}")
        print(f"  Confianza media (70-90%): {summary['medium_confidence_count']:,}")
        print(f"  Baja confianza (<70%): {summary['low_confidence_count']:,}")
        
        if summary['failed_predictions'] > 0:
            print(f"\n❌ ERRORES:")
            print(f"  Imágenes con error: {summary['failed_predictions']:,}")
            print(f"  Tasa de error: {summary['failed_predictions']/summary['total_images']*100:.1f}%")
    
    def save_results(self, results, summary, output_dir):
        """Guardar resultados en archivos."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar resumen en JSON
        summary_file = os.path.join(output_dir, f"batch_prediction_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            # Crear copia del resumen sin los resultados completos para el JSON
            summary_copy = summary.copy()
            del summary_copy['results']  # Los resultados completos van en CSV
            json.dump(summary_copy, f, indent=2)
        
        # Guardar resultados detallados en CSV
        csv_file = os.path.join(output_dir, f"batch_prediction_results_{timestamp}.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'image_path', 'predicted_class', 'confidence', 'is_chest_xray',
                'prob_chest_xray', 'prob_other_images', 'error'
            ])
            
            for result in results:
                writer.writerow([
                    result['image_path'],
                    result['predicted_class'],
                    result['confidence'],
                    result['is_chest_xray'],
                    result.get('probabilities', {}).get('chest_xray', 0),
                    result.get('probabilities', {}).get('other_images', 0),
                    result.get('error', '')
                ])
        
        # Guardar lista de radiografías de tórax
        chest_xray_file = os.path.join(output_dir, f"chest_xray_images_{timestamp}.txt")
        with open(chest_xray_file, 'w') as f:
            for result in results:
                if result.get('is_chest_xray', False):
                    f.write(f"{result['image_path']}\n")
        
        # Guardar lista de otras imágenes
        other_images_file = os.path.join(output_dir, f"other_images_{timestamp}.txt")
        with open(other_images_file, 'w') as f:
            for result in results:
                if not result.get('is_chest_xray', True) and 'error' not in result:
                    f.write(f"{result['image_path']}\n")
        
        print(f"\n💾 Resultados guardados en: {output_dir}")
        print(f"  📄 Resumen: {summary_file}")
        print(f"  📊 Detalles: {csv_file}")
        print(f"  🏥 Radiografías: {chest_xray_file}")
        print(f"  🖼️  Otras imágenes: {other_images_file}")

def main():
    """Función principal para procesamiento en lote."""
    parser = argparse.ArgumentParser(description='Procesar grandes cantidades de imágenes con DenseNet')
    parser.add_argument('--model', type=str, default='densenet_chest_xray_model.pth',
                       help='Ruta al modelo entrenado')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directorio con imágenes a procesar')
    parser.add_argument('--output_dir', type=str, default='batch_results',
                       help='Directorio para guardar resultados')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Tamaño del lote para procesamiento')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                       default='auto', help='Dispositivo a usar')
    parser.add_argument('--no_save', action='store_true',
                       help='No guardar resultados en archivos')
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*70)
    print("🔬 PROCESADOR EN LOTE - CLASIFICADOR DE RADIOGRAFÍAS DE TÓRAX")
    print("="*70)
    print(f"📁 Directorio de entrada: {args.input_dir}")
    print(f"💾 Directorio de salida: {args.output_dir}")
    print(f"🤖 Modelo: {args.model}")
    print(f"📊 Tamaño de lote: {args.batch_size}")
    print(f"💻 Dispositivo: {device}")
    print("="*70)
    
    try:
        # Crear predictor
        predictor = BatchPredictor(
            model_path=args.model,
            device=device,
            batch_size=args.batch_size
        )
        
        # Procesar directorio
        summary = predictor.process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir if not args.no_save else None,
            save_results=not args.no_save
        )
        
        if summary:
            print("\n✅ Procesamiento completado exitosamente!")
        else:
            print("\n❌ No se pudieron procesar las imágenes")
            
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
