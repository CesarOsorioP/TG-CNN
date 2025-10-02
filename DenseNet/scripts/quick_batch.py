"""
Script rápido para procesamiento en lote de imágenes.
Versión simplificada para uso fácil y rápido.

"""

import os
import sys
from batch_predictor import BatchPredictor

def quick_batch_process(input_directory, model_path="densenet_chest_xray_model.pth"):
    """
    Procesamiento rápido en lote.
    
    Args:
        input_directory: Directorio con imágenes a procesar
        model_path: Ruta al modelo entrenado
    """
    
    print("🚀 PROCESAMIENTO RÁPIDO EN LOTE")
    print("="*50)
    
    # Verificar que existe el directorio
    if not os.path.exists(input_directory):
        print(f"❌ Error: Directorio no encontrado: {input_directory}")
        return None
    
    # Verificar que existe el modelo
    if not os.path.exists(model_path):
        print(f"❌ Error: Modelo no encontrado: {model_path}")
        print("💡 Primero entrena el modelo con: python train_model.py")
        return None
    
    try:
        # Crear predictor
        print("🔄 Inicializando predictor...")
        predictor = BatchPredictor(model_path=model_path, batch_size=64)  # Lote más grande para velocidad
        
        # Procesar directorio
        print(f"📁 Procesando directorio: {input_directory}")
        summary = predictor.process_directory(
            input_dir=input_directory,
            output_dir="quick_batch_results",
            save_results=True
        )
        
        if summary:
            print("\n" + "="*50)
            print("🎉 ¡PROCESAMIENTO COMPLETADO!")
            print("="*50)
            print(f"📊 Total de imágenes: {summary['total_images']:,}")
            print(f"🏥 Radiografías de tórax: {summary['chest_xray_count']:,} ({summary['chest_xray_percentage']:.1f}%)")
            print(f"🖼️  Otras imágenes: {summary['other_count']:,} ({summary['other_percentage']:.1f}%)")
            print(f"✅ Predicciones exitosas: {summary['successful_predictions']:,}")
            if summary['failed_predictions'] > 0:
                print(f"❌ Errores: {summary['failed_predictions']:,}")
            print(f"📈 Confianza promedio: {summary['average_confidence']:.3f}")
            print("\n💾 Resultados guardados en: quick_batch_results/")
            return summary
        else:
            print("❌ No se pudieron procesar las imágenes")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Función principal."""
    if len(sys.argv) < 2:
        print("Uso: python quick_batch.py <directorio_con_imagenes> [modelo]")
        print("\nEjemplos:")
        print("  python quick_batch.py mi_dataset/")
        print("  python quick_batch.py mi_dataset/ mi_modelo.pth")
        return 1
    
    input_dir = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "densenet_chest_xray_model.pth"
    
    # Procesar
    summary = quick_batch_process(input_dir, model_path)
    
    if summary:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())
