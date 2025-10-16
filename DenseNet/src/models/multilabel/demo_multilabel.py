"""
Script de demostración para el sistema multi-label.
Muestra cómo usar todas las funcionalidades del sistema paso a paso.

"""

import os
import sys
from pathlib import Path

def print_banner():
    """Imprimir banner de demostración."""
    print("="*80)
    print(" DEMOSTRACIÓN DEL SISTEMA MULTI-LABEL")
    print("   Clasificación de Enfermedades en Radiografías de Tórax")
    print("="*80)

def print_step(step, title, description):
    """Imprimir paso de la demostración."""
    print(f"\n📋 PASO {step}: {title}")
    print("-" * 60)
    print(description)

def print_command(command, description=""):
    """Imprimir comando con descripción."""
    print(f"\n💻 Comando:")
    print(f"   {command}")
    if description:
        print(f"\n📝 Descripción: {description}")

def check_requirements():
    """Verificar requisitos del sistema."""
    print_step(0, "VERIFICACIÓN DE REQUISITOS", 
               "Verificando que el sistema esté configurado correctamente...")
    
    # Verificar estructura de directorios
    required_dirs = [
        'data_diseases',
        'results/models/multilabel',
        'scripts/PredicciónImágenes'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ Directorios faltantes:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        print("\n💡 Solución: Ejecuta primero el script de preparación de datos")
        return False
    
    print("✅ Estructura de directorios verificada")
    
    # Verificar archivos de datos
    data_diseases = Path('data_diseases')
    disease_folders = ['Neumonía', 'Cáncer', 'Atelectasia', 'Edema', 'Tuberculosis', 'COVID-19']
    
    missing_diseases = []
    for disease in disease_folders:
        disease_path = data_diseases / disease
        if not disease_path.exists() or len(list(disease_path.glob('*'))) == 0:
            missing_diseases.append(disease)
    
    if missing_diseases:
        print("❌ Carpetas de enfermedades faltantes o vacías:")
        for disease in missing_diseases:
            print(f"   - {disease}")
        print("\n💡 Solución: Ejecuta el script de preparación de datos")
        return False
    
    print("✅ Datos de enfermedades verificados")
    return True

def demonstrate_data_preparation():
    """Demostrar preparación de datos."""
    print_step(1, "PREPARACIÓN DE DATOS",
               "Organizar las imágenes filtradas en la estructura multi-label")
    
    print_command(
        "python src/models/multilabel/main_multilabel.py prepare-data",
        "Copia las imágenes filtradas de 'scripts/PredicciónImágenes' a 'data_diseases' con la estructura correcta para multi-label"
    )
    
    print("\n📁 Estructura resultante:")
    print("   data_diseases/")
    print("   ├── Neumonía/        # Imágenes con neumonía")
    print("   ├── Cáncer/          # Imágenes con cáncer")
    print("   ├── Atelectasia/     # Imágenes con atelectasia")
    print("   ├── Edema/           # Imágenes con edema")
    print("   ├── Tuberculosis/    # Imágenes con tuberculosis")
    print("   └── COVID-19/        # Imágenes con COVID-19")

def demonstrate_training():
    """Demostrar entrenamiento del modelo."""
    print_step(2, "ENTRENAMIENTO DEL MODELO",
               "Entrenar el modelo DenseNet multi-label con transfer learning")
    
    print_command(
        "python src/models/multilabel/main_multilabel.py train --data_dir data_diseases",
        "Entrena el modelo con configuración por defecto (20 épocas, batch_size=16, learning_rate=0.001)"
    )
    
    print("\n🔧 Configuración avanzada:")
    print_command(
        "python src/models/multilabel/main_multilabel.py train \\\n    --data_dir data_diseases \\\n    --batch_size 32 \\\n    --num_epochs 25 \\\n    --learning_rate 0.0005 \\\n    --loss_type focal \\\n    --fine_tune_epochs 10",
        "Entrenamiento con parámetros personalizados y fine-tuning"
    )
    
    print("\n📊 Archivos generados:")
    print("   results/models/multilabel/")
    print("   ├── densenet_multilabel_model.pth    # Modelo entrenado")
    print("   ├── training_config.json             # Configuración")
    print("   └── training_history.png             # Gráficos")

def demonstrate_prediction():
    """Demostrar predicción de imágenes."""
    print_step(3, "PREDICCIÓN DE IMÁGENES",
               "Usar el modelo entrenado para clasificar nuevas imágenes")
    
    print_command(
        "python src/models/multilabel/main_multilabel.py predict --image ruta/a/imagen.jpg",
        "Clasificar una imagen individual y mostrar enfermedades detectadas"
    )
    
    print("\n🎨 Con visualización:")
    print_command(
        "python src/models/multilabel/main_multilabel.py predict --image imagen.jpg --visualize",
        "Muestra la imagen con resultados visuales y gráficos de probabilidades"
    )
    
    print("\n📁 Procesar directorio completo:")
    print_command(
        "python src/models/multilabel/main_multilabel.py predict --directory ruta/a/directorio",
        "Procesa todas las imágenes en un directorio y genera reporte detallado"
    )
    
    print("\n🎯 Con umbral personalizado:")
    print_command(
        "python src/models/multilabel/main_multilabel.py predict --image imagen.jpg --threshold 0.6",
        "Usa umbral de 0.6 en lugar del 0.5 por defecto (más estricto)"
    )

def demonstrate_evaluation():
    """Demostrar evaluación del modelo."""
    print_step(4, "EVALUACIÓN DEL MODELO",
               "Evaluar el rendimiento del modelo con métricas multi-label")
    
    print_command(
        "python src/models/multilabel/main_multilabel.py evaluate",
        "Evalúa el modelo con datos de prueba y genera métricas detalladas"
    )
    
    print("\n📊 Métricas generadas:")
    print("   • F1-Score (Macro/Micro)")
    print("   • Hamming Loss")
    print("   • Exact Match Ratio")
    print("   • Jaccard Score")
    print("   • AUC-ROC por enfermedad")
    print("   • Análisis de errores")
    print("   • Distribución de confianza")
    
    print("\n📈 Gráficos generados:")
    print("   • Métricas por enfermedad")
    print("   • Análisis de errores")
    print("   • Rendimiento vs umbral de confianza")

def demonstrate_programmatic_usage():
    """Demostrar uso programático."""
    print_step(5, "USO PROGRAMÁTICO",
               "Usar el sistema desde código Python")
    
    print("\n🐍 Código de ejemplo:")
    print("""
from src.models.multilabel import MultiLabelDiseasePredictor

# Crear predictor
predictor = MultiLabelDiseasePredictor(
    model_path='results/models/multilabel/densenet_multilabel_model.pth',
    threshold=0.5
)

# Predecir imagen
result = predictor.predict_single_image('imagen.jpg')

# Mostrar resultados
print(f"Enfermedades detectadas: {len(result['detected_diseases'])}")
for disease in result['detected_diseases']:
    print(f"  • {disease['disease']}: {disease['probability']:.1%}")

# Procesar lote
results = predictor.predict_batch(['img1.jpg', 'img2.jpg'])
analysis = predictor.analyze_batch_results(results)
predictor.print_batch_analysis(analysis)
    """)

def demonstrate_interpretation():
    """Demostrar interpretación de resultados."""
    print_step(6, "INTERPRETACIÓN DE RESULTADOS",
               "Entender qué significan los resultados del modelo")
    
    print("\n🎯 Tipos de resultados:")
    print("   1. Radiografía Normal:")
    print("      • detected_diseases: []")
    print("      • is_normal: true")
    print("      • summary: 'No se detectaron enfermedades'")
    
    print("\n   2. Una Enfermedad:")
    print("      • detected_diseases: [{'disease': 'Neumonía', 'probability': 0.85}]")
    print("      • is_normal: false")
    print("      • num_diseases: 1")
    
    print("\n   3. Múltiples Enfermedades:")
    print("      • detected_diseases: [{'disease': 'Neumonía', 'probability': 0.85}, ...]")
    print("      • is_normal: false")
    print("      • num_diseases: 2")
    
    print("\n📊 Niveles de confianza:")
    print("   • Alta (>80%): Enfermedad muy probable")
    print("   • Media (60-80%): Requiere revisión adicional")
    print("   • Baja (<60%): Probablemente no presente")
    
    print("\n⚠️  Consideraciones importantes:")
    print("   • No es un diagnóstico médico")
    print("   • Requiere supervisión médica")
    print("   • Usar como herramienta de apoyo")

def demonstrate_troubleshooting():
    """Demostrar solución de problemas."""
    print_step(7, "SOLUCIÓN DE PROBLEMAS",
               "Resolver problemas comunes del sistema")
    
    print("\n❌ Problemas comunes:")
    
    print("\n   1. Error: 'Modelo no encontrado'")
    print("      💡 Solución: Entrena el modelo primero")
    print("      python src/models/multilabel/main_multilabel.py train")
    
    print("\n   2. Error: 'Directorio de datos no encontrado'")
    print("      💡 Solución: Prepara los datos primero")
    print("      python src/models/multilabel/main_multilabel.py prepare-data")
    
    print("\n   3. Error: 'CUDA out of memory'")
    print("      💡 Solución: Reduce el batch_size")
    print("      python src/models/multilabel/main_multilabel.py train --batch_size 8")
    
    print("\n   4. Bajo rendimiento del modelo")
    print("      💡 Solución: Ajusta hiperparámetros")
    print("      • Aumenta num_epochs")
    print("      • Ajusta learning_rate")
    print("      • Prueba diferentes loss_type")
    
    print("\n   5. Predicciones inconsistentes")
    print("      💡 Solución: Ajusta el threshold")
    print("      python src/models/multilabel/main_multilabel.py predict --threshold 0.6")

def main():
    """Función principal de la demostración."""
    print_banner()
    
    print("\n🎯 Esta demostración te guiará a través de todo el flujo de trabajo")
    print("   del sistema multi-label para clasificación de enfermedades.")
    
    # Verificar requisitos
    if not check_requirements():
        print("\n❌ No se pueden continuar las demostraciones sin los requisitos básicos.")
        print("   Por favor, ejecuta primero el script de preparación de datos.")
        return
    
    # Demostrar cada paso
    demonstrate_data_preparation()
    demonstrate_training()
    demonstrate_prediction()
    demonstrate_evaluation()
    demonstrate_programmatic_usage()
    demonstrate_interpretation()
    demonstrate_troubleshooting()
    
    print("\n" + "="*80)
    print("🎉 ¡DEMOSTRACIÓN COMPLETADA!")
    print("="*80)
    print("\n📚 Recursos adicionales:")
    print("   • README.md: Documentación completa")
    print("   • Código fuente: Comentarios detallados")
    print("   • Logs de entrenamiento: Para debugging")
    print("   • Gráficos generados: Para análisis visual")
    
    print("\n🚀 ¡Ahora estás listo para usar el sistema multi-label!")
    print("   Comienza con: python src/models/multilabel/main_multilabel.py prepare-data")

if __name__ == "__main__":
    main()
