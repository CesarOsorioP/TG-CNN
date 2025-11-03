"""
Script principal para el sistema multi-label de clasificación de enfermedades.
Proporciona una interfaz unificada para todas las operaciones multi-label.

"""

import argparse
import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent.parent))

def main():
    """
    Función principal que actúa como punto de entrada para todas las operaciones multi-label.
    """
    parser = argparse.ArgumentParser(
        description='Sistema Multi-Label para Clasificación de Enfermedades en Radiografías',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

1. Preparar datos:
   python src/models/multilabel/main_multilabel.py prepare-data

2. Entrenar modelo:
   python src/models/multilabel/main_multilabel.py train --data_dir data_diseases

3. Predecir imagen individual:
   python src/models/multilabel/main_multilabel.py predict --image ruta/a/imagen.jpg

4. Predecir directorio:
   python src/models/multilabel/main_multilabel.py predict --directory ruta/a/directorio

5. Evaluar modelo:
   python src/models/multilabel/main_multilabel.py evaluate --model results/models/multilabel/densenet_multilabel_model.pth
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando para preparar datos
    prepare_parser = subparsers.add_parser('prepare-data', help='Preparar datos para entrenamiento multi-label')
    prepare_parser.add_argument('--source_dir', type=str, default='scripts/PredicciónImágenes',
                               help='Directorio con imágenes filtradas')
    prepare_parser.add_argument('--target_dir', type=str, default='data_diseases',
                               help='Directorio destino para datos multi-label')
    
    # Comando para entrenar
    train_parser = subparsers.add_parser('train', help='Entrenar modelo multi-label')
    train_parser.add_argument('--data_dir', type=str, default='data_diseases',
                             help='Directorio con datos multi-label')
    train_parser.add_argument('--batch_size', type=int, default=32,
                             help='Tamaño del lote')
    train_parser.add_argument('--num_epochs', type=int, default=15,
                             help='Número de épocas')
    train_parser.add_argument('--learning_rate', type=float, default=0.001,
                             help='Tasa de aprendizaje')
    train_parser.add_argument('--freeze_backbone', action='store_true', default=True,
                             help='Congelar backbone durante entrenamiento')
    train_parser.add_argument('--fine_tune_epochs', type=int, default=3,
                             help='Épocas de fine-tuning')
    train_parser.add_argument('--loss_type', type=str, default='bce',
                             choices=['bce', 'focal', 'weighted_bce'],
                             help='Tipo de función de pérdida')
    train_parser.add_argument('--output_dir', type=str, default='results/models/multilabel',
                             help='Directorio para guardar resultados')
    
    # Comando para predecir
    predict_parser = subparsers.add_parser('predict', help='Realizar predicciones con modelo multi-label')
    predict_parser.add_argument('--model', type=str, default='results/models/multilabel/densenet_multilabel_model.pth',
                               help='Ruta al modelo entrenado')
    predict_parser.add_argument('--image', type=str, help='Ruta a una imagen individual')
    predict_parser.add_argument('--directory', type=str, help='Directorio con imágenes para procesar')
    predict_parser.add_argument('--output', type=str, help='Archivo de salida para guardar resultados')
    predict_parser.add_argument('--visualize', action='store_true',
                               help='Mostrar visualización de la predicción')
    predict_parser.add_argument('--threshold', type=float, default=0.5,
                               help='Umbral para considerar enfermedad presente')
    
    # Comando para evaluar
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluar modelo multi-label')
    evaluate_parser.add_argument('--model', type=str, default='results/models/multilabel/densenet_multilabel_model.pth',
                                help='Ruta al modelo entrenado')
    evaluate_parser.add_argument('--data_dir', type=str, default='data_diseases',
                                help='Directorio con datos de prueba')
    evaluate_parser.add_argument('--batch_size', type=int, default=16,
                                help='Tamaño del lote para evaluación')
    evaluate_parser.add_argument('--threshold', type=float, default=0.5,
                                help='Umbral para predicciones binarias')
    
    # Comando para mostrar información
    info_parser = subparsers.add_parser('info', help='Mostrar información del sistema multi-label')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print("🚀 SISTEMA MULTI-LABEL DE CLASIFICACIÓN DE ENFERMEDADES")
    print("="*60)
    
    try:
        if args.command == 'prepare-data':
            from prepare_data import main as prepare_main
            prepare_main()
            
        elif args.command == 'train':
            from train_multilabel import main as train_main
            # Pasar argumentos específicos de entrenamiento
            train_args = []
            for k, v in vars(args).items():
                if k != 'command' and v is not None:
                    if k == 'freeze_backbone' and v:  # Manejar argumentos store_true
                        train_args.append(f'--{k}')
                    elif k != 'freeze_backbone':  # Otros argumentos con valor
                        train_args.append(f'--{k}={v}')
            sys.argv = ['train_multilabel.py'] + train_args
            train_main()
            
        elif args.command == 'predict':
            from predict_multilabel import main as predict_main
            predict_args = []
            for k, v in vars(args).items():
                if k != 'command' and v is not None:
                    if isinstance(v, bool) and v:  # Handle store_true arguments
                        predict_args.append(f'--{k}')
                    elif not isinstance(v, bool):  # Other arguments with value
                        predict_args.append(f'--{k}={v}')
            sys.argv = ['predict_multilabel.py'] + predict_args
            predict_main()
            
        elif args.command == 'evaluate':
            from evaluate_multilabel import main as evaluate_main
            evaluate_args = []
            for k, v in vars(args).items():
                if k != 'command' and v is not None:
                    if isinstance(v, bool) and v:  # Handle store_true arguments
                        evaluate_args.append(f'--{k}')
                    elif not isinstance(v, bool):  # Other arguments with value
                        evaluate_args.append(f'--{k}={v}')
            sys.argv = ['evaluate_multilabel.py'] + evaluate_args
            evaluate_main()
            
        elif args.command == 'info':
            show_system_info()
            
    except Exception as e:
        print(f"❌ Error ejecutando comando '{args.command}': {e}")
        return 1
    
    return 0

def show_system_info():
    """Mostrar información del sistema multi-label."""
    print("\n📊 INFORMACIÓN DEL SISTEMA MULTI-LABEL")
    print("="*50)
    
    print("\n🏥 Enfermedades soportadas:")
    diseases = ['Neumonía', 'Atelectasia', 'Edema', 'Tuberculosis', 'COVID-19', 'Normal', 'Nodules', 'Mass']
    for i, disease in enumerate(diseases, 1):
        print(f"  {i}. {disease}")
    
    print("\n🔧 Características del modelo:")
    print("  • Arquitectura: DenseNet-121")
    print("  • Transfer Learning: Sí")
    print("  • Freeze Backbone: Sí (opcional)")
    print("  • Clasificación: Multi-label")
    print("  • Detección: Múltiples enfermedades simultáneas")
    
    print("\n📁 Estructura de datos esperada:")
    print("  data_diseases/")
    for disease in diseases:
        print(f"  ├── {disease}/")
    print("  └── (imágenes por enfermedad)")
    
    print("\n🎯 Tipos de resultados:")
    print("  • Radiografía normal (0 enfermedades)")
    print("  • Una enfermedad detectada")
    print("  • Múltiples enfermedades detectadas")
    
    print("\n📈 Métricas de evaluación:")
    print("  • F1-Score (Macro/Micro)")
    print("  • Hamming Loss")
    print("  • Exact Match Ratio")
    print("  • Jaccard Score")
    print("  • AUC-ROC por enfermedad")
    
    print("\n🚀 Comandos disponibles:")
    print("  • prepare-data: Preparar datos para entrenamiento")
    print("  • train: Entrenar modelo multi-label")
    print("  • predict: Realizar predicciones")
    print("  • evaluate: Evaluar modelo")
    print("  • info: Mostrar esta información")
    
    print("\n💡 Ejemplo de flujo completo:")
    print("  1. python src/models/multilabel/main_multilabel.py prepare-data")
    print("  2. python src/models/multilabel/main_multilabel.py train")
    print("  3. python src/models/multilabel/main_multilabel.py predict --image imagen.jpg")
    print("  4. python src/models/multilabel/main_multilabel.py evaluate")

if __name__ == "__main__":
    exit(main())
