#!/usr/bin/env python3
"""
DenseNet Chest X-Ray Classifier - Script Principal
==================================================

Script principal para acceder a todas las funcionalidades del proyecto.

Uso:
    python main.py [comando] [opciones]

Comandos disponibles:
    train       - Entrenar el modelo
    predict     - Predecir una imagen individual
    gui         - Abrir la interfaz gráfica
    batch       - Procesar imágenes en lote
    analyze     - Analizar resultados
    quick       - Inicio rápido
"""

import sys
import os
import argparse
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="DenseNet Chest X-Ray Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando train
    train_parser = subparsers.add_parser('train', help='Entrenar el modelo')
    train_parser.add_argument('--data_dir', type=str, default='data', 
                             help='Directorio con los datos de entrenamiento')
    train_parser.add_argument('--epochs', type=int, default=15, 
                             help='Número de épocas de entrenamiento')
    train_parser.add_argument('--batch_size', type=int, default=16, 
                             help='Tamaño del lote')
    train_parser.add_argument('--freeze_backbone', action='store_true', 
                             help='Congelar el backbone para evitar overfitting')
    
    # Comando predict
    predict_parser = subparsers.add_parser('predict', help='Predecir una imagen')
    predict_parser.add_argument('image_path', type=str, help='Ruta a la imagen')
    predict_parser.add_argument('--model', type=str, 
                               default='results/models/densenet_chest_xray_model.pth',
                               help='Ruta al modelo entrenado')
    
    # Comando gui
    gui_parser = subparsers.add_parser('gui', help='Abrir la interfaz gráfica')
    gui_parser.add_argument('--model', type=str, 
                           default='results/models/densenet_chest_xray_model.pth',
                           help='Ruta al modelo entrenado')
    
    # Comando batch
    batch_parser = subparsers.add_parser('batch', help='Procesar imágenes en lote')
    batch_parser.add_argument('input_dir', type=str, help='Directorio con imágenes')
    batch_parser.add_argument('--output_dir', type=str, default='results/predictions',
                             help='Directorio de salida')
    batch_parser.add_argument('--model', type=str, 
                             default='results/models/densenet_chest_xray_model.pth',
                             help='Ruta al modelo entrenado')
    batch_parser.add_argument('--batch_size', type=int, default=32,
                             help='Tamaño del lote')
    
    # Comando analyze
    analyze_parser = subparsers.add_parser('analyze', help='Analizar resultados')
    analyze_parser.add_argument('--model', type=str, 
                               default='results/models/densenet_chest_xray_model.pth',
                               help='Ruta al modelo entrenado')
    analyze_parser.add_argument('--results_dir', type=str, 
                               default='results/predictions',
                               help='Directorio con resultados')
    
    # Comando quick
    quick_parser = subparsers.add_parser('quick', help='Inicio rápido')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'train':
            from src.models.train_model import main as train_main
            train_main()
            
        elif args.command == 'predict':
            from src.models.predict import predict_image
            result = predict_image(args.image_path, args.model)
            print(f"Predicción: {result}")
            
        elif args.command == 'gui':
            from gui.gui_app import main as gui_main
            gui_main()
            
        elif args.command == 'batch':
            from scripts.batch_predictor import main as batch_main
            # Simular argumentos para batch_predictor
            sys.argv = ['batch_predictor.py', '--input_dir', args.input_dir, 
                       '--output_dir', args.output_dir, '--model', args.model,
                       '--batch_size', str(args.batch_size)]
            batch_main()
            
        elif args.command == 'analyze':
            from src.analysis.analyze_model import main as analyze_main
            analyze_main()
            
        elif args.command == 'quick':
            from scripts.quick_start import main as quick_main
            quick_main()
            
    except ImportError as e:
        print(f"Error importando módulo: {e}")
        print("Asegúrate de que todas las dependencias estén instaladas.")
        return 1
    except Exception as e:
        print(f"Error ejecutando comando '{args.command}': {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
