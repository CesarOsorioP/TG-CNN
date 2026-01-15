"""
Script principal para el sistema single-label de clasificación de enfermedades.
Proporciona una interfaz unificada para todas las operaciones single-label.

"""

import argparse
import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent.parent))

def main():
    """
    Función principal que actúa como punto de entrada para todas las operaciones single-label.
    """
    parser = argparse.ArgumentParser(
        description='Sistema Single-Label para Clasificación de Enfermedades en Radiografías',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

1. Preparar datos (usa el mismo script de multilabel):
   python src/models/multilabel/prepare_data.py

2. Entrenar modelo:
   python src/models/singlelabel/main_singlelabel.py train --data_dir data_diseases

3. Predecir imagen individual:
   python src/models/singlelabel/main_singlelabel.py predict --image ruta/a/imagen.jpg

4. Predecir directorio:
   python src/models/singlelabel/main_singlelabel.py predict --directory ruta/a/directorio

5. Evaluar modelo:
   python src/models/singlelabel/main_singlelabel.py evaluate --model results/models/singlelabel/densenet_singlelabel_model.pth
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando para entrenar
    train_parser = subparsers.add_parser('train', help='Entrenar modelo single-label')
    train_parser.add_argument('--data_dir', type=str, default='data_diseases',
                             help='Directorio con datos single-label')
    train_parser.add_argument('--batch_size', type=int, default=16,
                             help='Tamaño del lote')
    train_parser.add_argument('--num_epochs', type=int, default=25,
                             help='Número de épocas')
    train_parser.add_argument('--learning_rate', type=float, default=0.0005,
                             help='Tasa de aprendizaje')
    train_parser.add_argument('--freeze_backbone', action='store_true', default=True,
                             help='Congelar backbone durante entrenamiento')
    train_parser.add_argument('--fine_tune_epochs', type=int, default=8,
                             help='Épocas de fine-tuning')
    train_parser.add_argument('--fine_tune_lr', type=float, default=0.00005,
                             help='Tasa de aprendizaje para fine-tuning')
    train_parser.add_argument('--loss_type', type=str, default='ce',
                             choices=['ce', 'weighted_ce', 'focal', 'weighted_focal'],
                             help='Tipo de función de pérdida')
    train_parser.add_argument('--gamma', type=float, default=2.0,
                             help='Factor gamma para Focal Loss (mayor = más enfoque en ejemplos difíciles)')
    train_parser.add_argument('--alpha', type=float, default=None,
                             help='Peso alpha para Focal Loss (opcional, para balance adicional)')
    train_parser.add_argument('--output_dir', type=str, default='results/models/singlelabel',
                             help='Directorio para guardar resultados')
    train_parser.add_argument('--device', type=str, default='auto',
                             choices=['auto', 'cuda', 'cpu'],
                             help='Dispositivo a usar')
    
    # Comando para predecir
    predict_parser = subparsers.add_parser('predict', help='Realizar predicciones con modelo single-label')
    predict_parser.add_argument('--model', type=str, default='results/models/singlelabel/densenet_singlelabel_model.pth',
                               help='Ruta al modelo entrenado')
    predict_parser.add_argument('--image', type=str, help='Ruta a una imagen individual')
    predict_parser.add_argument('--directory', type=str, help='Directorio con imágenes para procesar')
    predict_parser.add_argument('--output', type=str, help='Archivo de salida para guardar resultados')
    predict_parser.add_argument('--visualize', action='store_true',
                               help='Mostrar visualización de la predicción')
    predict_parser.add_argument('--device', type=str, default='auto',
                               choices=['auto', 'cuda', 'cpu'],
                               help='Dispositivo a usar')
    
    # Comando para evaluar
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluar modelo single-label')
    evaluate_parser.add_argument('--model', type=str, default='results/models/singlelabel/densenet_singlelabel_model.pth',
                                help='Ruta al modelo entrenado')
    evaluate_parser.add_argument('--data_dir', type=str, default='data_diseases',
                                help='Directorio con datos de prueba')
    evaluate_parser.add_argument('--batch_size', type=int, default=16,
                                help='Tamaño del lote para evaluación')
    evaluate_parser.add_argument('--device', type=str, default='auto',
                                choices=['auto', 'cuda', 'cpu'],
                                help='Dispositivo a usar')
    
    # Comando para mostrar información
    info_parser = subparsers.add_parser('info', help='Mostrar información del sistema single-label')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Ejecutar comando correspondiente
    if args.command == 'train':
        try:
            from .train_singlelabel import main as train_main
        except ImportError:
            from train_singlelabel import main as train_main
        sys.argv = ['train_singlelabel.py'] + [
            f'--data_dir={args.data_dir}',
            f'--batch_size={args.batch_size}',
            f'--num_epochs={args.num_epochs}',
            f'--learning_rate={args.learning_rate}',
            f'--fine_tune_epochs={args.fine_tune_epochs}',
            f'--fine_tune_lr={args.fine_tune_lr}',
            f'--loss_type={args.loss_type}',
            f'--gamma={args.gamma}',
            f'--output_dir={args.output_dir}',
            f'--device={args.device}'
        ]
        if args.alpha is not None:
            sys.argv.append(f'--alpha={args.alpha}')
        if args.freeze_backbone:
            sys.argv.append('--freeze_backbone')
        train_main()
    
    elif args.command == 'predict':
        try:
            from .predict_singlelabel import main as predict_main
        except ImportError:
            from predict_singlelabel import main as predict_main
        sys.argv = ['predict_singlelabel.py'] + [
            f'--model={args.model}',
            f'--device={args.device}'
        ]
        if args.image:
            sys.argv.append(f'--image={args.image}')
        if args.directory:
            sys.argv.append(f'--directory={args.directory}')
        if args.output:
            sys.argv.append(f'--output={args.output}')
        if args.visualize:
            sys.argv.append('--visualize')
        predict_main()
    
    elif args.command == 'evaluate':
        import torch
        try:
            from .dataset import create_data_loaders
            from .model import create_model
            from .metrics import evaluate_model_single_label, SingleLabelMetrics
            from .train_singlelabel import get_transforms
        except ImportError:
            from dataset import create_data_loaders
            from model import create_model
            from metrics import evaluate_model_single_label, SingleLabelMetrics
            from train_singlelabel import get_transforms
        
        # Configurar dispositivo
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        
        print("📊 EVALUACIÓN DE MODELO SINGLE-LABEL")
        print("="*60)
        
        # Cargar modelo
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        class_names = checkpoint['class_names']
        num_classes = checkpoint['num_classes']
        
        model = create_model(num_classes=num_classes, pretrained=False, freeze_backbone=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✅ Modelo cargado: {args.model}")
        print(f"📊 Clases: {class_names}")
        
        # Crear DataLoader de prueba
        train_transform, val_transform = get_transforms()
        _, _, test_loader, _ = create_data_loaders(
            data_dir=args.data_dir,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=args.batch_size,
            include_normal=True,
            num_workers=0
        )
        
        # Evaluar
        test_metrics, _, _, _ = evaluate_model_single_label(
            model, test_loader, class_names, device
        )
        
        # Imprimir métricas
        metrics_calculator = SingleLabelMetrics(class_names)
        metrics_calculator.print_metrics(test_metrics)
        
        # Guardar matriz de confusión
        import matplotlib.pyplot as plt
        save_path = os.path.join(os.path.dirname(args.model), 'confusion_matrix.png')
        metrics_calculator.plot_confusion_matrix(test_metrics, save_path)
    
    elif args.command == 'info':
        print("="*70)
        print("📋 INFORMACIÓN DEL SISTEMA SINGLE-LABEL")
        print("="*70)
        
        print("\n🏥 Clases soportadas:")
        classes = ['Neumonía', 'Atelectasia', 'Edema', 'Tuberculosis', 
                  'COVID-19', 'Normal', 'Nodules', 'Mass']
        for i, cls in enumerate(classes, 1):
            print(f"  {i}. {cls}")
        
        print("\n🔧 Características del modelo:")
        print("  • Arquitectura: DenseNet-121")
        print("  • Transfer Learning: Sí")
        print("  • Activación: Softmax (single-label)")
        print("  • Función de pérdida: CrossEntropyLoss")
        print("  • Una sola clase por imagen")
        
        print("\n📁 Estructura de directorios:")
        print("  • Datos: data_diseases/")
        print("  • Modelos: results/models/singlelabel/")
        print("  • Código: src/models/singlelabel/")
        
        print("\n🚀 Comandos principales:")
        print("  • Entrenar: python src/models/singlelabel/main_singlelabel.py train")
        print("  • Predecir: python src/models/singlelabel/main_singlelabel.py predict --image imagen.jpg")
        print("  • Evaluar: python src/models/singlelabel/main_singlelabel.py evaluate")

if __name__ == "__main__":
    main()

