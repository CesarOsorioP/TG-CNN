"""
Script para visualizar los umbrales adaptativos (optimal_thresholds) 
guardados en un modelo entrenado multi-label.
"""

import torch
import argparse
import os
import sys

# Forzar utf-8 para la salida en consola de Windows y evitar UnicodeEncodeError
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')


def show_thresholds(model_path):
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encontró el archivo del modelo en {model_path}")
        return

    print(f"🔄 Cargando checkpoint desde: {model_path}")
    try:
        # Cargamos en CPU para que sea rápido y funcione en cualquier máquina
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        disease_names = checkpoint.get('disease_names', [])
        optimal_thresholds = checkpoint.get('optimal_thresholds', None)
        thresholds_comparison = checkpoint.get('thresholds_comparison', None)

        if not optimal_thresholds:
            print("⚠️ Este modelo no contiene umbrales adaptativos ('optimal_thresholds') guardados.")
            return

        print("\n" + "="*60)
        print("🎯 UMBRALES ADAPTATIVOS POR ENFERMEDAD")
        print("="*60)
        
        # Mostrar de forma tabulada
        print(f"{'Enfermedad':<20} | {'Umbral Óptimo':<15}")
        print("-" * 40)
        for disease in disease_names:
            threshold = optimal_thresholds.get(disease, "No encontrado")
            if isinstance(threshold, float):
                print(f"{disease:<20} | {threshold:.4f}")
            else:
                print(f"{disease:<20} | {threshold}")
                
        if thresholds_comparison:
            print("\n" + "="*60)
            print("📈 COMPARACIÓN DE RENDIMIENTO (F1-Score) VS UMBRAL FIJO 0.5")
            print("="*60)
            print(f"{'Enfermedad':<20} | {'F1 Óptimo':<10} | {'F1 Fijo (0.5)':<14} | {'Mejora %':<10}")
            print("-" * 65)
            for disease in disease_names:
                comp = thresholds_comparison.get(disease, {})
                if comp:
                    f1_opt = comp.get('f1_optimal', 0.0)
                    f1_fix = comp.get('f1_fixed', 0.0)
                    imp = comp.get('improvement', 0.0)
                    print(f"{disease:<20} | {f1_opt:<10.4f} | {f1_fix:<14.4f} | {imp:>+7.2f}%")

    except Exception as e:
        print(f"❌ Error al leer el archivo del modelo: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ver los umbrales adaptativos de un modelo multi-label.')
    parser.add_argument(
        '--model', 
        type=str, 
        default='results/models/multilabel/densenet_multilabel_model.pth',
        help='Ruta al archivo .pth del modelo multi-label'
    )
    args = parser.parse_args()
    
    show_thresholds(args.model)
