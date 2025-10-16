"""
Script de evaluación para modelo multi-label de clasificación de enfermedades.
Evalúa el rendimiento del modelo en conjuntos de prueba con métricas específicas multi-label.

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Importar módulos del proyecto
from dataset import create_data_loaders
from model import create_model
from metrics import MultiLabelMetrics, evaluate_model_multi_label

def evaluate_model_comprehensive(model, test_loader, disease_names, device='cuda', threshold=0.5):
    """
    Evaluación comprehensiva del modelo multi-label.
    
    Args:
        model: Modelo a evaluar
        test_loader: DataLoader con datos de prueba
        disease_names: Lista de nombres de enfermedades
        device: Dispositivo a usar
        threshold: Umbral para predicciones binarias
        
    Returns:
        dict: Resultados de evaluación comprehensiva
    """
    print("🔄 Iniciando evaluación comprehensiva...")
    
    # Evaluación básica
    metrics, predictions, targets, probabilities = evaluate_model_multi_label(
        model, test_loader, disease_names, device, threshold
    )
    
    # Análisis adicional
    analysis = {
        'basic_metrics': metrics,
        'detailed_analysis': {},
        'confusion_matrices': {},
        'roc_curves': {},
        'error_analysis': {}
    }
    
    # Análisis detallado por enfermedad
    print("📊 Analizando rendimiento por enfermedad...")
    for i, disease in enumerate(disease_names):
        disease_metrics = {
            'precision': metrics['per_class_metrics'][disease]['precision'],
            'recall': metrics['per_class_metrics'][disease]['recall'],
            'f1_score': metrics['per_class_metrics'][disease]['f1'],
            'support': metrics['per_class_metrics'][disease]['support'],
            'auc': metrics['auc_per_class'][i] if 'auc_per_class' in metrics else None
        }
        analysis['detailed_analysis'][disease] = disease_metrics
    
    # Análisis de errores
    print("🔍 Analizando errores de clasificación...")
    error_analysis = analyze_classification_errors(targets, predictions, disease_names)
    analysis['error_analysis'] = error_analysis
    
    # Análisis de confianza
    print("📈 Analizando distribución de confianza...")
    confidence_analysis = analyze_confidence_distribution(probabilities, targets, disease_names)
    analysis['confidence_analysis'] = confidence_analysis
    
    return analysis

def analyze_classification_errors(y_true, y_pred, disease_names):
    """
    Analizar errores de clasificación.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        disease_names: Lista de nombres de enfermedades
        
    Returns:
        dict: Análisis de errores
    """
    errors = {
        'false_positives': {},
        'false_negatives': {},
        'confusion_pairs': {},
        'error_rate_by_disease': {}
    }
    
    for i, disease in enumerate(disease_names):
        # Falsos positivos (predicho pero no verdadero)
        fp_mask = (y_pred[:, i] == 1) & (y_true[:, i] == 0)
        errors['false_positives'][disease] = np.sum(fp_mask)
        
        # Falsos negativos (verdadero pero no predicho)
        fn_mask = (y_true[:, i] == 1) & (y_pred[:, i] == 0)
        errors['false_negatives'][disease] = np.sum(fn_mask)
        
        # Tasa de error por enfermedad
        total_true = np.sum(y_true[:, i])
        total_pred = np.sum(y_pred[:, i])
        if total_true > 0:
            errors['error_rate_by_disease'][disease] = {
                'false_positive_rate': errors['false_positives'][disease] / (len(y_true) - total_true),
                'false_negative_rate': errors['false_negatives'][disease] / total_true
            }
    
    return errors

def analyze_confidence_distribution(probabilities, targets, disease_names):
    """
    Analizar distribución de confianza.
    
    Args:
        probabilities: Probabilidades de predicción
        targets: Etiquetas verdaderas
        disease_names: Lista de nombres de enfermedades
        
    Returns:
        dict: Análisis de confianza
    """
    confidence_analysis = {
        'overall_stats': {},
        'by_disease': {},
        'confidence_thresholds': {}
    }
    
    # Estadísticas generales
    all_probs = probabilities.flatten()
    confidence_analysis['overall_stats'] = {
        'mean': np.mean(all_probs),
        'std': np.std(all_probs),
        'min': np.min(all_probs),
        'max': np.max(all_probs),
        'median': np.median(all_probs)
    }
    
    # Análisis por enfermedad
    for i, disease in enumerate(disease_names):
        disease_probs = probabilities[:, i]
        disease_targets = targets[:, i]
        
        # Probabilidades cuando la enfermedad está presente
        present_probs = disease_probs[disease_targets == 1]
        # Probabilidades cuando la enfermedad no está presente
        absent_probs = disease_probs[disease_targets == 0]
        
        confidence_analysis['by_disease'][disease] = {
            'present_mean': np.mean(present_probs) if len(present_probs) > 0 else 0,
            'absent_mean': np.mean(absent_probs) if len(absent_probs) > 0 else 0,
            'separation': np.mean(present_probs) - np.mean(absent_probs) if len(present_probs) > 0 and len(absent_probs) > 0 else 0
        }
    
    # Análisis de umbrales de confianza
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        pred_at_threshold = (probabilities > threshold).astype(int)
        f1_macro = f1_score(targets, pred_at_threshold, average='macro', zero_division=0)
        f1_micro = f1_score(targets, pred_at_threshold, average='micro', zero_division=0)
        
        confidence_analysis['confidence_thresholds'][f'threshold_{threshold}'] = {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro
        }
    
    return confidence_analysis

def plot_evaluation_results(analysis, disease_names, save_dir=None):
    """
    Crear gráficos de evaluación.
    
    Args:
        analysis: Resultados de análisis
        disease_names: Lista de nombres de enfermedades
        save_dir: Directorio para guardar gráficos
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Gráfico 1: Métricas por enfermedad
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    diseases = list(disease_names)
    precision = [analysis['detailed_analysis'][d]['precision'] for d in diseases]
    recall = [analysis['detailed_analysis'][d]['recall'] for d in diseases]
    f1_scores = [analysis['detailed_analysis'][d]['f1_score'] for d in diseases]
    
    x = np.arange(len(diseases))
    width = 0.25
    
    ax1.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax1.bar(x, recall, width, label='Recall', alpha=0.8)
    ax1.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax1.set_xlabel('Enfermedades')
    ax1.set_ylabel('Puntuación')
    ax1.set_title('Métricas por Enfermedad')
    ax1.set_xticks(x)
    ax1.set_xticklabels(diseases, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Análisis de errores
    fp_counts = [analysis['error_analysis']['false_positives'][d] for d in diseases]
    fn_counts = [analysis['error_analysis']['false_negatives'][d] for d in diseases]
    
    ax2.bar(x - width/2, fp_counts, width, label='Falsos Positivos', alpha=0.8, color='red')
    ax2.bar(x + width/2, fn_counts, width, label='Falsos Negativos', alpha=0.8, color='blue')
    
    ax2.set_xlabel('Enfermedades')
    ax2.set_ylabel('Número de Errores')
    ax2.set_title('Análisis de Errores por Enfermedad')
    ax2.set_xticks(x)
    ax2.set_xticklabels(diseases, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plot_path = os.path.join(save_dir, 'evaluation_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico de métricas guardado en: {plot_path}")
    
    plt.show()
    
    # Gráfico 3: Distribución de confianza
    if 'confidence_analysis' in analysis:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        f1_macro_scores = [analysis['confidence_analysis']['confidence_thresholds'][f'threshold_{t}']['f1_macro'] 
                          for t in thresholds]
        f1_micro_scores = [analysis['confidence_analysis']['confidence_thresholds'][f'threshold_{t}']['f1_micro'] 
                          for t in thresholds]
        
        ax.plot(thresholds, f1_macro_scores, 'o-', label='F1-Macro', linewidth=2, markersize=6)
        ax.plot(thresholds, f1_micro_scores, 's-', label='F1-Micro', linewidth=2, markersize=6)
        
        ax.set_xlabel('Umbral de Confianza')
        ax.set_ylabel('F1-Score')
        ax.set_title('Rendimiento vs Umbral de Confianza')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            plot_path = os.path.join(save_dir, 'confidence_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📈 Gráfico de confianza guardado en: {plot_path}")
        
        plt.show()

def print_evaluation_summary(analysis, disease_names):
    """
    Imprimir resumen de evaluación.
    
    Args:
        analysis: Resultados de análisis
        disease_names: Lista de nombres de enfermedades
    """
    print("\n" + "="*80)
    print("📊 RESUMEN DE EVALUACIÓN MULTI-LABEL")
    print("="*80)
    
    # Métricas básicas
    basic_metrics = analysis['basic_metrics']
    print(f"\n🎯 MÉTRICAS GENERALES:")
    print(f"  Hamming Loss: {basic_metrics['hamming_loss']:.4f}")
    print(f"  Exact Match Ratio: {basic_metrics['exact_match']:.4f}")
    print(f"  F1-Score (Macro): {basic_metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Micro): {basic_metrics['f1_micro']:.4f}")
    print(f"  Jaccard Score (Macro): {basic_metrics['jaccard_macro']:.4f}")
    
    # Métricas por enfermedad
    print(f"\n🏥 RENDIMIENTO POR ENFERMEDAD:")
    print(f"{'Enfermedad':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 70)
    
    for disease in disease_names:
        metrics = analysis['detailed_analysis'][disease]
        print(f"{disease:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
              f"{metrics['f1_score']:<10.4f} {metrics['support']:<8}")
    
    # Análisis de errores
    print(f"\n❌ ANÁLISIS DE ERRORES:")
    error_analysis = analysis['error_analysis']
    print(f"{'Enfermedad':<15} {'Falsos Pos.':<12} {'Falsos Neg.':<12} {'Error Rate':<12}")
    print("-" * 60)
    
    for disease in disease_names:
        fp = error_analysis['false_positives'][disease]
        fn = error_analysis['false_negatives'][disease]
        error_rate = fp + fn
        print(f"{disease:<15} {fp:<12} {fn:<12} {error_rate:<12}")
    
    # Análisis de confianza
    if 'confidence_analysis' in analysis:
        print(f"\n📈 ANÁLISIS DE CONFIANZA:")
        conf_stats = analysis['confidence_analysis']['overall_stats']
        print(f"  Confianza promedio: {conf_stats['mean']:.4f}")
        print(f"  Desviación estándar: {conf_stats['std']:.4f}")
        print(f"  Rango: [{conf_stats['min']:.4f}, {conf_stats['max']:.4f}]")

def main():
    """
    Función principal para evaluar el modelo multi-label.
    """
    parser = argparse.ArgumentParser(description='Evaluar modelo multi-label de enfermedades')
    parser.add_argument('--model', type=str, default='results/models/multilabel/densenet_multilabel_model.pth',
                       help='Ruta al modelo entrenado')
    parser.add_argument('--data_dir', type=str, default='data_diseases',
                       help='Directorio con datos de prueba')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Tamaño del lote para evaluación')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Umbral para predicciones binarias')
    parser.add_argument('--output_dir', type=str, default='results/analysis/multilabel',
                       help='Directorio para guardar resultados de evaluación')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                       default='auto', help='Dispositivo a usar')
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🔍 EVALUACIÓN DE MODELO MULTI-LABEL")
    print("="*60)
    print(f"🤖 Modelo: {args.model}")
    print(f"📁 Datos: {args.data_dir}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"🎯 Umbral: {args.threshold}")
    print(f"💻 Dispositivo: {device}")
    
    # Verificar archivos
    if not os.path.exists(args.model):
        print(f"❌ Error: Modelo no encontrado en {args.model}")
        return 1
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Directorio de datos no encontrado en {args.data_dir}")
        return 1
    
    # Cargar modelo
    print(f"\n🔄 Cargando modelo...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    disease_names = checkpoint['disease_names']
    num_diseases = checkpoint['num_diseases']
    
    model = create_model(
        num_diseases=num_diseases,
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Modelo cargado - {num_diseases} enfermedades")
    
    # Preparar datos
    print(f"\n📊 Preparando datos de prueba...")
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    _, _, test_loader, _ = create_data_loaders(
        data_dir=args.data_dir,
        train_transform=val_transform,  # Usar misma transformación
        val_transform=val_transform,
        batch_size=args.batch_size,
        include_normal=False,
        num_workers=4
    )
    
    # Evaluar modelo
    print(f"\n🔍 Evaluando modelo...")
    analysis = evaluate_model_comprehensive(
        model, test_loader, disease_names, device, args.threshold
    )
    
    # Imprimir resultados
    print_evaluation_summary(analysis, disease_names)
    
    # Crear gráficos
    print(f"\n📈 Generando gráficos...")
    plot_evaluation_results(analysis, disease_names, args.output_dir)
    
    # Guardar resultados
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Guardar análisis completo
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        # Convertir numpy arrays a listas para JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Aplicar conversión recursivamente
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(item) for item in d]
            else:
                return convert_numpy(d)
        
        analysis_serializable = recursive_convert(analysis)
        json.dump(analysis_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Evaluación completada!")
    print(f"📁 Resultados guardados en: {args.output_dir}")
    print(f"📄 Análisis detallado: {results_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())
