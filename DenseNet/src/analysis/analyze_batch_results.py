"""
Script para analizar resultados de procesamiento en lote.
Genera reportes detallados y visualizaciones.

Autor: Asistente AI
Fecha: 2024
"""

import json
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
from pathlib import Path
import argparse

def load_batch_results(results_dir):
    """
    Cargar resultados de procesamiento en lote.
    
    Args:
        results_dir: Directorio con los resultados
        
    Returns:
        tuple: (summary, results_df)
    """
    # Buscar archivos de resultados
    summary_files = list(Path(results_dir).glob("batch_prediction_summary_*.json"))
    csv_files = list(Path(results_dir).glob("batch_prediction_results_*.csv"))
    
    if not summary_files or not csv_files:
        print(f"❌ No se encontraron archivos de resultados en: {results_dir}")
        return None, None
    
    # Cargar el archivo más reciente
    latest_summary = max(summary_files, key=os.path.getctime)
    latest_csv = max(csv_files, key=os.path.getctime)
    
    print(f"📄 Cargando resumen: {latest_summary}")
    print(f"📊 Cargando detalles: {latest_csv}")
    
    # Cargar resumen
    with open(latest_summary, 'r') as f:
        summary = json.load(f)
    
    # Cargar resultados detallados
    results_df = pd.read_csv(latest_csv)
    
    return summary, results_df

def create_visualizations(summary, results_df, output_dir):
    """Crear visualizaciones de los resultados."""
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Análisis de Resultados - Clasificación de Radiografías de Tórax', 
                 fontsize=16, fontweight='bold')
    
    # 1. Distribución de clases
    ax1 = axes[0, 0]
    classes = ['Radiografías de Tórax', 'Otras Imágenes']
    counts = [summary['chest_xray_count'], summary['other_count']]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax1.bar(classes, counts, color=colors, alpha=0.8)
    ax1.set_title('Distribución de Clases', fontweight='bold')
    ax1.set_ylabel('Número de Imágenes')
    
    # Agregar valores en las barras
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Distribución de confianza
    ax2 = axes[0, 1]
    confidences = results_df['confidence'].dropna()
    
    ax2.hist(confidences, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
    ax2.axvline(confidences.mean(), color='red', linestyle='--', 
               label=f'Promedio: {confidences.mean():.3f}')
    ax2.set_title('Distribución de Confianza', fontweight='bold')
    ax2.set_xlabel('Confianza')
    ax2.set_ylabel('Frecuencia')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confianza por clase
    ax3 = axes[0, 2]
    chest_conf = results_df[results_df['predicted_class'] == 'chest_xray']['confidence']
    other_conf = results_df[results_df['predicted_class'] == 'other_images']['confidence']
    
    ax3.hist([chest_conf, other_conf], bins=20, alpha=0.7, 
            label=['Radiografías', 'Otras'], color=['#2ecc71', '#e74c3c'])
    ax3.set_title('Confianza por Clase', fontweight='bold')
    ax3.set_xlabel('Confianza')
    ax3.set_ylabel('Frecuencia')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Progreso de procesamiento
    ax4 = axes[1, 0]
    total = summary['total_images']
    successful = summary['successful_predictions']
    failed = summary['failed_predictions']
    
    labels = ['Exitosos', 'Fallidos']
    sizes = [successful, failed]
    colors = ['#2ecc71', '#e74c3c']
    
    wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90)
    ax4.set_title('Tasa de Éxito', fontweight='bold')
    
    # 5. Distribución de probabilidades
    ax5 = axes[1, 1]
    prob_chest = results_df['prob_chest_xray'].dropna()
    prob_other = results_df['prob_other_images'].dropna()
    
    ax5.scatter(prob_chest, prob_other, alpha=0.6, s=20)
    ax5.plot([0, 1], [1, 0], 'r--', alpha=0.5, label='Línea de decisión')
    ax5.set_title('Probabilidades: Tórax vs Otras', fontweight='bold')
    ax5.set_xlabel('Probabilidad Radiografía de Tórax')
    ax5.set_ylabel('Probabilidad Otras Imágenes')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Estadísticas de confianza
    ax6 = axes[1, 2]
    conf_stats = [
        summary['high_confidence_count'],
        summary['medium_confidence_count'],
        summary['low_confidence_count']
    ]
    conf_labels = ['Alta (>90%)', 'Media (70-90%)', 'Baja (<70%)']
    conf_colors = ['#27ae60', '#f39c12', '#e74c3c']
    
    bars = ax6.bar(conf_labels, conf_stats, color=conf_colors, alpha=0.8)
    ax6.set_title('Distribución de Confianza', fontweight='bold')
    ax6.set_ylabel('Número de Imágenes')
    
    # Agregar valores en las barras
    for bar, count in zip(bars, conf_stats):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + max(conf_stats)*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar visualización
    output_file = os.path.join(output_dir, 'batch_analysis_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Visualización guardada en: {output_file}")
    
    plt.show()

def generate_detailed_report(summary, results_df, output_dir):
    """Generar reporte detallado en texto."""
    
    report_file = os.path.join(output_dir, 'detailed_analysis_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DETALLADO - CLASIFICACIÓN DE RADIOGRAFÍAS DE TÓRAX\n")
        f.write("="*80 + "\n\n")
        
        # Resumen general
        f.write("📊 RESUMEN GENERAL\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total de imágenes procesadas: {summary['total_images']:,}\n")
        f.write(f"Predicciones exitosas: {summary['successful_predictions']:,}\n")
        f.write(f"Predicciones fallidas: {summary['failed_predictions']:,}\n")
        f.write(f"Tasa de éxito: {summary['successful_predictions']/summary['total_images']*100:.1f}%\n\n")
        
        # Distribución de clases
        f.write("🏥 DISTRIBUCIÓN DE CLASES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Radiografías de tórax: {summary['chest_xray_count']:,} ({summary['chest_xray_percentage']:.1f}%)\n")
        f.write(f"Otras imágenes: {summary['other_count']:,} ({summary['other_percentage']:.1f}%)\n\n")
        
        # Estadísticas de confianza
        f.write("📈 ESTADÍSTICAS DE CONFIANZA\n")
        f.write("-" * 40 + "\n")
        f.write(f"Confianza promedio: {summary['average_confidence']:.3f}\n")
        f.write(f"Alta confianza (>90%): {summary['high_confidence_count']:,}\n")
        f.write(f"Confianza media (70-90%): {summary['medium_confidence_count']:,}\n")
        f.write(f"Baja confianza (<70%): {summary['low_confidence_count']:,}\n\n")
        
        # Análisis de errores
        if summary['failed_predictions'] > 0:
            f.write("❌ ANÁLISIS DE ERRORES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Imágenes con error: {summary['failed_predictions']:,}\n")
            f.write(f"Tasa de error: {summary['failed_predictions']/summary['total_images']*100:.1f}%\n\n")
        
        # Recomendaciones
        f.write("💡 RECOMENDACIONES\n")
        f.write("-" * 40 + "\n")
        
        if summary['chest_xray_percentage'] > 80:
            f.write("• El dataset parece estar muy sesgado hacia radiografías de tórax\n")
            f.write("• Considera balancear el dataset con más imágenes de otras categorías\n")
        elif summary['chest_xray_percentage'] < 20:
            f.write("• El dataset tiene pocas radiografías de tórax\n")
            f.write("• Considera agregar más radiografías para mejor entrenamiento\n")
        else:
            f.write("• El dataset tiene una distribución balanceada\n")
            f.write("• Bueno para entrenamiento y validación\n")
        
        if summary['average_confidence'] > 0.9:
            f.write("• El modelo muestra alta confianza en las predicciones\n")
            f.write("• Los resultados son muy confiables\n")
        elif summary['average_confidence'] < 0.7:
            f.write("• El modelo muestra baja confianza en las predicciones\n")
            f.write("• Considera reentrenar o ajustar el modelo\n")
        else:
            f.write("• El modelo muestra confianza moderada en las predicciones\n")
            f.write("• Los resultados son aceptables\n")
        
        if summary['low_confidence_count'] > summary['successful_predictions'] * 0.3:
            f.write("• Muchas predicciones tienen baja confianza\n")
            f.write("• Revisa manualmente las imágenes con baja confianza\n")
    
    print(f"📄 Reporte detallado guardado en: {report_file}")

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Analizar resultados de procesamiento en lote')
    parser.add_argument('--results_dir', type=str, default='quick_batch_results',
                       help='Directorio con los resultados')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                       help='Directorio para guardar análisis')
    
    args = parser.parse_args()
    
    print("🔍 ANÁLISIS DE RESULTADOS DE PROCESAMIENTO EN LOTE")
    print("="*60)
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar resultados
    summary, results_df = load_batch_results(args.results_dir)
    
    if summary is None or results_df is None:
        return 1
    
    print(f"✅ Resultados cargados exitosamente")
    print(f"📊 Total de imágenes: {summary['total_images']:,}")
    
    # Generar visualizaciones
    print("\n📊 Generando visualizaciones...")
    create_visualizations(summary, results_df, args.output_dir)
    
    # Generar reporte detallado
    print("\n📄 Generando reporte detallado...")
    generate_detailed_report(summary, results_df, args.output_dir)
    
    print(f"\n✅ Análisis completado! Resultados en: {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())
