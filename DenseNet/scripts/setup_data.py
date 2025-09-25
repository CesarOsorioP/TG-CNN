"""
Script para ayudar a organizar y preparar los datos para el entrenamiento.
Este script verifica la estructura de datos y proporciona estadísticas útiles.

Autor: Asistente AI
Fecha: 2024
"""

import os
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def check_data_structure(data_dir="data"):
    """
    Verificar la estructura de datos y mostrar estadísticas.
    
    Args:
        data_dir: Directorio con los datos
    """
    print("="*60)
    print("VERIFICACIÓN DE ESTRUCTURA DE DATOS")
    print("="*60)
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directorio '{data_dir}' no encontrado")
        print("\n📁 Creando estructura de directorios...")
        os.makedirs(os.path.join(data_dir, "chest_xray"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "other_images"), exist_ok=True)
        print("✅ Estructura creada. Ahora coloca tus imágenes en:")
        print(f"   - {data_dir}/chest_xray/     (radiografías de tórax)")
        print(f"   - {data_dir}/other_images/   (otras imágenes)")
        return False
    
    # Verificar subdirectorios
    chest_dir = os.path.join(data_dir, "chest_xray")
    other_dir = os.path.join(data_dir, "other_images")
    
    chest_exists = os.path.exists(chest_dir)
    other_exists = os.path.exists(other_dir)
    
    print(f"📁 Directorio base: {data_dir}")
    print(f"📁 chest_xray/: {'✅' if chest_exists else '❌'}")
    print(f"📁 other_images/: {'✅' if other_exists else '❌'}")
    
    if not chest_exists or not other_exists:
        print("\n❌ Estructura incompleta. Creando directorios faltantes...")
        if not chest_exists:
            os.makedirs(chest_dir, exist_ok=True)
            print(f"✅ Creado: {chest_dir}")
        if not other_exists:
            os.makedirs(other_dir, exist_ok=True)
            print(f"✅ Creado: {other_dir}")
        return False
    
    return True

def analyze_images(data_dir="data"):
    """
    Analizar las imágenes en el directorio de datos.
    
    Args:
        data_dir: Directorio con los datos
    """
    print("\n" + "="*60)
    print("ANÁLISIS DE IMÁGENES")
    print("="*60)
    
    chest_dir = os.path.join(data_dir, "chest_xray")
    other_dir = os.path.join(data_dir, "other_images")
    
    # Extensiones soportadas
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def count_images(directory):
        """Contar imágenes en un directorio."""
        if not os.path.exists(directory):
            return 0, [], []
        
        image_files = []
        image_sizes = []
        
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                ext = Path(file).suffix.lower()
                if ext in supported_extensions:
                    image_files.append(file_path)
                    
                    # Obtener tamaño de la imagen
                    try:
                        with Image.open(file_path) as img:
                            image_sizes.append(img.size)
                    except Exception as e:
                        print(f"⚠️  Error al procesar {file}: {e}")
        
        return len(image_files), image_files, image_sizes
    
    # Contar imágenes por clase
    chest_count, chest_files, chest_sizes = count_images(chest_dir)
    other_count, other_files, other_sizes = count_images(other_dir)
    
    total_images = chest_count + other_count
    
    print(f"📊 Estadísticas de imágenes:")
    print(f"   Radiografías de tórax: {chest_count}")
    print(f"   Otras imágenes: {other_count}")
    print(f"   Total: {total_images}")
    
    if total_images == 0:
        print("\n❌ No se encontraron imágenes. Asegúrate de que:")
        print("   - Las imágenes están en los directorios correctos")
        print("   - Las extensiones son válidas (.jpg, .png, etc.)")
        return
    
    # Análisis de tamaños
    if chest_sizes:
        chest_widths = [size[0] for size in chest_sizes]
        chest_heights = [size[1] for size in chest_sizes]
        print(f"\n📏 Radiografías de tórax:")
        print(f"   Ancho promedio: {np.mean(chest_widths):.0f}px")
        print(f"   Alto promedio: {np.mean(chest_heights):.0f}px")
        print(f"   Rango de ancho: {min(chest_widths)}-{max(chest_widths)}px")
        print(f"   Rango de alto: {min(chest_heights)}-{max(chest_heights)}px")
    
    if other_sizes:
        other_widths = [size[0] for size in other_sizes]
        other_heights = [size[1] for size in other_sizes]
        print(f"\n📏 Otras imágenes:")
        print(f"   Ancho promedio: {np.mean(other_widths):.0f}px")
        print(f"   Alto promedio: {np.mean(other_heights):.0f}px")
        print(f"   Rango de ancho: {min(other_widths)}-{max(other_widths)}px")
        print(f"   Rango de alto: {min(other_heights)}-{max(other_heights)}px")
    
    # Verificar balance de clases
    if chest_count > 0 and other_count > 0:
        ratio = chest_count / other_count
        print(f"\n⚖️  Balance de clases:")
        print(f"   Proporción chest_xray:other_images = {ratio:.2f}:1")
        
        if ratio < 0.5 or ratio > 2.0:
            print("   ⚠️  Advertencia: Clases desbalanceadas")
            print("   💡 Considera recopilar más imágenes de la clase minoritaria")
        else:
            print("   ✅ Clases bien balanceadas")
    
    # Recomendaciones
    print(f"\n💡 Recomendaciones:")
    if total_images < 200:
        print("   - Considera recopilar más imágenes (mínimo 100 por clase)")
    elif total_images < 500:
        print("   - Buen número de imágenes para empezar")
    else:
        print("   - Excelente cantidad de datos para entrenamiento")
    
    if chest_count == 0:
        print("   - ❌ No hay radiografías de tórax. Agrega imágenes a chest_xray/")
    if other_count == 0:
        print("   - ❌ No hay otras imágenes. Agrega imágenes a other_images/")
    
    return {
        'chest_count': chest_count,
        'other_count': other_count,
        'total_count': total_images,
        'chest_sizes': chest_sizes,
        'other_sizes': other_sizes
    }

def visualize_sample_images(data_dir="data", num_samples=4):
    """
    Visualizar muestras de imágenes de cada clase.
    
    Args:
        data_dir: Directorio con los datos
        num_samples: Número de muestras a mostrar por clase
    """
    print("\n" + "="*60)
    print("VISUALIZACIÓN DE MUESTRAS")
    print("="*60)
    
    chest_dir = os.path.join(data_dir, "chest_xray")
    other_dir = os.path.join(data_dir, "other_images")
    
    def get_sample_images(directory, num_samples):
        """Obtener muestras de imágenes de un directorio."""
        if not os.path.exists(directory):
            return []
        
        image_files = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                ext = Path(file).suffix.lower()
                if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
                    image_files.append(file_path)
        
        return image_files[:num_samples]
    
    chest_samples = get_sample_images(chest_dir, num_samples)
    other_samples = get_sample_images(other_dir, num_samples)
    
    if not chest_samples and not other_samples:
        print("❌ No se encontraron imágenes para mostrar")
        return
    
    # Crear visualización
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    # Mostrar radiografías de tórax
    for i in range(num_samples):
        ax = axes[0, i]
        if i < len(chest_samples):
            try:
                img = Image.open(chest_samples[i])
                ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
                ax.set_title(f"Radiografía {i+1}\n{os.path.basename(chest_samples[i])}")
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{str(e)[:20]}...", 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Sin imagen", ha='center', va='center', 
                   transform=ax.transAxes, color='gray')
        ax.axis('off')
    
    # Mostrar otras imágenes
    for i in range(num_samples):
        ax = axes[1, i]
        if i < len(other_samples):
            try:
                img = Image.open(other_samples[i])
                ax.imshow(img)
                ax.set_title(f"Otra imagen {i+1}\n{os.path.basename(other_samples[i])}")
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{str(e)[:20]}...", 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Sin imagen", ha='center', va='center', 
                   transform=ax.transAxes, color='gray')
        ax.axis('off')
    
    plt.suptitle("Muestras de Imágenes", fontsize=16)
    plt.tight_layout()
    plt.show()

def generate_data_report(data_dir="data", output_file="data_report.json"):
    """
    Generar reporte completo de los datos.
    
    Args:
        data_dir: Directorio con los datos
        output_file: Archivo de salida para el reporte
    """
    print("\n" + "="*60)
    print("GENERANDO REPORTE DE DATOS")
    print("="*60)
    
    # Verificar estructura
    structure_ok = check_data_structure(data_dir)
    
    # Analizar imágenes
    analysis = analyze_images(data_dir)
    
    # Crear reporte
    report = {
        'data_directory': data_dir,
        'structure_valid': structure_ok,
        'analysis': analysis,
        'recommendations': []
    }
    
    # Agregar recomendaciones
    if analysis:
        total = analysis['total_count']
        chest = analysis['chest_count']
        other = analysis['other_count']
        
        if total < 100:
            report['recommendations'].append("Recopilar más imágenes (mínimo 50 por clase)")
        elif total < 200:
            report['recommendations'].append("Considerar recopilar más imágenes para mejor rendimiento")
        
        if chest == 0:
            report['recommendations'].append("Agregar radiografías de tórax a chest_xray/")
        if other == 0:
            report['recommendations'].append("Agregar otras imágenes a other_images/")
        
        if chest > 0 and other > 0:
            ratio = chest / other
            if ratio < 0.5 or ratio > 2.0:
                report['recommendations'].append("Considerar balancear las clases")
    
    # Guardar reporte
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Reporte guardado en: {output_file}")
    return report

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preparar y analizar datos para entrenamiento')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directorio con los datos')
    parser.add_argument('--visualize', action='store_true',
                       help='Mostrar muestras de imágenes')
    parser.add_argument('--report', action='store_true',
                       help='Generar reporte de datos')
    parser.add_argument('--samples', type=int, default=4,
                       help='Número de muestras a visualizar')
    
    args = parser.parse_args()
    
    # Verificar estructura
    check_data_structure(args.data_dir)
    
    # Analizar imágenes
    analyze_images(args.data_dir)
    
    # Visualizar muestras
    if args.visualize:
        visualize_sample_images(args.data_dir, args.samples)
    
    # Generar reporte
    if args.report:
        generate_data_report(args.data_dir)

if __name__ == "__main__":
    main()
