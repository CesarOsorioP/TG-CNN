"""
Script para preparar datos multi-label para el entrenamiento del modelo de enfermedades.
Organiza las imágenes filtradas en la estructura necesaria para clasificación multi-label.

"""

import os
import shutil
from pathlib import Path
import json
from collections import Counter
import argparse
import random

def create_disease_directories(base_dir):
    """
    Crear directorios para cada enfermedad y Normal, con subcarpetas train/val/test.
    
    Args:
        base_dir: Directorio base donde crear las carpetas
    """
    diseases = [
        'Neumonía',
        'Cáncer', 
        'Atelectasia',
        'Edema',
        'Tuberculosis',
        'COVID-19',
        'Normal'
    ]
    
    for disease in diseases:
        disease_dir = os.path.join(base_dir, disease)
        os.makedirs(disease_dir, exist_ok=True)
        
        # Crear subcarpetas train/val/test
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(disease_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            print(f"[OK] Directorio creado: {split_dir}")
    
    return diseases

def split_images_train_val_test(image_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Dividir lista de imágenes en train/val/test.
    
    Args:
        image_files: Lista de archivos de imagen
        train_ratio: Proporción para entrenamiento (default: 0.7)
        val_ratio: Proporción para validación (default: 0.15)
        test_ratio: Proporción para prueba (default: 0.15)
    
    Returns:
        tuple: (train_files, val_files, test_files)
    """
    # Mezclar aleatoriamente
    random.shuffle(image_files)
    
    total_files = len(image_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    return train_files, val_files, test_files

def copy_filtered_images(source_dir, target_dir, disease_mapping):
    """
    Copiar imágenes filtradas a la estructura multi-label con división train/val/test.
    
    Args:
        source_dir: Directorio con imágenes filtradas
        target_dir: Directorio destino para estructura multi-label
        disease_mapping: Mapeo de carpetas fuente a nombres de enfermedades
    """
    copied_files = {}
    
    for source_folder, disease_name in disease_mapping.items():
        source_path = os.path.join(source_dir, source_folder)
        
        if not os.path.exists(source_path):
            print(f"[WARN] Directorio fuente no encontrado: {source_path}")
            continue
        
        print(f"[INFO] Procesando {source_folder} -> {disease_name}")
        
        # Obtener lista de archivos de imagen
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file in os.listdir(source_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        if not image_files:
            print(f"[WARN] No se encontraron imágenes en: {source_path}")
            continue
        
        # Dividir en train/val/test
        train_files, val_files, test_files = split_images_train_val_test(image_files)
        
        # Copiar archivos a cada subcarpeta
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        total_copied = 0
        for split_name, files in splits.items():
            split_dir = os.path.join(target_dir, disease_name, split_name)
            split_count = 0
            
            for file in files:
                source_file = os.path.join(source_path, file)
                target_file = os.path.join(split_dir, file)
                
                try:
                    shutil.copy2(source_file, target_file)
                    split_count += 1
                except Exception as e:
                    print(f"[ERROR] Error copiando {file} a {split_name}: {e}")
            
            total_copied += split_count
            print(f"  [{split_name}]: {split_count} imágenes")
        
        copied_files[disease_name] = total_copied
        print(f"[OK] {disease_name}: {total_copied} imágenes copiadas en total")
    
    return copied_files

def create_data_summary(data_dir, copied_files):
    """
    Crear resumen de los datos preparados.
    
    Args:
        data_dir: Directorio con datos multi-label
        copied_files: Diccionario con archivos copiados por enfermedad
    """
    summary = {
        'total_diseases': len(copied_files),
        'diseases': {},
        'total_images': 0,
        'preparation_timestamp': str(Path().cwd())
    }
    
    for disease, count in copied_files.items():
        disease_dir = os.path.join(data_dir, disease)
        actual_files = len([f for f in os.listdir(disease_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        
        summary['diseases'][disease] = {
            'expected_files': count,
            'actual_files': actual_files,
            'directory': disease_dir
        }
        summary['total_images'] += actual_files
    
    return summary

def save_data_summary(summary, output_file):
    """
    Guardar resumen de datos en archivo JSON.
    
    Args:
        summary: Diccionario con resumen de datos
        output_file: Archivo donde guardar el resumen
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Resumen guardado en: {output_file}")

def print_data_summary(summary):
    """
    Imprimir resumen de datos preparados.
    
    Args:
        summary: Diccionario con resumen de datos
    """
    print("\n" + "="*60)
    print("RESUMEN DE DATOS PREPARADOS")
    print("="*60)
    print(f"Total de enfermedades: {summary['total_diseases']}")
    print(f"Total de imágenes: {summary['total_images']:,}")
    print("\nDetalle por enfermedad:")
    
    for disease, info in summary['diseases'].items():
        print(f"  {disease}:")
        print(f"    Directorio: {info['directory']}")
        print(f"    Imágenes: {info['actual_files']:,}")
        if info['expected_files'] != info['actual_files']:
            print(f"    [WARN] Esperadas: {info['expected_files']:,}")

def main():
    """
    Función principal para preparar datos multi-label.
    """
    parser = argparse.ArgumentParser(description='Preparar datos para modelo multi-label')
    parser.add_argument('--source_dir', type=str, 
                       default='scripts/PredicciónImágenes',
                       help='Directorio con imágenes filtradas')
    parser.add_argument('--target_dir', type=str, 
                       default='data_diseases',
                       help='Directorio destino para datos multi-label')
    parser.add_argument('--summary_file', type=str,
                       default='results/models/multilabel/data_summary.json',
                       help='Archivo para guardar resumen de datos')
    
    args = parser.parse_args()
    
    print("PREPARACION DE DATOS MULTI-LABEL")
    print("="*50)
    print(f"Directorio fuente: {args.source_dir}")
    print(f"Directorio destino: {args.target_dir}")
    print(f"Archivo de resumen: {args.summary_file}")
    
    # Mapeo de carpetas fuente a nombres de enfermedades
    disease_mapping = {
        'Neumonía': 'Neumonía',
        'Cancer': 'Cáncer',
        'Atelectasia': 'Atelectasia', 
        'Edema': 'Edema',
        'Tuberculosis': 'Tuberculosis',
        'Covid-19': 'COVID-19',
        'Normal': 'Normal'
    }
    
    # Crear directorios de enfermedades
    print("\n[INFO] Creando directorios de enfermedades...")
    diseases = create_disease_directories(args.target_dir)
    
    # Copiar imágenes filtradas
    print("\n[INFO] Copiando imágenes filtradas...")
    copied_files = copy_filtered_images(args.source_dir, args.target_dir, disease_mapping)
    
    # Crear resumen de datos
    print("\n[INFO] Creando resumen de datos...")
    summary = create_data_summary(args.target_dir, copied_files)
    
    # Guardar resumen
    os.makedirs(os.path.dirname(args.summary_file), exist_ok=True)
    save_data_summary(summary, args.summary_file)
    
    # Imprimir resumen
    print_data_summary(summary)
    
    print("\n[OK] Preparacion de datos completada!")
    print(f"Los datos estan listos en: {args.target_dir}")
    print(f"Resumen guardado en: {args.summary_file}")

if __name__ == "__main__":
    main()
