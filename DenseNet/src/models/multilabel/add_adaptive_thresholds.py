"""
Script para calcular y guardar umbrales adaptativos (por clase) en un checkpoint
multi-label ya entrenado, SIN reentrenar el modelo.

Uso recomendado:
  python src/models/multilabel/add_adaptive_thresholds.py \
    --model results/models/multilabel/densenet_multilabel_third_model.pth \
    --data_dir data_diseases --batch_size 16 --device auto
"""

import os
import argparse
import torch

# Reutilizar utilidades del proyecto
from train_multilabel import (
    get_transforms,
    calculate_optimal_thresholds,
)
from dataset import create_data_loaders
from model import create_model


def main():
    parser = argparse.ArgumentParser(
        description="Calcular y guardar umbrales adaptativos en un checkpoint multi-label existente"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ruta al checkpoint .pth existente (se sobreescribirá con los umbrales)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_diseases",
        help="Directorio de datos con estructura train/val/test",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Tamaño de lote para el cálculo en validación",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Dispositivo a usar",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Crear copia de respaldo del .pth antes de sobreescribir",
    )

    args = parser.parse_args()

    # Configurar dispositivo
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not os.path.exists(args.model):
        print(f"❌ Checkpoint no encontrado: {args.model}")
        return 1

    print("=" * 70)
    print("🧪 CÁLCULO DE UMBRALES ADAPTATIVOS (SIN REENTRENAR)")
    print("=" * 70)
    print(f"📁 Checkpoint: {args.model}")
    print(f"🗂️  Datos: {args.data_dir}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"💻 Dispositivo: {device}")

    # Cargar checkpoint
    ckpt = torch.load(args.model, map_location=device)

    # Validaciones mínimas
    if "model_state_dict" not in ckpt:
        print("❌ El checkpoint no contiene 'model_state_dict'")
        return 1
    if "disease_names" not in ckpt or "num_diseases" not in ckpt:
        print("❌ El checkpoint no contiene 'disease_names' o 'num_diseases'")
        return 1

    disease_names = ckpt["disease_names"]
    num_diseases = ckpt["num_diseases"]

    print(f"📊 Clases del checkpoint ({num_diseases}): {disease_names}")

    # Transformaciones y DataLoaders (usaremos el split de validación)
    train_tf, val_tf = get_transforms()
    _, val_loader, _, _ = create_data_loaders(
        data_dir=args.data_dir,
        train_transform=train_tf,
        val_transform=val_tf,
        batch_size=args.batch_size,
        include_normal=True,
        num_workers=0,
    )

    # Construir modelo y cargar pesos
    model = create_model(
        num_diseases=num_diseases,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Calcular umbrales óptimos por clase
    optimal_thresholds, thresholds_comparison = calculate_optimal_thresholds(
        model=model,
        val_loader=val_loader,
        disease_names=disease_names,
        device=device,
    )

    # Respaldo opcional
    if args.backup:
        backup_path = args.model.replace(".pth", "_backup.pth")
        torch.save(ckpt, backup_path)
        print(f"🗄️  Copia de respaldo creada: {backup_path}")

    # Guardar umbrales en el checkpoint
    ckpt["optimal_thresholds"] = optimal_thresholds
    ckpt["thresholds_comparison"] = thresholds_comparison
    torch.save(ckpt, args.model)

    print("\n✅ Umbrales adaptativos añadidos y guardados en el checkpoint")
    print("   Clases con umbral:")
    for k, v in optimal_thresholds.items():
        print(f"   - {k}: {v:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


