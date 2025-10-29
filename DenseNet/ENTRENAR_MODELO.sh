#!/bin/bash
# Script para entrenar el modelo multi-label con umbrales adaptativos

echo "🚀 Iniciando entrenamiento del modelo multi-label..."
echo "================================================="
echo ""

cd "$(dirname "$0")"

python src/models/multilabel/train_multilabel.py train \
    --data_dir data_diseases \
    --batch_size 16 \
    --num_epochs 25 \
    --learning_rate 0.0005 \
    --loss_type focal \
    --fine_tune_epochs 8 \
    --fine_tune_lr 0.00005 \
    --freeze_backbone \
    --device auto \
    --output_dir results/models/multilabel

echo ""
echo "✅ Entrenamiento completado!"
