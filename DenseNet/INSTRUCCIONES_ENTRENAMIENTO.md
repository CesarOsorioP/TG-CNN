# 🚀 Instrucciones para Entrenar el Modelo Multi-Label

## 📋 Configuración del Entrenamiento

El modelo se entrena en **dos fases** con configuraciones automáticas de batch_size:

### **Fase 1: Entrenamiento con Backbone Congelado**
- **Batch Size**: 16
- **Backbone**: Congelado (solo entrenable el clasificador)
- **Learning Rate**: 0.0005
- **Scheduler**: ReduceLROnPlateau (adaptativo)
- **Early Stopping**: Patience=10 con mejora mínima de 0.1%
- **Loss**: Focal Loss (gamma=2.0)
- **Épocas**: 25 (configurable)

### **Fase 2: Fine-tuning (Opcional)**
- **Batch Size**: 8 (automático, para evitar out-of-memory)
- **Backbone**: Descongelado (entrenamiento completo)
- **Learning Rate**: 0.00005 (10x menor que fase 1)
- **Épocas**: 8 (configurable)

### **Umbrales Adaptativos**
- Se calculan **automáticamente** al final del entrenamiento
- Usan validación para encontrar umbral óptimo por enfermedad
- Se guardan junto con el modelo

---

## 💻 Comandos de Ejecución

### **Windows (CMD/PowerShell)**
```bash
cd DenseNet
python src/models/multilabel/train_multilabel.py train ^
    --data_dir data_diseases ^
    --batch_size 16 ^
    --num_epochs 20 ^
    --learning_rate 0.001 ^
    --loss_type focal ^
    --fine_tune_epochs 5 ^
    --freeze_backbone ^
    --device auto
```

### **Linux/Mac (Bash)**
```bash
cd DenseNet
python src/models/multilabel/train_multilabel.py train \
    --data_dir data_diseases \
    --batch_size 16 \
    --num_epochs 20 \
    --learning_rate 0.001 \
    --loss_type focal \
    --fine_tune_epochs 5 \
    --freeze_backbone \
    --device auto
```

### **Usando Script (Windows)**
```bash
cd DenseNet
ENTRENAR_MODELO.bat
```

### **Usando Script (Linux/Mac)**
```bash
cd DenseNet
chmod +x ENTRENAR_MODELO.sh
./ENTRENAR_MODELO.sh
```

---

## 📊 Parámetros del Comando

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `--data_dir` | `data_diseases` | Carpeta con las imágenes organizadas por enfermedad |
| `--batch_size` | `16` | Tamaño de lote (16 para backbone congelado, 8 automático para fine-tuning) |
| `--num_epochs` | `25` | Número de épocas de entrenamiento inicial |
| `--learning_rate` | `0.0005` | Learning rate para backbone congelado |
| `--fine_tune_lr` | `0.00005` | Learning rate para fine-tuning (10x menor) |
| `--loss_type` | `focal` | Tipo de función de pérdida (focal para mejor manejo de clases desbalanceadas) |
| `--fine_tune_epochs` | `8` | Número de épocas para fine-tuning |
| `--freeze_backbone` | - | Congela el backbone durante entrenamiento inicial |
| `--device` | `auto` | Detecta automáticamente GPU/CPU |
| `--output_dir` | `results/models/multilabel` | Directorio para guardar resultados |

---

## 🎯 ¿Qué Hace el Entrenamiento?

### **1. Fase de Entrenamiento (20 épocas)**
```
🔒 Backbone congelado (DenseNet-121)
📦 Batch size: 16
🎯 Función de pérdida: Focal Loss
📈 Learning rate: 0.001

✅ Beneficios:
- Entrenamiento rápido
- Menos consumo de memoria
- Aprende características específicas del dominio
```

### **2. Fase de Fine-tuning (5 épocas)**
```
🔓 Backbone descongelado
📦 Batch size: 8 (automático para evitar OOM)
🎯 Función de pérdida: Focal Loss
📈 Learning rate: 0.0001 (reducido 10x)

✅ Beneficios:
- Ajuste fino de toda la red
- Mejor adaptación a las radiografías
- Mayor precisión final
```

### **3. Cálculo de Umbrales Adaptativos**
```
📊 Usa datos de validación
🎯 Calcula curva ROC para cada enfermedad
🔍 Encuentra umbral que maximiza F1-score

✅ Beneficios:
- Mejor detección de enfermedades sutiles (COVID-19)
- Mejor balance Precision/Recall
- Umbral óptimo por enfermedad
```

---

## 📁 Archivos Generados

Al finalizar el entrenamiento se guardan:

```
results/models/multilabel/
├── densenet_multilabel_pre_ft.pth          # Checkpoint antes de fine-tuning
├── densenet_multilabel_model.pth          # Modelo final CON umbrales óptimos
├── training_config.json                   # Configuración completa
├── training_history.png                   # Gráficos de entrenamiento
└── ...

results/analysis/
└── result/
    └── [gráficos de análisis]
```

**Importante**: El archivo `densenet_multilabel_model.pth` es el modelo final que debes usar para predicciones.

---

## ⚙️ Personalización

### **Sin Fine-tuning**
```bash
python src/models/multilabel/train_multilabel.py train \
    --fine_tune_epochs 0 \
    --num_epochs 25
```

### **Más Épocas**
```bash
python src/models/multilabel/train_multilabel.py train \
    --num_epochs 30 \
    --fine_tune_epochs 10
```

### **Solo Entrenamiento Inicial**
```bash
python src/models/multilabel/train_multilabel.py train \
    --fine_tune_epochs 0 \
    --num_epochs 25
```

### **Entrenamiento Intensivo**
```bash
python src/models/multilabel/train_multilabel.py train \
    --num_epochs 30 \
    --fine_tune_epochs 15 \
    --learning_rate 0.0005
```

---

## 🐛 Solución de Problemas

### **Error: "CUDA out of memory"**
```bash
# Reducir batch_size
--batch_size 8
```

### **Error: "Directorio no encontrado"**
```bash
# Verificar estructura
ls data_diseases/
# Debe tener: Atelectasia, Cáncer, COVID-19, Edema, Neumonía, Normal, Tuberculosis
```

### **Entrenamiento muy lento**
```bash
# Desactivar num_workers en Windows
# El código ya usa num_workers=0 por defecto
```

---

## 📊 Seguimiento del Entrenamiento

Durante el entrenamiento verás:

```
🚀 ENTRENAMIENTO DE MODELO MULTI-LABEL
============================================================
📁 Directorio de datos: data_diseases
📊 Tamaño de lote: 16
🔄 Épocas: 20
📈 Learning rate: 0.001
💻 Dispositivo: cuda
🎯 Función de pérdida: focal

ÉPOCA 1/20
==================
📊 MÉTRICAS DE LA ÉPOCA:
  Entrenamiento - Loss: 0.4521, F1-Macro: 0.6234
  Validación - Loss: 0.3824, F1-Macro: 0.7123

...

🎯 CALCULANDO UMBRALES ÓPTIMOS POR ENFERMEDAD
============================================================

COVID-19:
  Threshold óptimo: 0.382
  F1 con threshold óptimo: 0.741
  F1 con threshold 0.5: 0.561
  Mejora: +32.1%

✅ Entrenamiento completado!
💾 Modelo guardado con umbrales óptimos
```

---

## ✅ Checklist Pre-Entrenamiento

- [ ] Datos organizados en `data_diseases/` con estructura train/val/test
- [ ] Cada enfermedad tiene ~3000 imágenes
- [ ] GPU disponible (opcional, pero recomendado)
- [ ] Espacio en disco (al menos 5 GB)
- [ ] Python 3.x instalado
- [ ] Librerías instaladas: `pip install -r requirements.txt`

---

## 🎉 ¡Listo para Entrenar!

Ejecuta el comando y el sistema hará:
1. ✅ Entrenar modelo con backbone congelado (batch_size=16)
2. ✅ Fine-tuning con backbone descongelado (batch_size=8 automático)
3. ✅ Calcular umbrales óptimos por enfermedad
4. ✅ Guardar modelo final con umbrales

**El modelo final ya tendrá los umbrales óptimos aplicados automáticamente** 🚀

