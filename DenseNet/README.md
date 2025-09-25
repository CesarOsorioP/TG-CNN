# Clasificación de Radiografías de Tórax con DenseNet

Este proyecto implementa un clasificador de imágenes médicas usando DenseNet con transfer learning para distinguir entre radiografías de tórax y otras imágenes.

## 🎯 Objetivo

Entrenar un modelo de deep learning que pueda clasificar automáticamente:
- **Radiografías de tórax** (clase positiva)
- **Otras imágenes** (clase negativa)

## 🏗️ Arquitectura

- **Modelo base**: DenseNet-121 pre-entrenado en ImageNet
- **Técnica**: Transfer Learning con Freeze del Backbone
- **Clasificador**: Red personalizada con dropout y capas densas
- **Entrada**: Imágenes RGB de 224x224 píxeles
- **Optimización**: Backbone congelado para evitar overfitting en datasets pequeños

## 📁 Estructura del Proyecto

```
DenseNet/
├── main.py                 # Script principal de entrada
├── requirements.txt        # Dependencias del proyecto
├── README.md              # Este archivo
├── src/                   # Código fuente principal
│   ├── __init__.py
│   ├── models/            # Modelos y entrenamiento
│   │   ├── __init__.py
│   │   ├── train_model.py      # Entrenamiento del modelo
│   │   ├── predict.py          # Predicción individual
│   │   └── freeze_experiments.py # Experimentos de freeze
│   ├── analysis/          # Análisis y visualización
│   │   ├── __init__.py
│   │   ├── analyze_model.py    # Análisis del modelo
│   │   └── analyze_batch_results.py # Análisis de lotes
│   └── utils/             # Utilidades auxiliares
│       └── __init__.py
├── gui/                   # Interfaz gráfica
│   └── gui_app.py         # Aplicación GUI principal
├── scripts/               # Scripts de utilidad
│   ├── batch_predictor.py # Procesamiento en lote avanzado
│   ├── quick_batch.py     # Procesamiento en lote rápido
│   ├── quick_start.py     # Guía interactiva paso a paso
│   └── setup_data.py      # Preparación de datos
├── data/                  # Datos de entrenamiento
│   ├── chest_xray/        # Imágenes de radiografías de tórax
│   └── other_images/      # Otras imágenes (no radiografías)
├── results/               # Resultados y modelos
│   ├── models/            # Modelos entrenados
│   ├── predictions/       # Resultados de predicción
│   └── analysis/          # Resultados de análisis
└── docs/                  # Documentación adicional
```

## 🚀 Instalación

### 1. Clonar o descargar el proyecto
```bash
# Si tienes git instalado
git clone <url-del-repositorio>
cd DenseNet

# O simplemente descarga y extrae los archivos
```

### 2. Crear entorno virtual (recomendado)
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Preparar datos
Crea la siguiente estructura de directorios:
```
data/
├── chest_xray/     # Coloca aquí las radiografías de tórax
└── other_images/   # Coloca aquí otras imágenes (no radiografías)
```

## 📊 Preparación de Datos

### Estructura de directorios requerida:
```
data/
├── chest_xray/          # Radiografías de tórax
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── other_images/        # Otras imágenes
    ├── photo1.jpg
    ├── photo2.png
    └── ...
```

### Formatos de imagen soportados:
- JPG/JPEG
- PNG
- BMP (se convertirá automáticamente)

### Recomendaciones para los datos:
- **Mínimo**: 100 imágenes por clase
- **Óptimo**: 500+ imágenes por clase
- **Resolución**: Cualquier tamaño (se redimensionará automáticamente)
- **Calidad**: Imágenes claras y bien etiquetadas

## 🏋️ Uso del Proyecto

### Script Principal (Recomendado)
```bash
# Ver todas las opciones disponibles
python main.py --help

# Entrenar el modelo
python main.py train

# Predecir una imagen
python main.py predict ruta/a/imagen.jpg

# Abrir interfaz gráfica
python main.py gui

# Procesar imágenes en lote
python main.py batch ruta/a/directorio

# Analizar resultados
python main.py analyze
```

### Uso Directo de Scripts
```bash
# Entrenamiento básico
python src/models/train_model.py

# Predicción individual
python src/models/predict.py ruta/a/imagen.jpg

# Interfaz gráfica
python gui/gui_app.py

# Procesamiento en lote
python scripts/batch_predictor.py --input_dir ruta/a/directorio
```

### Estrategias de Freeze para Datasets Pequeños

El modelo incluye **freeze del backbone** por defecto para evitar overfitting en datasets pequeños (~1200 imágenes). Esto significa que solo se entrenan las capas del clasificador, no las características pre-entrenadas.

#### Configuración por defecto optimizada:
- **Backbone congelado**: Solo se entrena el clasificador
- **Batch size**: 16 (reducido para datasets pequeños)
- **Épocas**: 15 (aumentado para compensar el freeze)
- **Fine-tuning opcional**: 5 épocas adicionales con backbone descongelado

#### Experimentar con diferentes estrategias:
```bash
# Comparar todas las estrategias de freeze
python freeze_experiments.py --compare

# Ver recomendaciones para tu tamaño de dataset
python freeze_experiments.py --dataset_size 1200
```

### Configuración personalizada:
Puedes modificar los parámetros en el archivo `train_model.py`:

```python
# Configuración principal
DATA_DIR = "data"           # Directorio con las imágenes
BATCH_SIZE = 32            # Tamaño del lote
NUM_EPOCHS = 20            # Número de épocas
LEARNING_RATE = 0.001      # Tasa de aprendizaje
TRAIN_SPLIT = 0.7          # 70% para entrenamiento
VAL_SPLIT = 0.15           # 15% para validación
TEST_SPLIT = 0.15          # 15% para prueba
```

### Durante el entrenamiento verás:
- Progreso por época
- Pérdida y precisión en entrenamiento/validación
- Mejor modelo guardado automáticamente
- Gráficos de entrenamiento

### Archivos generados:
- `densenet_chest_xray_model.pth` - Modelo entrenado
- `training_history.png` - Gráficos de entrenamiento
- `training_config.json` - Configuración usada

## 🖥️ Interfaz Gráfica (GUI)

### Lanzar la GUI:
```bash
# Lanzador simple con verificaciones
python run_gui.py

# O directamente
python gui_app.py

# Demostración completa
python demo_gui.py
```

### Características de la GUI:
- **🎨 Interfaz moderna**: Diseño limpio y profesional
- **📁 Carga de modelos**: Selección automática o manual de modelos
- **🖼️ Visualización de imágenes**: Preview con redimensionamiento automático
- **🔍 Análisis en tiempo real**: Predicciones instantáneas
- **📊 Estadísticas del modelo**: Información detallada de parámetros
- **📈 Probabilidades detalladas**: Barras de progreso para cada clase
- **🖱️ Drag & Drop**: Arrastra imágenes directamente a la interfaz

### Funcionalidades principales:
1. **Carga de modelo**: Automática o manual con verificación de estado
2. **Selección de imágenes**: Botón de archivo o drag & drop
3. **Análisis visual**: Predicción con confianza y probabilidades
4. **Información del modelo**: Estadísticas de parámetros y dispositivo
5. **Interfaz responsiva**: Se adapta al tamaño de la ventana

## 🔮 Predicción por Línea de Comandos

### Clasificar una imagen individual:
```bash
python predict.py --image ruta/a/imagen.jpg --visualize
```

### Clasificar todas las imágenes de un directorio:
```bash
python predict.py --directory ruta/a/directorio --output resultados.json
```

### Opciones disponibles:
- `--model`: Ruta al modelo (por defecto: `densenet_chest_xray_model.pth`)
- `--image`: Imagen individual para clasificar
- `--directory`: Directorio con imágenes para procesar
- `--output`: Archivo JSON para guardar resultados
- `--visualize`: Mostrar visualización de la predicción
- `--probabilities`: Incluir probabilidades en los resultados

### Ejemplo de uso:
```bash
# Clasificar una imagen con visualización
python predict.py --image "mi_radiografia.jpg" --visualize --probabilities

# Procesar directorio completo
python predict.py --directory "nuevas_imagenes/" --output "resultados.json"
```

## 🚀 Procesamiento en Lote (5000+ Imágenes)

### Procesamiento Rápido:
```bash
# Procesar directorio completo de manera rápida
python quick_batch.py mi_dataset/

# Especificar modelo personalizado
python quick_batch.py mi_dataset/ mi_modelo.pth
```

### Procesamiento Avanzado:
```bash
# Procesamiento con opciones avanzadas
python batch_predictor.py --input_dir mi_dataset/ --output_dir resultados/ --batch_size 64

# Con GPU para máxima velocidad
python batch_predictor.py --input_dir mi_dataset/ --device cuda --batch_size 128
```

### Opciones del procesamiento en lote:
- `--input_dir`: Directorio con imágenes a procesar (requerido)
- `--output_dir`: Directorio para guardar resultados
- `--batch_size`: Tamaño del lote (32, 64, 128, etc.)
- `--device`: Dispositivo (cuda, cpu, auto)
- `--model`: Ruta al modelo entrenado

### Análisis de Resultados:
```bash
# Analizar resultados del procesamiento en lote
python analyze_batch_results.py --results_dir quick_batch_results/

# Con directorio de salida personalizado
python analyze_batch_results.py --results_dir resultados/ --output_dir analisis/
```

### Archivos Generados:
- **Resumen JSON**: Estadísticas generales del procesamiento
- **CSV detallado**: Resultados de cada imagen individual
- **Lista de radiografías**: Archivo con rutas de radiografías detectadas
- **Lista de otras imágenes**: Archivo con rutas de otras imágenes
- **Visualizaciones**: Gráficos y análisis visual
- **Reporte detallado**: Análisis completo en texto

### Ejemplo de Salida:
```
📊 RESUMEN DE RESULTADOS
====================================
Total de imágenes procesadas: 5,247
Predicciones exitosas: 5,201
Predicciones fallidas: 46

🏥 RADIOGRAFÍAS DE TÓRAX:
  Cantidad: 3,156
  Porcentaje: 60.7%

🖼️  OTRAS IMÁGENES:
  Cantidad: 2,045
  Porcentaje: 39.3%

📈 ESTADÍSTICAS DE CONFIANZA:
  Confianza promedio: 0.892
  Alta confianza (>90%): 4,123
  Confianza media (70-90%): 1,078
  Baja confianza (<70%): 0
```

## 📈 Análisis del Modelo

### Análisis completo:
```bash
python analyze_model.py --model densenet_chest_xray_model.pth --report
```

### Análisis de imagen específica:
```bash
python analyze_model.py --model densenet_chest_xray_model.pth --image "imagen.jpg"
```

### Funciones de análisis incluidas:
- **Arquitectura del modelo**: Parámetros, capas, etc.
- **Mapas de características**: Visualización de activaciones
- **Grad-CAM**: Mapa de calor de atención
- **Métricas de rendimiento**: Matriz de confusión, ROC, etc.
- **Embeddings t-SNE**: Visualización 2D de características
- **Reporte completo**: Análisis detallado en JSON

## 📊 Interpretación de Resultados

### Salida de predicción:
```json
{
  "image_path": "imagen.jpg",
  "predicted_class": "chest_xray",
  "confidence": 0.95,
  "is_chest_xray": true,
  "probabilities": {
    "chest_xray": 0.95,
    "other_images": 0.05
  }
}
```

### Métricas importantes:
- **Precisión (Accuracy)**: Porcentaje de predicciones correctas
- **Confianza**: Probabilidad de la predicción (0-1)
- **Sensibilidad**: Capacidad de detectar radiografías de tórax
- **Especificidad**: Capacidad de rechazar imágenes que no son radiografías

## ⚙️ Configuración Avanzada

### Modificar la arquitectura:
Edita la clase `DenseNetClassifier` en `train_model.py`:

```python
# Cambiar número de clases
model = DenseNetClassifier(num_classes=2)

# Usar diferentes arquitecturas
self.backbone = models.densenet121(pretrained=True)  # DenseNet-121
self.backbone = models.densenet161(pretrained=True)  # DenseNet-161
self.backbone = models.densenet201(pretrained=True)  # DenseNet-201
```

### Ajustar hiperparámetros:
```python
# En la función main() de train_model.py
BATCH_SIZE = 16          # Reducir si tienes poca memoria
NUM_EPOCHS = 50          # Más épocas para mejor rendimiento
LEARNING_RATE = 0.0001   # Tasa de aprendizaje más conservadora
```

### Data Augmentation personalizada:
```python
# En get_transforms()
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),        # Más rotación
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])
```

## 🐛 Solución de Problemas

### Error: "No se encontraron imágenes"
- Verifica que el directorio `data/` existe
- Asegúrate de que las subcarpetas `chest_xray/` y `other_images/` existen
- Verifica que las imágenes tienen extensiones válidas (.jpg, .png, .jpeg)

### Error: "CUDA out of memory"
- Reduce el `BATCH_SIZE` en `train_model.py`
- Usa `device='cpu'` si no tienes GPU

### Bajo rendimiento del modelo:
- Aumenta el número de épocas (`NUM_EPOCHS`)
- Ajusta la tasa de aprendizaje (`LEARNING_RATE`)
- Verifica la calidad y cantidad de datos de entrenamiento
- Considera usar más data augmentation

### Error de dependencias:
```bash
# Reinstalar PyTorch con CUDA (si tienes GPU NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# O instalar versión CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 📚 Conceptos Técnicos

### ¿Qué es DenseNet?
DenseNet (Dense Convolutional Network) es una arquitectura que conecta cada capa con todas las capas anteriores, creando conexiones densas. Esto permite:
- Mejor flujo de información
- Menos parámetros que otras arquitecturas
- Excelente rendimiento en clasificación de imágenes

### ¿Qué es Transfer Learning?
Transfer Learning es una técnica donde:
1. Usamos un modelo pre-entrenado (DenseNet entrenado en ImageNet)
2. Reemplazamos la capa clasificadora final
3. Entrenamos solo las nuevas capas con nuestros datos específicos

### ¿Qué es Freeze del Backbone?
El **freeze del backbone** es una técnica avanzada donde:
1. **Congelamos** los parámetros del modelo pre-entrenado (backbone)
2. Solo entrenamos las **nuevas capas** del clasificador
3. Esto evita que el modelo "olvide" las características generales aprendidas

### Ventajas del Transfer Learning + Freeze:
- **Menos datos**: Funciona excelente con pocas imágenes (~1200)
- **Menos overfitting**: El backbone congelado previene sobreajuste
- **Menos tiempo**: Entrenamiento más rápido (menos parámetros)
- **Mejor rendimiento**: Aprovecha características generales sin destruirlas
- **Más estable**: Menos propenso a divergir durante el entrenamiento

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Algunas ideas:
- Mejorar la documentación
- Agregar nuevas arquitecturas de modelo
- Implementar técnicas de data augmentation
- Optimizar el código para mejor rendimiento

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.

## 📞 Soporte

Si tienes problemas o preguntas:
1. Revisa la sección de solución de problemas
2. Verifica que todas las dependencias estén instaladas correctamente
3. Asegúrate de que la estructura de datos sea correcta

## 🎉 ¡Feliz Clasificación!

¡Esperamos que este proyecto te ayude a clasificar radiografías de tórax de manera efectiva! Recuerda que este es un proyecto educativo y de investigación, y no debe usarse para diagnóstico médico real sin la supervisión de profesionales médicos calificados.
