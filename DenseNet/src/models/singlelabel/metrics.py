"""
Métricas de evaluación para clasificación single-label.
Incluye accuracy, precision, recall, F1-Score, y confusion matrix.

"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class SingleLabelMetrics:
    """
    Clase para calcular métricas de evaluación single-label.
    """
    
    def __init__(self, class_names):
        """
        Inicializar calculadora de métricas.
        
        Args:
            class_names: Lista de nombres de clases
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calcular todas las métricas single-label.
        
        Args:
            y_true: Etiquetas verdaderas (numpy array o tensor) - índices de clase
            y_pred: Predicciones (numpy array o tensor) - índices de clase
            y_prob: Probabilidades (opcional, para métricas que las requieren)
            
        Returns:
            dict: Diccionario con todas las métricas
        """
        # Convertir a numpy si es necesario
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        if y_prob is not None and torch.is_tensor(y_prob):
            y_prob = y_prob.cpu().numpy()
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # F1-Score
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Precision y Recall
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion Matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Métricas por clase
        metrics['per_class_metrics'] = self._calculate_per_class_metrics(y_true, y_pred)
        
        # Métricas con probabilidades (si están disponibles)
        if y_prob is not None:
            try:
                # Para multi-class, usar one-vs-rest
                metrics['auc_macro'] = roc_auc_score(
                    y_true, y_prob, 
                    average='macro', 
                    multi_class='ovr',
                    labels=list(range(self.num_classes))
                )
                metrics['auc_per_class'] = roc_auc_score(
                    y_true, y_prob,
                    average=None,
                    multi_class='ovr',
                    labels=list(range(self.num_classes))
                )
            except Exception as e:
                print(f"⚠️  No se pudo calcular AUC: {e}")
                metrics['auc_macro'] = 0.0
                metrics['auc_per_class'] = np.zeros(self.num_classes)
        
        return metrics
    
    def _calculate_per_class_metrics(self, y_true, y_pred):
        """Calcular métricas por clase."""
        per_class = {}
        
        for i, class_name in enumerate(self.class_names):
            # Calcular TP, FP, FN, TN para esta clase
            tp = np.sum((y_true == i) & (y_pred == i))
            fp = np.sum((y_true != i) & (y_pred == i))
            fn = np.sum((y_true == i) & (y_pred != i))
            tn = np.sum((y_true != i) & (y_pred != i))
            
            # Calcular métricas
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            support = tp + fn
            
            per_class[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn)
            }
        
        return per_class
    
    def print_metrics(self, metrics):
        """
        Imprimir métricas de forma legible.
        
        Args:
            metrics: Diccionario con métricas calculadas
        """
        print("\n" + "="*70)
        print("📊 MÉTRICAS DE EVALUACIÓN SINGLE-LABEL")
        print("="*70)
        
        print(f"\n🎯 MÉTRICAS GENERALES:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        
        print(f"\n📈 PRECISION & RECALL:")
        print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
        
        if 'auc_macro' in metrics:
            print(f"\n📊 AUC-ROC:")
            print(f"  Macro: {metrics['auc_macro']:.4f}")
        
        print(f"\n🏥 MÉTRICAS POR CLASE:")
        print(f"{'Clase':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        
        for i, class_name in enumerate(self.class_names):
            class_metrics = metrics['per_class_metrics'][class_name]
            print(f"{class_name:<15} {class_metrics['precision']:<12.4f} "
                  f"{class_metrics['recall']:<12.4f} {class_metrics['f1_score']:<12.4f} "
                  f"{class_metrics['support']:<10}")
        
        # Classification Report
        print(f"\n📋 CLASSIFICATION REPORT:")
        print(classification_report(
            y_true=None,  # Se calculará internamente
            y_pred=None,
            target_names=self.class_names,
            output_dict=False
        ))
    
    def plot_confusion_matrix(self, metrics, save_path=None):
        """
        Visualizar matriz de confusión.
        
        Args:
            metrics: Diccionario con métricas (debe incluir 'confusion_matrix')
            save_path: Ruta para guardar la figura
        """
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
        plt.ylabel('Etiqueta Verdadera', fontsize=12)
        plt.xlabel('Predicción', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Matriz de confusión guardada en: {save_path}")
        
        plt.show()

def evaluate_model_single_label(model, data_loader, class_names, device='cuda'):
    """
    Evaluar modelo single-label en un DataLoader.
    
    Args:
        model: Modelo a evaluar
        data_loader: DataLoader con datos de prueba
        class_names: Lista de nombres de clases
        device: Dispositivo a usar
        
    Returns:
        tuple: (metrics, predictions, targets, probabilities)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Evaluando modelo"):
            data, target = data.to(device), target.to(device)
            
            logits = model(data)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calcular métricas
    metrics_calculator = SingleLabelMetrics(class_names)
    metrics = metrics_calculator.calculate_metrics(
        np.array(all_targets),
        np.array(all_predictions),
        np.array(all_probabilities)
    )
    
    return metrics, np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)

if __name__ == "__main__":
    # Ejemplo de uso
    print("🧪 Probando métricas single-label...")
    
    # Crear datos de ejemplo
    n_samples = 100
    n_classes = 8
    class_names = ['Neumonía', 'Atelectasia', 'Edema', 'Tuberculosis', 
                   'COVID-19', 'Normal', 'Nodules', 'Mass']
    
    # Generar etiquetas y predicciones aleatorias
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.rand(n_samples, n_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalizar
    
    # Calcular métricas
    metrics_calc = SingleLabelMetrics(class_names)
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_prob)
    
    # Imprimir métricas
    metrics_calc.print_metrics(metrics)
    
    print("\n✅ Pruebas completadas!")

