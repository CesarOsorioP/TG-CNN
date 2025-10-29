"""
Métricas de evaluación para clasificación multi-label.
Incluye métricas específicas para problemas multi-label como Hamming Loss, F1-Score, etc.

"""

import torch
import numpy as np
from sklearn.metrics import (
    f1_score, hamming_loss, jaccard_score, 
    precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix
)
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class MultiLabelMetrics:
    """
    Clase para calcular métricas de evaluación multi-label.
    """
    
    def __init__(self, disease_names, threshold=0.5):
        """
        Inicializar calculadora de métricas.
        
        Args:
            disease_names: Lista de nombres de enfermedades
            threshold: Umbral para considerar enfermedad presente
        """
        self.disease_names = disease_names
        self.threshold = threshold
        self.num_diseases = len(disease_names)
    
    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calcular todas las métricas multi-label.
        
        Args:
            y_true: Etiquetas verdaderas (numpy array o tensor)
            y_pred: Predicciones binarias (numpy array o tensor)
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
        
        # Métricas básicas multi-label
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        metrics['exact_match'] = self._exact_match_ratio(y_true, y_pred)
        
        # F1-Score
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Precision y Recall
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Jaccard Score
        metrics['jaccard_micro'] = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['jaccard_per_class'] = jaccard_score(y_true, y_pred, average=None, zero_division=0)
        
        # Métricas por clase
        metrics['per_class_metrics'] = self._calculate_per_class_metrics(y_true, y_pred)
        
        # Métricas con probabilidades (si están disponibles)
        if y_prob is not None:
            metrics['auc_micro'] = roc_auc_score(y_true, y_prob, average='micro')
            metrics['auc_macro'] = roc_auc_score(y_true, y_prob, average='macro')
            metrics['auc_per_class'] = roc_auc_score(y_true, y_prob, average=None)
        
        # Estadísticas de predicción
        metrics['prediction_stats'] = self._calculate_prediction_stats(y_true, y_pred)
        
        return metrics
    
    def _exact_match_ratio(self, y_true, y_pred):
        """Calcular ratio de coincidencia exacta."""
        return np.mean(np.all(y_true == y_pred, axis=1))
    
    def _calculate_per_class_metrics(self, y_true, y_pred):
        """Calcular métricas por clase."""
        per_class = {}
        
        for i, disease in enumerate(self.disease_names):
            per_class[disease] = {
                'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'support': int(np.sum(y_true[:, i]))
            }
        
        return per_class
    
    def _calculate_prediction_stats(self, y_true, y_pred):
        """Calcular estadísticas de predicción."""
        # Número de etiquetas por imagen
        true_labels_per_image = np.sum(y_true, axis=1)
        pred_labels_per_image = np.sum(y_pred, axis=1)
        
        # Distribución de etiquetas
        stats = {
            'avg_true_labels': np.mean(true_labels_per_image),
            'avg_pred_labels': np.mean(pred_labels_per_image),
            'max_true_labels': np.max(true_labels_per_image),
            'max_pred_labels': np.max(pred_labels_per_image),
            'single_label_true': np.sum(true_labels_per_image == 1),
            'multi_label_true': np.sum(true_labels_per_image > 1),
            'single_label_pred': np.sum(pred_labels_per_image == 1),
            'multi_label_pred': np.sum(pred_labels_per_image > 1),
            'normal_true': np.sum(true_labels_per_image == 0),
            'normal_pred': np.sum(pred_labels_per_image == 0)
        }
        
        return stats
    
    def print_metrics(self, metrics, detailed=True):
        """
        Imprimir métricas de forma organizada.
        
        Args:
            metrics: Diccionario con métricas
            detailed: Si mostrar detalles por clase
        """
        print("\n" + "="*70)
        print("📊 MÉTRICAS DE EVALUACIÓN MULTI-LABEL")
        print("="*70)
        
        # Métricas generales
        print(f"\n🎯 MÉTRICAS GENERALES:")
        print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
        print(f"  Exact Match Ratio: {metrics['exact_match']:.4f}")
        print(f"  Jaccard Score (Macro): {metrics['jaccard_macro']:.4f}")
        print(f"  Jaccard Score (Micro): {metrics['jaccard_micro']:.4f}")
        
        # F1-Score
        print(f"\n📈 F1-SCORE:")
        print(f"  Macro: {metrics['f1_macro']:.4f}")
        print(f"  Micro: {metrics['f1_micro']:.4f}")
        print(f"  Weighted: {metrics['f1_weighted']:.4f}")
        
        # Precision y Recall
        print(f"\n🎯 PRECISION & RECALL:")
        print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"  Precision (Micro): {metrics['precision_micro']:.4f}")
        print(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
        print(f"  Recall (Micro): {metrics['recall_micro']:.4f}")
        
        # AUC (si está disponible)
        if 'auc_macro' in metrics:
            print(f"\n📊 AUC-ROC:")
            print(f"  Macro: {metrics['auc_macro']:.4f}")
            print(f"  Micro: {metrics['auc_micro']:.4f}")
        
        # Estadísticas de predicción
        stats = metrics['prediction_stats']
        print(f"\n📋 ESTADÍSTICAS DE PREDICCIÓN:")
        print(f"  Promedio etiquetas verdaderas: {stats['avg_true_labels']:.2f}")
        print(f"  Promedio etiquetas predichas: {stats['avg_pred_labels']:.2f}")
        print(f"  Imágenes con una etiqueta (verdad): {stats['single_label_true']:,}")
        print(f"  Imágenes con múltiples etiquetas (verdad): {stats['multi_label_true']:,}")
        print(f"  Imágenes normales (verdad): {stats['normal_true']:,}")
        
        # Métricas por clase (si se solicita detalle)
        if detailed:
            print(f"\n🏥 MÉTRICAS POR ENFERMEDAD:")
            per_class = metrics['per_class_metrics']
            print(f"{'Enfermedad':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
            print("-" * 60)
            
            for disease, class_metrics in per_class.items():
                print(f"{disease:<15} {class_metrics['precision']:<10.4f} "
                      f"{class_metrics['recall']:<10.4f} {class_metrics['f1']:<10.4f} "
                      f"{class_metrics['support']:<8}")
    
    def plot_confusion_matrices(self, y_true, y_pred, save_path=None):
        """
        Crear matrices de confusión para cada enfermedad.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            save_path: Ruta para guardar la figura
        """
        # Convertir a numpy si es necesario
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        # Calcular número de subplots
        n_diseases = len(self.disease_names)
        n_cols = 3
        n_rows = (n_diseases + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, disease in enumerate(self.disease_names):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Matriz de confusión para esta enfermedad
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            
            # Crear heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
            ax.set_title(f'{disease}\nConfusion Matrix')
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Verdadero')
        
        # Ocultar subplots vacíos
        for i in range(n_diseases, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Matrices de confusión guardadas en: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, metrics_history, save_path=None):
        """
        Crear gráfico de comparación de métricas durante el entrenamiento.
        
        Args:
            metrics_history: Lista de diccionarios con métricas por época
            save_path: Ruta para guardar la figura
        """
        epochs = range(1, len(metrics_history) + 1)
        
        # Extraer métricas
        f1_macro = [m['f1_macro'] for m in metrics_history]
        f1_micro = [m['f1_micro'] for m in metrics_history]
        hamming_loss = [m['hamming_loss'] for m in metrics_history]
        exact_match = [m['exact_match'] for m in metrics_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # F1-Score
        ax1.plot(epochs, f1_macro, 'b-', label='F1 Macro', linewidth=2)
        ax1.plot(epochs, f1_micro, 'r-', label='F1 Micro', linewidth=2)
        ax1.set_title('F1-Score durante el entrenamiento')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('F1-Score')
        ax1.legend()
        ax1.grid(True)
        
        # Hamming Loss
        ax2.plot(epochs, hamming_loss, 'g-', label='Hamming Loss', linewidth=2)
        ax2.set_title('Hamming Loss durante el entrenamiento')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Hamming Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Exact Match
        ax3.plot(epochs, exact_match, 'm-', label='Exact Match', linewidth=2)
        ax3.set_title('Exact Match Ratio durante el entrenamiento')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Exact Match Ratio')
        ax3.legend()
        ax3.grid(True)
        
        # Comparación de F1 por clase (última época)
        if metrics_history:
            last_metrics = metrics_history[-1]
            f1_per_class = last_metrics['f1_per_class']
            
            ax4.bar(range(len(self.disease_names)), f1_per_class, 
                   color='skyblue', alpha=0.7)
            ax4.set_title('F1-Score por Enfermedad (Última Época)')
            ax4.set_xlabel('Enfermedad')
            ax4.set_ylabel('F1-Score')
            ax4.set_xticks(range(len(self.disease_names)))
            ax4.set_xticklabels(self.disease_names, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Gráfico de métricas guardado en: {save_path}")
        
        plt.show()

def evaluate_model_multi_label(model, data_loader, disease_names, device='cuda', threshold=0.5, optimal_thresholds=None):
    """
    Evaluar modelo multi-label en un DataLoader.
    
    Args:
        model: Modelo a evaluar
        data_loader: DataLoader con datos de prueba
        disease_names: Lista de nombres de enfermedades
        device: Dispositivo a usar
        threshold: Umbral fijo para predicciones binarias (si optimal_thresholds es None)
        optimal_thresholds: Diccionario con umbrales óptimos por enfermedad
        
    Returns:
        dict: Métricas de evaluación
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    print("🔄 Evaluando modelo multi-label...")
    if optimal_thresholds is not None:
        print("📊 Usando umbrales adaptativos")
    else:
        print(f"📊 Usando threshold fijo: {threshold}")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            # Obtener predicciones
            probabilities = model(data)
            
            # Aplicar umbrales adaptativos o fijo
            if optimal_thresholds is not None:
                # Usar umbrales adaptativos
                predictions = torch.zeros_like(probabilities)
                for i, disease in enumerate(disease_names):
                    disease_threshold = optimal_thresholds.get(disease, threshold)
                    predictions[:, i] = (probabilities[:, i] > disease_threshold).float()
            else:
                # Usar threshold fijo
                predictions = (probabilities > threshold).float()
            
            all_predictions.append(predictions.cpu())
            all_targets.append(target.cpu())
            all_probabilities.append(probabilities.cpu())
    
    # Concatenar todos los resultados
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_probabilities = torch.cat(all_probabilities, dim=0).numpy()
    
    # Calcular métricas
    # Usar el threshold apropiado para el objeto Metrics
    used_threshold = threshold if optimal_thresholds is None else optimal_thresholds
    metrics_calculator = MultiLabelMetrics(disease_names, used_threshold)
    metrics = metrics_calculator.calculate_metrics(
        all_targets, all_predictions, all_probabilities
    )
    
    return metrics, all_predictions, all_targets, all_probabilities

if __name__ == "__main__":
    # Ejemplo de uso
    print("🧪 Probando métricas multi-label...")
    
    # Crear datos de ejemplo
    n_samples = 100
    n_diseases = 6
    disease_names = ['Neumonía', 'Cáncer', 'Atelectasia', 'Edema', 'Tuberculosis', 'COVID-19']
    
    # Generar etiquetas y predicciones aleatorias
    y_true = np.random.randint(0, 2, (n_samples, n_diseases))
    y_pred = np.random.randint(0, 2, (n_samples, n_diseases))
    y_prob = np.random.rand(n_samples, n_diseases)
    
    # Calcular métricas
    metrics_calc = MultiLabelMetrics(disease_names)
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_prob)
    
    # Imprimir métricas
    metrics_calc.print_metrics(metrics)
    
    print("\n✅ Pruebas de métricas completadas!")
