"""
Módulo de Análisis
==================

Contiene herramientas para el análisis y visualización de resultados del modelo.

Archivos:
- analyze_model.py: Análisis detallado del modelo entrenado
- analyze_batch_results.py: Análisis de resultados de procesamiento en lote
"""

from .analyze_model import analyze_model, plot_training_history, plot_confusion_matrix
from .analyze_batch_results import analyze_batch_results, plot_batch_analysis

__all__ = ['analyze_model', 'plot_training_history', 'plot_confusion_matrix', 
           'analyze_batch_results', 'plot_batch_analysis']
