"""
GUI moderna para predicción de radiografías de tórax con DenseNet
Interfaz gráfica intuitiva para cargar modelos y hacer predicciones

Autor: Asistente AI
Fecha: 2024
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import os
import sys
import threading
import json
from datetime import datetime

# Agregar el directorio src al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.predict import ChestXrayPredictor

class ChestXrayGUI:
    """
    Interfaz gráfica moderna para predicción de radiografías de tórax
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Clasificador de Radiografías de Tórax - DenseNet")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Configurar estilo
        self.setup_styles()
        
        # Variables
        self.predictor = None
        self.current_image_path = None
        self.current_image = None
        self.original_image = None  # Imagen original sin redimensionar
        self.model_path = "densenet_chest_xray_model.pth"
        self.current_scale = 1.0  # Factor de escala actual
        
        # Crear interfaz
        self.create_widgets()
        
        # Cargar modelo automáticamente si existe
        self.load_model_auto()
    
    def setup_styles(self):
        """Configurar estilos de la interfaz"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar colores
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'), foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Success.TLabel', font=('Arial', 10), foreground='#27ae60')
        style.configure('Error.TLabel', font=('Arial', 10), foreground='#e74c3c')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='#3498db')
        
        # Configurar botones
        style.configure('Action.TButton', font=('Arial', 10, 'bold'))
        style.configure('Success.TButton', font=('Arial', 10, 'bold'))
    
    def create_widgets(self):
        """Crear los elementos de la interfaz"""
        
        # Frame principal con scroll
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Título principal
        title_label = ttk.Label(
            main_frame, 
            text="🔬 Clasificador de Radiografías de Tórax",
            style='Title.TLabel'
        )
        title_label.pack(pady=(0, 20))
        
        # Frame para información del modelo
        model_frame = ttk.LabelFrame(main_frame, text="📁 Modelo", padding=15)
        model_frame.pack(fill="x", pady=(0, 10))
        
        # Información del modelo
        model_info_frame = ttk.Frame(model_frame)
        model_info_frame.pack(fill="x")
        
        # Ruta del modelo
        ttk.Label(model_info_frame, text="Ruta del modelo:", style='Header.TLabel').pack(anchor="w")
        self.model_path_var = tk.StringVar(value=self.model_path)
        model_path_entry = ttk.Entry(model_info_frame, textvariable=self.model_path_var, width=50)
        model_path_entry.pack(fill="x", pady=(5, 10))
        
        # Botones de modelo
        model_buttons_frame = ttk.Frame(model_frame)
        model_buttons_frame.pack(fill="x")
        
        self.load_model_btn = ttk.Button(
            model_buttons_frame,
            text="🔄 Cargar Modelo",
            command=self.load_model,
            style='Action.TButton'
        )
        self.load_model_btn.pack(side="left", padx=(0, 10))
        
        self.browse_model_btn = ttk.Button(
            model_buttons_frame,
            text="📂 Buscar Modelo",
            command=self.browse_model
        )
        self.browse_model_btn.pack(side="left", padx=(0, 10))
        
        # Estado del modelo
        self.model_status = ttk.Label(
            model_buttons_frame,
            text="❌ Modelo no cargado",
            style='Error.TLabel'
        )
        self.model_status.pack(side="right")
        
        # Frame para selección de imagen
        image_frame = ttk.LabelFrame(main_frame, text="🖼️ Imagen", padding=15)
        image_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Frame para botones de imagen
        image_buttons_frame = ttk.Frame(image_frame)
        image_buttons_frame.pack(fill="x", pady=(0, 10))
        
        # Botón para seleccionar imagen
        self.select_image_btn = ttk.Button(
            image_buttons_frame,
            text="📁 Seleccionar Imagen",
            command=self.select_image,
            state="disabled",
            style='Action.TButton'
        )
        self.select_image_btn.pack(side="left", padx=(0, 10))
        
        # Botón para ajustar tamaño
        self.fit_image_btn = ttk.Button(
            image_buttons_frame,
            text="🔍 Ajustar Tamaño",
            command=self.fit_image,
            state="disabled"
        )
        self.fit_image_btn.pack(side="left", padx=(0, 10))
        
        # Botón para tamaño original
        self.original_size_btn = ttk.Button(
            image_buttons_frame,
            text="📏 Tamaño Original",
            command=self.show_original_size,
            state="disabled"
        )
        self.original_size_btn.pack(side="left")
        
        # Frame para mostrar imagen con scroll
        image_display_frame = ttk.Frame(image_frame)
        image_display_frame.pack(fill="both", expand=True)
        
        # Crear Canvas con scrollbars
        self.image_canvas = tk.Canvas(
            image_display_frame,
            bg="#f8f9fa",
            relief="sunken",
            bd=2,
            width=600,
            height=190
        )
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(
            image_display_frame,
            orient="vertical",
            command=self.image_canvas.yview
        )
        h_scrollbar = ttk.Scrollbar(
            image_display_frame,
            orient="horizontal",
            command=self.image_canvas.xview
        )
        
        # Configurar canvas
        self.image_canvas.configure(
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )
        
        # Posicionar elementos
        self.image_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Configurar grid weights
        image_display_frame.grid_rowconfigure(0, weight=1)
        image_display_frame.grid_columnconfigure(0, weight=1)
        
        # Label para mostrar imagen (dentro del canvas)
        self.image_label = tk.Label(
            self.image_canvas,
            text="No hay imagen seleccionada\n\nArrastra una imagen aquí o usa el botón 'Seleccionar Imagen'",
            bg="#f8f9fa",
            fg="#6c757d",
            font=("Arial", 12),
            anchor="center"
        )
        
        # Crear ventana en el canvas para el label
        self.image_window = self.image_canvas.create_window(0, 0, anchor="nw", window=self.image_label)
        
        # Configurar scroll region inicial
        self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
        
        # Bind para actualizar scroll region cuando cambie el tamaño
        self.image_label.bind("<Configure>", self.on_image_configure)
        
        # Configurar zoom con rueda del mouse
        self.image_canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.image_canvas.bind("<Button-4>", self.on_mousewheel)  # Linux
        self.image_canvas.bind("<Button-5>", self.on_mousewheel)  # Linux
        
        # Configurar drag and drop
        self.setup_drag_drop()
        
        # Frame para predicción
        prediction_frame = ttk.LabelFrame(main_frame, text="🔍 Predicción", padding=15)
        prediction_frame.pack(fill="x", pady=(0, 10))
        
        # Botón para hacer predicción
        self.predict_btn = ttk.Button(
            prediction_frame,
            text="🔍 Analizar Imagen",
            command=self.predict_image,
            state="disabled",
            style='Success.TButton'
        )
        self.predict_btn.pack(pady=(0, 10))
        
        # Frame para resultados
        results_frame = ttk.Frame(prediction_frame)
        results_frame.pack(fill="x")
        
        # Resultado principal
        self.result_label = ttk.Label(
            results_frame,
            text="",
            style='Header.TLabel'
        )
        self.result_label.pack(pady=(0, 5))
        
        # Confianza
        self.confidence_label = ttk.Label(
            results_frame,
            text="",
            style='Info.TLabel'
        )
        self.confidence_label.pack(pady=(0, 5))
        
        # Probabilidades detalladas
        self.probabilities_frame = ttk.Frame(results_frame)
        self.probabilities_frame.pack(fill="x", pady=(5, 0))
        
        # Barra de progreso
        self.progress = ttk.Progressbar(
            prediction_frame,
            mode='indeterminate'
        )
        self.progress.pack(fill="x", pady=(10, 0))
        self.progress.pack_forget()  # Ocultar inicialmente
        
        # Frame para estadísticas
        stats_frame = ttk.LabelFrame(main_frame, text="📊 Estadísticas", padding=15)
        stats_frame.pack(fill="x")
        
        # Estadísticas del modelo
        self.stats_text = tk.Text(
            stats_frame,
            height=4,
            font=("Consolas", 9),
            bg="#f8f9fa",
            fg="#495057"
        )
        self.stats_text.pack(fill="x")
        
        # Scrollbar para estadísticas
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        stats_scrollbar.pack(side="right", fill="y")
        self.stats_text.config(yscrollcommand=stats_scrollbar.set)
    
    def on_image_configure(self, event):
        """Actualizar scroll region cuando cambie el tamaño de la imagen"""
        self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
    
    def on_mousewheel(self, event):
        """Manejar zoom con la rueda del mouse"""
        if not self.original_image or not self.current_image_path:
            return
        
        try:
            # Factor de zoom
            zoom_factor = 1.1 if event.delta > 0 or event.num == 4 else 0.9
            
            # Calcular nueva escala
            new_scale = self.current_scale * zoom_factor
            
            # Limitar escala (mínimo 0.1, máximo 5.0)
            new_scale = max(0.1, min(5.0, new_scale))
            
            if new_scale == self.current_scale:
                return  # No hay cambio
            
            # Actualizar escala
            self.current_scale = new_scale
            
            # Calcular nuevas dimensiones
            original_width, original_height = self.original_image.size
            new_width = int(original_width * new_scale)
            new_height = int(original_height * new_scale)
            
            # Redimensionar imagen
            if new_scale != 1.0:
                image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                image = self.original_image
            
            # Convertir para tkinter
            self.current_image = ImageTk.PhotoImage(image)
            
            # Actualizar el label
            self.image_label.config(
                image=self.current_image,
                text="",
                width=new_width,
                height=new_height
            )
            
            # Actualizar información
            info_text = f"Imagen: {os.path.basename(self.current_image_path)}\nTamaño original: {original_width}x{original_height}\nTamaño mostrado: {new_width}x{new_height} (Escala: {new_scale:.2f})"
            
            if hasattr(self, 'info_label'):
                self.info_label.destroy()
            
            self.info_label = tk.Label(
                self.image_canvas,
                text=info_text,
                bg="#f8f9fa",
                fg="#495057",
                font=("Arial", 10),
                anchor="nw"
            )
            
            # Posicionar el label de información debajo de la imagen
            info_window = self.image_canvas.create_window(0, new_height + 10, anchor="nw", window=self.info_label)
            
            # Actualizar scroll region
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            
        except Exception as e:
            print(f"Error en zoom: {e}")
    
    
    def setup_drag_drop(self):
        """Configurar drag and drop para imágenes"""
        def on_drop(event):
            files = self.root.tk.splitlist(event.data)
            if files:
                file_path = files[0]
                if self.is_image_file(file_path):
                    self.current_image_path = file_path
                    self.display_image(file_path)
                    self.predict_btn.config(state="normal")
                else:
                    messagebox.showerror("Error", "Por favor selecciona un archivo de imagen válido")
        
        # Configurar drag and drop
        self.image_label.bind("<Button-1>", lambda e: self.select_image())
        self.root.bind("<B1-Motion>", lambda e: None)  # Prevenir comportamiento por defecto
    
    def is_image_file(self, file_path):
        """Verificar si el archivo es una imagen válida"""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        return any(file_path.lower().endswith(ext) for ext in valid_extensions)
    
    def load_model_auto(self):
        """Intenta cargar el modelo automáticamente si existe"""
        if os.path.exists(self.model_path):
            self.load_model()
    
    def browse_model(self):
        """Permitir al usuario buscar un archivo de modelo"""
        filetypes = [
            ("Modelos PyTorch", "*.pth *.pt"),
            ("Todos los archivos", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Seleccionar modelo",
            filetypes=filetypes,
            initialdir="."
        )
        
        if filename:
            self.model_path = filename
            self.model_path_var.set(filename)
            self.load_model()
    
    def load_model(self):
        """Carga el modelo entrenado"""
        try:
            # Mostrar barra de progreso
            self.progress.pack(fill="x", pady=(10, 0))
            self.progress.start()
            
            # Deshabilitar botones
            self.load_model_btn.config(state="disabled")
            self.browse_model_btn.config(state="disabled")
            
            # Cargar modelo en hilo separado
            def load_in_thread():
                try:
                    self.predictor = ChestXrayPredictor(self.model_path)
                    if self.predictor.model is not None:
                        self.root.after(0, self.on_model_loaded_success)
                    else:
                        self.root.after(0, self.on_model_loaded_error)
                except Exception as e:
                    self.root.after(0, lambda: self.on_model_loaded_error(str(e)))
            
            threading.Thread(target=load_in_thread, daemon=True).start()
            
        except Exception as e:
            self.on_model_loaded_error(str(e))
    
    def on_model_loaded_success(self):
        """Callback cuando el modelo se carga exitosamente"""
        self.progress.stop()
        self.progress.pack_forget()
        
        self.model_status.config(
            text="✅ Modelo cargado correctamente",
            style='Success.TLabel'
        )
        
        self.load_model_btn.config(state="normal")
        self.browse_model_btn.config(state="normal")
        self.select_image_btn.config(state="normal")
        # Los botones de imagen se habilitarán cuando se cargue una imagen
        
        # Actualizar estadísticas
        self.update_model_stats()
        
        messagebox.showinfo("Éxito", "Modelo cargado correctamente")
    
    def on_model_loaded_error(self, error_msg=""):
        """Callback cuando hay error al cargar el modelo"""
        self.progress.stop()
        self.progress.pack_forget()
        
        self.model_status.config(
            text="❌ Error cargando modelo",
            style='Error.TLabel'
        )
        
        self.load_model_btn.config(state="normal")
        self.browse_model_btn.config(state="normal")
        
        messagebox.showerror("Error", f"Error cargando el modelo:\n{error_msg}")
    
    def update_model_stats(self):
        """Actualizar estadísticas del modelo"""
        if self.predictor and self.predictor.model:
            try:
                # Contar parámetros
                total_params = sum(p.numel() for p in self.predictor.model.parameters())
                trainable_params = sum(p.numel() for p in self.predictor.model.parameters() if p.requires_grad)
                frozen_params = total_params - trainable_params
                
                # Información del dispositivo
                device = next(self.predictor.model.parameters()).device
                
                # Crear texto de estadísticas
                stats_text = f"""Modelo: DenseNet-121
Parámetros totales: {total_params:,}
Parámetros entrenables: {trainable_params:,}
Parámetros congelados: {frozen_params:,}
Dispositivo: {device}
Clases: {list(self.predictor.idx_to_class.values())}"""
                
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(1.0, stats_text)
                
            except Exception as e:
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(1.0, f"Error obteniendo estadísticas: {str(e)}")
    
    def select_image(self):
        """Permite al usuario seleccionar una imagen"""
        filetypes = [
            ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
            ("PNG", "*.png"),
            ("JPEG", "*.jpg *.jpeg"),
            ("BMP", "*.bmp"),
            ("TIFF", "*.tiff *.tif"),
            ("Todos los archivos", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=filetypes
        )
        
        if filename:
            self.current_image_path = filename
            self.display_image(filename)
            self.predict_btn.config(state="normal")
    
    def display_image(self, image_path):
        """Muestra la imagen seleccionada en la GUI manteniendo dimensiones originales"""
        try:
            # Cargar imagen original y guardarla
            self.original_image = Image.open(image_path)
            original_width, original_height = self.original_image.size
            
            # Para el canvas, mostramos la imagen en tamaño original o ajustado
            # Calcular tamaño máximo para el área de visualización
            max_display_width = 800  # Aumentado para mejor visualización
            max_display_height = 600
            
            # Calcular factor de escala para ajustar a la ventana sin recortar
            scale_width = max_display_width / original_width
            scale_height = max_display_height / original_height
            scale_factor = min(scale_width, scale_height, 1.0)  # No agrandar si es más pequeña
            
            # Guardar factor de escala actual
            self.current_scale = scale_factor
            
            # Calcular nuevas dimensiones
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Redimensionar manteniendo proporción y calidad
            if scale_factor < 1.0:
                image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                image = self.original_image  # Usar imagen original si es más pequeña que el área
            
            # Convertir para tkinter
            self.current_image = ImageTk.PhotoImage(image)
            
            # Configurar el label con la imagen
            self.image_label.config(
                image=self.current_image,
                text="",
                width=new_width,
                height=new_height
            )
            
            # Mostrar información de la imagen
            info_text = f"Imagen: {os.path.basename(image_path)}\nTamaño original: {original_width}x{original_height}\nTamaño mostrado: {new_width}x{new_height} (Escala: {scale_factor:.2f})"
            
            # Crear un label separado para la información
            if hasattr(self, 'info_label'):
                self.info_label.destroy()
            
            self.info_label = tk.Label(
                self.image_canvas,
                text=info_text,
                bg="#f8f9fa",
                fg="#495057",
                font=("Arial", 10),
                anchor="nw"
            )
            
            # Posicionar el label de información debajo de la imagen
            info_window = self.image_canvas.create_window(0, new_height + 10, anchor="nw", window=self.info_label)
            
            # Actualizar scroll region
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            
            # Habilitar botones de imagen
            self.fit_image_btn.config(state="normal")
            self.original_size_btn.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error cargando imagen:\n{str(e)}")
            self.image_label.config(
                image="",
                text="Error cargando imagen"
            )
    
    def predict_image(self):
        """Hace la predicción de la imagen seleccionada"""
        if not self.predictor or not self.current_image_path:
            messagebox.showerror("Error", "Modelo no cargado o imagen no seleccionada")
            return
        
        try:
            # Mostrar barra de progreso
            self.progress.pack(fill="x", pady=(10, 0))
            self.progress.start()
            
            # Deshabilitar botones
            self.predict_btn.config(state="disabled")
            
            # Hacer predicción en hilo separado
            def predict_in_thread():
                try:
                    result = self.predictor.predict_single_image(
                        self.current_image_path, 
                        return_probabilities=True
                    )
                    self.root.after(0, lambda: self.on_prediction_complete(result))
                except Exception as e:
                    self.root.after(0, lambda: self.on_prediction_error(str(e)))
            
            threading.Thread(target=predict_in_thread, daemon=True).start()
            
        except Exception as e:
            self.on_prediction_error(str(e))
    
    def on_prediction_complete(self, result):
        """Callback cuando la predicción se completa exitosamente"""
        self.progress.stop()
        self.progress.pack_forget()
        
        # Mostrar resultado principal
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        
        # Configurar colores y emojis según la predicción
        if predicted_class == 'chest_xray':
            color = "#27ae60"  # Verde
            emoji = "🫁"
            class_name = "Radiografía de Tórax"
        else:
            color = "#e67e22"  # Naranja
            emoji = "🖼️"
            class_name = "Otra Imagen"
        
        self.result_label.config(
            text=f"{emoji} {class_name}",
            foreground=color
        )
        
        self.confidence_label.config(
            text=f"Confianza: {confidence:.1%}",
            foreground="#3498db"
        )
        
        # Mostrar probabilidades detalladas
        self.show_probabilities(result.get('probabilities', {}))
        
        # Habilitar botón
        self.predict_btn.config(state="normal")
        
        # Mostrar detalles en consola
        self.print_prediction_details(result)
    
    def show_probabilities(self, probabilities):
        """Mostrar probabilidades detalladas"""
        # Limpiar frame de probabilidades
        for widget in self.probabilities_frame.winfo_children():
            widget.destroy()
        
        if probabilities:
            # Título
            ttk.Label(
                self.probabilities_frame,
                text="Probabilidades detalladas:",
                style='Header.TLabel'
            ).pack(anchor="w")
            
            # Frame para barras de probabilidad
            prob_bars_frame = ttk.Frame(self.probabilities_frame)
            prob_bars_frame.pack(fill="x", pady=(5, 0))
            
            for class_name, prob in probabilities.items():
                # Frame para cada clase
                class_frame = ttk.Frame(prob_bars_frame)
                class_frame.pack(fill="x", pady=2)
                
                # Nombre de la clase
                class_label = ttk.Label(class_frame, text=f"{class_name}:", width=15)
                class_label.pack(side="left")
                
                # Barra de progreso
                progress = ttk.Progressbar(
                    class_frame,
                    length=200,
                    mode='determinate',
                    value=prob * 100
                )
                progress.pack(side="left", padx=(5, 10))
                
                # Porcentaje
                percent_label = ttk.Label(class_frame, text=f"{prob:.1%}")
                percent_label.pack(side="left")
    
    def print_prediction_details(self, result):
        """Imprimir detalles de la predicción en consola"""
        print(f"\n{'='*60}")
        print(f"🔍 PREDICCIÓN COMPLETADA")
        print(f"{'='*60}")
        print(f"Imagen: {os.path.basename(self.current_image_path)}")
        print(f"Predicción: {result['predicted_class']}")
        print(f"Confianza: {result['confidence']:.1%}")
        print(f"Es radiografía de tórax: {'Sí' if result['is_chest_xray'] else 'No'}")
        
        if 'probabilities' in result:
            print(f"\nProbabilidades detalladas:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.1%}")
        
        print(f"{'='*60}")
    
    def on_prediction_error(self, error_msg):
        """Callback cuando hay error en la predicción"""
        self.progress.stop()
        self.progress.pack_forget()
        
        self.result_label.config(text="❌ Error en predicción", foreground="#e74c3c")
        self.confidence_label.config(text="")
        
        # Limpiar probabilidades
        for widget in self.probabilities_frame.winfo_children():
            widget.destroy()
        
        self.predict_btn.config(state="normal")
        
        messagebox.showerror("Error", f"Error en la predicción:\n{error_msg}")
    
    def fit_image(self):
        """Ajustar la imagen al tamaño de la ventana"""
        if not self.original_image or not self.current_image_path:
            return
        
        try:
            original_width, original_height = self.original_image.size
            
            # Calcular tamaño máximo para el área de visualización
            max_display_width = 800
            max_display_height = 600
            
            # Calcular factor de escala para ajustar a la ventana sin recortar
            scale_width = max_display_width / original_width
            scale_height = max_display_height / original_height
            scale_factor = min(scale_width, scale_height, 1.0)
            
            # Actualizar escala actual
            self.current_scale = scale_factor
            
            # Calcular nuevas dimensiones
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Redimensionar
            if scale_factor < 1.0:
                image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                image = self.original_image
            
            # Convertir para tkinter
            self.current_image = ImageTk.PhotoImage(image)
            
            # Actualizar el label
            self.image_label.config(
                image=self.current_image,
                text="",
                width=new_width,
                height=new_height
            )
            
            # Actualizar información
            info_text = f"Imagen: {os.path.basename(self.current_image_path)}\nTamaño original: {original_width}x{original_height}\nTamaño mostrado: {new_width}x{new_height} (Escala: {scale_factor:.2f})"
            
            if hasattr(self, 'info_label'):
                self.info_label.destroy()
            
            self.info_label = tk.Label(
                self.image_canvas,
                text=info_text,
                bg="#f8f9fa",
                fg="#495057",
                font=("Arial", 10),
                anchor="nw"
            )
            
            # Posicionar el label de información debajo de la imagen
            info_window = self.image_canvas.create_window(0, new_height + 10, anchor="nw", window=self.info_label)
            
            # Actualizar scroll region
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            
        except Exception as e:
            messagebox.showerror("Error", f"Error ajustando imagen:\n{str(e)}")
    
    def show_original_size(self):
        """Mostrar la imagen en su tamaño original"""
        if not self.original_image or not self.current_image_path:
            return
        
        try:
            original_width, original_height = self.original_image.size
            
            # Actualizar escala actual
            self.current_scale = 1.0
            
            # Usar imagen original
            self.current_image = ImageTk.PhotoImage(self.original_image)
            
            # Actualizar el label
            self.image_label.config(
                image=self.current_image,
                text="",
                width=original_width,
                height=original_height
            )
            
            # Actualizar información
            info_text = f"Imagen: {os.path.basename(self.current_image_path)}\nTamaño original: {original_width}x{original_height}\nTamaño mostrado: {original_width}x{original_height} (Escala: 1.00)"
            
            if hasattr(self, 'info_label'):
                self.info_label.destroy()
            
            self.info_label = tk.Label(
                self.image_canvas,
                text=info_text,
                bg="#f8f9fa",
                fg="#495057",
                font=("Arial", 10),
                anchor="nw"
            )
            
            # Posicionar el label de información debajo de la imagen
            info_window = self.image_canvas.create_window(0, original_height + 10, anchor="nw", window=self.info_label)
            
            # Actualizar scroll region
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            
        except Exception as e:
            messagebox.showerror("Error", f"Error mostrando tamaño original:\n{str(e)}")

def main():
    """Función principal para ejecutar la GUI"""
    # Verificar dependencias
    try:
        import torch
        import PIL
    except ImportError as e:
        print(f"❌ Error: Dependencia faltante: {e}")
        print("Por favor instala las dependencias con: pip install -r requirements.txt")
        return
    
    # Crear y ejecutar GUI
    root = tk.Tk()
    app = ChestXrayGUI(root)
    
    # Centrar ventana
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Configurar cierre
    def on_closing():
        if messagebox.askokcancel("Salir", "¿Estás seguro de que quieres salir?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Ejecutar
    root.mainloop()

if __name__ == "__main__":
    main()
