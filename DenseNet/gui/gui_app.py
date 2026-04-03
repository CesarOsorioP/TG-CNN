"""
GUI moderna para predicción de radiografías de tórax con DenseNet
Interfaz gráfica intuitiva y amigable usando CustomTkinter.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image

# Intentar importar CustomTkinter
try:
    import customtkinter as ctk
except ImportError:
    print("❌ Error: CustomTkinter no está instalado.")
    print("Por favor instala las dependencias con: pip install customtkinter pillow")
    sys.exit(1)

# Agregar el directorio src al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from models.predict import ChestXrayPredictor
except ImportError as e:
    print(f"Advertencia: No se encontró el módulo predictor o sus dependencias. {e}")
    ChestXrayPredictor = None

# Configuración visual de la aplicación (Moderna, Dark mode preferido)
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class ChestXrayModernGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🔬 Analizador Inteligente de Radiografías")
        self.geometry("900x900")
        self.minsize(800, 750)
        
        # Variables de estado
        self.predictor = None
        self.current_image_path = None
        self.original_image = None
        self.model_path = ctk.StringVar(value="densenet_chest_xray_modelnew.pth")
        
        # Configurar grilla principal (El contenedor de la imagen tomará todo el espacio posible)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.create_widgets()
        
        # Empezar hilo automático para cargar modelo si existe
        self.after(500, self.load_model_auto)

    def create_widgets(self):
        # ----------------------------------------------------
        # 1. HEADER COUCH (Top) - Configuración del Modelo
        # ----------------------------------------------------
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, padx=25, pady=(20, 10), sticky="ew")
        self.header_frame.grid_columnconfigure(1, weight=1)

        title_label = ctk.CTkLabel(
            self.header_frame, 
            text="Analizador de Radiografías de Tórax", 
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 20))

        # Sección de carga de modelo
        model_label = ctk.CTkLabel(self.header_frame, text="📁 Modelo:", font=ctk.CTkFont(size=14, weight="bold"))
        model_label.grid(row=1, column=0, padx=(0, 10), sticky="w")

        self.model_entry = ctk.CTkEntry(
            self.header_frame, 
            textvariable=self.model_path, 
            placeholder_text="Ruta del modelo .pth",
            height=35
        )
        self.model_entry.grid(row=1, column=1, sticky="ew", padx=(0, 10))

        self.load_model_btn = ctk.CTkButton(
            self.header_frame, 
            text="🔄 Cargar Modelo", 
            width=130, height=35,
            font=ctk.CTkFont(weight="bold"),
            command=self.load_model
        )
        self.load_model_btn.grid(row=1, column=2, padx=(0, 15))

        self.model_status_label = ctk.CTkLabel(
            self.header_frame, 
            text="❌ Desconectado", 
            text_color="#e74c3c",
            font=ctk.CTkFont(weight="bold", size=14)
        )
        self.model_status_label.grid(row=1, column=3, sticky="e")

        # ----------------------------------------------------
        # 2. IMAGE VIEWER (Middle) - Centrado y Responsivo
        # ----------------------------------------------------
        self.image_container = ctk.CTkFrame(self, corner_radius=15, fg_color=("gray85", "gray15"))
        self.image_container.grid(row=1, column=0, padx=25, pady=5, sticky="nsew")
        self.image_container.grid_propagate(False)
        self.image_container.bind("<Configure>", self.on_image_resize)
        
        self.image_container.grid_rowconfigure(0, weight=1)
        self.image_container.grid_columnconfigure(0, weight=1)

        # Botón grande decorativo / de acción cuando no hay imagen
        self.placeholder_btn = ctk.CTkButton(
            self.image_container,
            text="📁\n\nHaz clic aquí para seleccionar\nuna imagen radiográfica",
            font=ctk.CTkFont(size=18, weight="bold"),
            width=300,
            height=200,
            corner_radius=15,
            fg_color="transparent",
            border_width=2,
            text_color=("gray40", "gray70"),
            hover_color=("gray80", "gray20"),
            command=self.select_image
        )
        self.placeholder_btn.grid(row=0, column=0)

        # Label para mostrar la imagen real
        self.image_label = ctk.CTkLabel(self.image_container, text="")
        
        self.ctk_image = None
        
        # ----------------------------------------------------
        # 3. ACTION CONTROLS (Bottom-Mid)
        # ----------------------------------------------------
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=2, column=0, padx=25, pady=15, sticky="ew")
        self.controls_frame.grid_columnconfigure(0, weight=1)
        self.controls_frame.grid_columnconfigure(1, weight=1)

        self.select_img_btn = ctk.CTkButton(
            self.controls_frame, 
            text="🖼️ Cambiar Imagen", 
            command=self.select_image,
            state="disabled",
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=("gray30", "gray60"),
            hover_color=("gray20", "gray50")
        )
        self.select_img_btn.grid(row=0, column=0, padx=(0, 10), sticky="ew")

        self.predict_btn = ctk.CTkButton(
            self.controls_frame, 
            text="✨ Analizar Imagen", 
            command=self.predict_image,
            state="disabled",
            fg_color="#27ae60",
            hover_color="#219653",
            text_color="white",
            height=45,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.predict_btn.grid(row=0, column=1, padx=(10, 0), sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self.controls_frame, mode="indeterminate", height=8)
        self.progress_bar.grid(row=1, column=0, columnspan=2, pady=(20, 0), sticky="ew")
        self.progress_bar.set(0)
        self.progress_bar.grid_remove() # Ocultar inicialmente

        # ----------------------------------------------------
        # 4. RESULTS DASHBOARD (Bottom)
        # ----------------------------------------------------
        self.results_frame = ctk.CTkFrame(self, corner_radius=15, fg_color=("gray90", "gray13"))
        self.results_frame.grid(row=3, column=0, padx=25, pady=(5, 25), sticky="ew")
        self.results_frame.grid_columnconfigure(0, weight=1)

        self.result_title = ctk.CTkLabel(
            self.results_frame, 
            text="Esperando imagen...", 
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=("gray40", "gray60")
        )
        self.result_title.pack(pady=(20, 5))

        self.confidence_label = ctk.CTkLabel(
            self.results_frame, 
            text="", 
            font=ctk.CTkFont(size=16)
        )
        self.confidence_label.pack(pady=(0, 10))

        # Contenedor para métricas / barras de progreso de probabilidad
        self.details_frame = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        self.details_frame.pack(fill="x", padx=30, pady=(5, 20))

    def on_image_resize(self, event=None):
        """Redimensionar la imagen eficientemente cuando la ventana cambia de tamaño."""
        if self.original_image and self.current_image_path and self.image_label.winfo_ismapped():
            if self.image_container.winfo_width() > 10:
                self.display_image(self.current_image_path)

    def load_model_auto(self):
        """Intenta cargar el modelo si el archivo por defecto existe (al iniciar)."""
        if os.path.exists(self.model_path.get()):
            self.load_model()
            
    def load_model(self):
        path = self.model_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error de Modelo", "No se pudo encontrar el archivo del modelo en la ruta especificada.")
            return

        # UI Updates correspondientes a la carga
        self.load_model_btn.configure(state="disabled")
        self.model_status_label.configure(text="⏳ Cargando...", text_color="#f39c12")
        self.progress_bar.grid()
        self.progress_bar.start()

        def _load_job():
            try:
                if ChestXrayPredictor is None:
                    raise Exception("No se pudo cargar 'ChestXrayPredictor'. Verifica src/models/predict.py.")
                self.predictor = ChestXrayPredictor(path)
                self.after(0, self.on_model_loaded_success)
            except Exception as e:
                self.after(0, lambda: self.on_model_loaded_error(str(e)))

        threading.Thread(target=_load_job, daemon=True).start()

    def on_model_loaded_success(self):
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.load_model_btn.configure(state="normal")
        self.model_status_label.configure(text="✅ Conectado", text_color="#2ecc71")
        
        self.select_img_btn.configure(state="normal")
        if self.current_image_path:
            self.predict_btn.configure(state="normal")

    def on_model_loaded_error(self, err_msg):
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.load_model_btn.configure(state="normal")
        self.model_status_label.configure(text="❌ Error", text_color="#e74c3c")
        messagebox.showerror("Fallo en la Carga", err_msg)

    def select_image(self):
        filetypes = [("Archivos de Imagen", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        filepath = filedialog.askopenfilename(title="Selecciona una Radiografía", filetypes=filetypes)
        
        if filepath:
            self.current_image_path = filepath
            self.original_image = None
            
            # Cambiar de placeholder a imagen real
            self.placeholder_btn.grid_remove()
            self.image_label.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
            
            # Reset UI States
            self.select_img_btn.configure(state="normal")
            if self.predictor is not None:
                self.predict_btn.configure(state="normal")
                
            self.result_title.configure(text="Imagen cargada, lista para analizar", text_color=("gray40", "gray70"))
            self.confidence_label.configure(text="")
            for widget in self.details_frame.winfo_children():
                widget.destroy()

            # Render de imagen
            self.display_image(filepath)

    def display_image(self, filepath):
        try:
            if not self.original_image:
                self.original_image = Image.open(filepath)
            
            # Dimensiones del contenedor garantizadas
            self.update_idletasks()
            container_w = self.image_container.winfo_width()
            container_h = self.image_container.winfo_height()
            
            # Defaults al inicializar vista
            if container_w <= 10 or container_h <= 10:
                container_w = 700
                container_h = 450
                
            max_w = container_w - 30
            max_h = container_h - 30

            orig_w, orig_h = self.original_image.size
            ratio = min(max_w/orig_w, max_h/orig_h)
            
            if ratio < 1.0: # Conservar proporciones solo para compactar
                new_w = int(orig_w * ratio)
                new_h = int(orig_h * ratio)
            else:
                new_w = orig_w
                new_h = orig_h

            # CustomTkinter CTkImage soporta alta definición dinámica
            self.ctk_image = ctk.CTkImage(
                light_image=self.original_image, 
                dark_image=self.original_image, 
                size=(max(new_w, 1), max(new_h, 1))
            )
            self.image_label.configure(image=self.ctk_image)
            
        except Exception as e:
            messagebox.showerror("Error con la Imagen", f"No se pudo cargar la imagen: {e}")

    def predict_image(self):
        if not self.predictor or not self.current_image_path:
            return

        self.predict_btn.configure(state="disabled")
        self.select_img_btn.configure(state="disabled")
        self.result_title.configure(text="Analizando radiografía...", text_color="#f39c12")
        self.confidence_label.configure(text="")
        
        for widget in self.details_frame.winfo_children():
            widget.destroy()

        self.progress_bar.grid()
        self.progress_bar.start()

        def _run_prediction():
            try:
                res = self.predictor.predict_single_image(self.current_image_path, return_probabilities=True)
                self.after(0, lambda: self.on_predict_success(res))
            except Exception as e:
                self.after(0, lambda: self.on_predict_error(str(e)))

        threading.Thread(target=_run_prediction, daemon=True).start()

    def on_predict_success(self, result):
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.predict_btn.configure(state="normal")
        self.select_img_btn.configure(state="normal")

        predicted_class = result['predicted_class']
        confidence = result['confidence']

        # Interpretar el análisis
        if predicted_class == 'chest_xray':
            color = "#2ecc71"  # Verde brillante 
            text = "🫁 Radiografía de Tórax"
        else:
            color = "#e67e22"  # Naranja
            text =  "No es una radiografía de tórax."

        self.result_title.configure(text=text, text_color=color)
        self.confidence_label.configure(text=f"Nivel de Confianza: {confidence:.2%}")

        # Configurar detalle de sub-probabilidades de manera limpia
        probs = result.get('probabilities', {})
        if probs:
            for i, (cls_name, prob) in enumerate(probs.items()):
                row_frame = ctk.CTkFrame(self.details_frame, fg_color="transparent")
                row_frame.pack(fill="x", pady=6)
                
                # Nombre de la clase
                lbl = ctk.CTkLabel(row_frame, text=cls_name.upper().replace("_", " "), width=150, anchor="e", font=ctk.CTkFont(weight="bold", size=13))
                lbl.pack(side="left", padx=(0, 15))
                
                # Barra dinámica de confianza
                bar = ctk.CTkProgressBar(row_frame, height=18, bg_color="transparent")
                bar.pack(side="left", fill="x", expand=True, padx=10)
                bar.set(prob)
                
                # Colores según qué tan probable es
                if prob > 0.8:
                    bar.configure(progress_color="#2ecc71")
                elif prob > 0.4:
                    bar.configure(progress_color="#f1c40f")
                else:
                    bar.configure(progress_color="#95a5a6")
                
                # Valor exacto porcentual
                val = ctk.CTkLabel(row_frame, text=f"{prob:.1%}", width=70, anchor="w", font=ctk.CTkFont(size=13))
                val.pack(side="left")

    def on_predict_error(self, err_msg):
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.predict_btn.configure(state="normal")
        self.select_img_btn.configure(state="normal")
        
        self.result_title.configure(text="❌ Hubo un error procesando", text_color="#e74c3c")
        messagebox.showerror("Error de Análisis", f"Fallo al hacer la predicción:\n{err_msg}")

def main():
    app = ChestXrayModernGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
