import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as tb
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from PIL import Image, ImageTk 
import warnings

# Ignorar warnings específicos
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class SplashScreen(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.title("Iniciando...")
        
        # Hacer la ventana del tamaño deseado y centrarla
        width = 400
        height = 300
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(f'{width}x{height}+{x}+{y}')
        
        # Configurar la ventana
        self.overrideredirect(True)
        self.configure(bg='white')
        
        # Frame principal
        main_frame = tk.Frame(self, bg='white')
        main_frame.pack(expand=True, fill='both')
        
        try:
            # Cargar y mostrar el icono
            icon_path = Path("icono.ico")
            if icon_path.exists():
                icon_image = Image.open(icon_path)
                icon_image.thumbnail((150, 150))
                icon_photo = ImageTk.PhotoImage(icon_image)
                icon_label = tk.Label(main_frame, image=icon_photo, bg='white')
                icon_label.image = icon_photo
                icon_label.pack(pady=20)
        except Exception as e:
            print(f"Error al cargar el icono: {e}")
        
        # Título
        title_label = tk.Label(
            main_frame,
            text="Iniciando Calculadora\nPredictora de Cultivos",
            font=("Segoe UI", 16, "bold"),
            bg='white'
        )
        title_label.pack(pady=10)
        
        # Barra de progreso
        self.progress = ttk.Progressbar(
            main_frame,
            length=300,
            mode='indeterminate'
        )
        self.progress.pack(pady=20)
        self.progress.start(15)

    def destroy(self):
        """Sobrescribir destroy"""
        try:
            self.progress.stop()
        except:
            pass
        super().destroy()

class ModernCropPredictor(tb.Window):
    def __init__(self):
        # Primero inicializar la ventana principal
        super().__init__(themename="litera")
        
        # Mostrar splash screen después de la ventana principal
        self.splash = None
        self.setup_splash()
        
        try:
            # Configuración básica de la ventana
            self.title("Predictor de Probabilidad de Compra de Cultivos")
            self.geometry("1200x800")

            # Inicializar variables de instancia
            self.canvas_size = 200
            self.input_widgets = {}
            self._after_id = None
            self.result_canvas = None
            self.probability_label = None
            self.message_label = None
            self.variables_frame = None
            self.results_frame = None
            self.models = {}
            self.scalers = {}

            # Definir el orden de variables para el modelo
            self.MODEL_VARIABLES = {
                'colza': {
                    'Ventas_Colza_N-1': (0, 1000, 500, 0),  # Cambiado a entero
                    'Visitas_Colza_N': (0, 5, 2, 0),
                    'Reclamaciones_Colza_N-1': (0, 20, 5, 0),  # Cambiado a entero
                    'Jornada_Campo_Colza_N': (0, 3, 1, 0),
                    'Potencial_Colza_has.': (0, 1000, 500, 2),
                    'CuotaMercado_Zona_Colza': (0, 100, 50, 2),
                    'Rendimiento_Colza_N (kg/ha)': (1000, 5000, 3000, 2),
                    'PrecAcum Septiembre 2024': (0, 1000, 500, 2)
                },
                'girasol': {
                    'Ventas_Girasol_N-1': (0, 1000, 500, 0),  # Cambiado a entero
                    'Visitas_Girasol_N': (0, 5, 2, 0),
                    'Reclamaciones_Girasol_N-1': (0, 20, 5, 0),  # Cambiado a entero
                    'Jornada_Campo_Girasol_N': (0, 3, 1, 0),
                    'CuotaMercado_Zona_Girasol': (0, 100, 50, 2),
                    'Potencial_Girasol_has.': (0, 1000, 500, 2),
                    'Rendimiento_Girasol_N (kg/ha)': (1000, 5000, 3000, 2),
                    'PrecAcum  (Andalucia:EneMar2024 Resto:MarMay2024)': (0, 1000, 500, 2)
                },
                'maiz': {
                    'Ventas_Maiz_N-1': (0, 1000, 500, 0),  # Cambiado a entero
                    'Visitas_Maiz_N': (0, 5, 2, 0),
                    'Reclamaciones_Maiz_N-1': (0, 20, 5, 0),  # Cambiado a entero
                    'Jornada_Campo_Maiz_N': (0, 3, 1, 0),
                    'CuotaMercado_Zona_Maiz': (0, 100, 50, 2),
                    'Potencial_Maiz_has.': (0, 1000, 500, 2),
                    'Rendimiento_Maiz_N (kg/ha)': (1000, 5000, 3000, 2),
                    '% embalse abril N': (0, 100, 50, 2)
                }
            }

            # Definir el orden de visualización
            self.DISPLAY_ORDER = {
                'colza': [
                    'Visitas_Colza_N',
                    'Ventas_Colza_N-1',
                    'Potencial_Colza_has.',
                    'PrecAcum Septiembre 2024',
                    'CuotaMercado_Zona_Colza',
                    'Rendimiento_Colza_N (kg/ha)',
                    'Reclamaciones_Colza_N-1',
                    'Jornada_Campo_Colza_N'
                ],
                'girasol': [
                    'Visitas_Girasol_N',
                    'PrecAcum  (Andalucia:EneMar2024 Resto:MarMay2024)',
                    'CuotaMercado_Zona_Girasol',
                    'Ventas_Girasol_N-1',
                    'Potencial_Girasol_has.',
                    'Rendimiento_Girasol_N (kg/ha)',
                    'Reclamaciones_Girasol_N-1',
                    'Jornada_Campo_Girasol_N'
                ],
                'maiz': [
                    'Ventas_Maiz_N-1',
                    'Rendimiento_Maiz_N (kg/ha)',
                    'Potencial_Maiz_has.',
                    'Visitas_Maiz_N',
                    'CuotaMercado_Zona_Maiz',
                    '% embalse abril N',
                    'Jornada_Campo_Maiz_N',
                    'Reclamaciones_Maiz_N-1'
                ]
            }

            # Cargar modelos y configurar GUI
            self.load_models()
            self.setup_gui()
            
            # Asegurar que la ventana principal esté lista
            self.update()

            # Cargar icono
            try:
                self.iconbitmap("icono.ico")
            except:
                pass

            # Programar la destrucción del splash screen
            if self.splash:
                self.after(1000, self.destroy_splash)
            
        except Exception as e:
            if self.splash:
                self.splash.destroy()
            messagebox.showerror("Error", f"Error al iniciar la aplicación: {str(e)}")
            raise e
        
    def setup_splash(self):
        """Configurar y mostrar splash screen"""
        try:
            self.splash = SplashScreen()
            self.splash.update()
        except:
            self.splash = None

    def destroy_splash(self):
            """Destruir splash screen de forma segura"""
            try:
                if self.splash:
                    self.splash.destroy()
                    self.splash = None
            except:
                pass

    def load_models(self):
        """Cargar modelos y scalers"""
        model_files = {
            'colza': 'Colza_XGBoost_Con_SMOTE.joblib',
            'girasol': 'Girasol_XGBoost_Sin_SMOTE.joblib',
            'maiz': 'Maiz_XGBoost_Con_SMOTE.joblib'
        }
        
        scaler_files = {
            'colza': 'scaler_colza.joblib',
            'girasol': 'scaler_girasol.joblib',
            'maiz': 'scaler_maiz.joblib'
        }
        
        for crop in ['colza', 'girasol', 'maiz']:
            # Verificar que existan los archivos
            model_path = Path(model_files[crop])
            scaler_path = Path(scaler_files[crop])
            
            if not model_path.exists():
                raise FileNotFoundError(f"No se encuentra el archivo del modelo: {model_files[crop]}")
            if not scaler_path.exists():
                raise FileNotFoundError(f"No se encuentra el archivo del scaler: {scaler_files[crop]}")
            
            # Cargar modelo y scaler
            self.models[crop] = joblib.load(model_path)
            self.scalers[crop] = joblib.load(scaler_path)

    def setup_gui(self):
        """Configurar toda la interfaz gráfica"""
        # Notebook principal
        self.notebook = tb.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Pestañas principales
        self.individual_tab = tb.Frame(self.notebook)
        self.batch_tab = tb.Frame(self.notebook)
        
        self.notebook.add(self.individual_tab, text=" Predicción Individual ")
        self.notebook.add(self.batch_tab, text=" Predicción Masiva ")
        
        # Configurar pestañas
        self.setup_individual_tab()
        self.setup_batch_tab()

    def setup_individual_tab(self):
        """Configurar pestaña de predicción individual"""
        # División en dos columnas
        left_frame = tb.Frame(self.individual_tab)
        right_frame = tb.Frame(self.individual_tab)
        
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Selector de cultivo
        crop_frame = tb.LabelFrame(left_frame, text="Seleccionar Cultivo", padding=10)
        crop_frame.pack(fill="x", pady=10)
        
        self.crop_var = tk.StringVar(value="colza")
        crop_combo = tb.Combobox(
            crop_frame,
            textvariable=self.crop_var,
            values=list(self.MODEL_VARIABLES.keys()),  # Cambiado aquí
            state="readonly"
        )
        crop_combo.pack(fill="x")
        crop_combo.bind('<<ComboboxSelected>>', lambda e: self.on_crop_change())
        
        # Frame de variables
        self.variables_frame = tb.LabelFrame(left_frame, text="Variables de Entrada", padding=10)
        self.variables_frame.pack(fill="both", expand=True, pady=10)
        
        # Frame de resultados
        self.results_frame = tb.LabelFrame(right_frame, text="Resultados", padding=10)
        self.results_frame.pack(fill="both", expand=True)
        
        # Canvas de resultados
        self.result_canvas = tk.Canvas(
            self.results_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='white',
            highlightthickness=0
        )
        self.result_canvas.pack(pady=20)
        
        # Labels de resultados
        self.probability_label = tb.Label(
            self.results_frame,
            text="0%",
            font=("Segoe UI", 20, "bold")
        )
        self.probability_label.pack(pady=10)
        
        self.message_label = tb.Label(
            self.results_frame,
            text="",
            font=("Segoe UI", 12)
        )
        self.message_label.pack(pady=10)
        
        # Crear inputs iniciales
        self.create_variable_inputs("colza")
        
        # Dibujar progreso inicial
        self.draw_progress(0)

    # Modificar la función create_variable_inputs para usar el orden de visualización
    def create_variable_inputs(self, crop):
        """Crear inputs para las variables del cultivo seleccionado"""
        if not self.variables_frame:
            return
            
        # Limpiar frame anterior
        for widget in self.variables_frame.winfo_children():
            widget.destroy()
            
        self.input_widgets = {}
        
        # Usar el orden de visualización
        for var_name in self.DISPLAY_ORDER[crop]:
            min_val, max_val, default, decimales = self.MODEL_VARIABLES[crop][var_name]
            
            frame = tb.Frame(self.variables_frame)
            frame.pack(fill="x", pady=5, padx=5)
            
            # Label
            tb.Label(frame, text=var_name).pack(anchor="w")
            
            # Frame para entrada y slider
            control_frame = tb.Frame(frame)
            control_frame.pack(fill="x", expand=True)
            
            # Variable con tipo correcto
            if decimales == 0:
                var = tk.IntVar(value=int(default))
                min_val = int(min_val)
                max_val = int(max_val)
            else:
                var = tk.DoubleVar(value=default)
            
            self.input_widgets[var_name] = var
            
            # Entry
            entry = tb.Entry(control_frame, textvariable=var, width=10)
            entry.pack(side="left", padx=5)
            
            # Slider 
            slider = tb.Scale(
                control_frame,
                from_=min_val,
                to=max_val,
                variable=var,
                orient="horizontal"
            )
            slider.pack(side="left", fill="x", expand=True, padx=5)
            
            def update_value(event=None, v=var, d=decimales, s=slider):
                try:
                    value = float(v.get())
                    if d == 0:
                        value = int(value)
                        v.set(value)
                    else:
                        v.set(round(value, d))
                    self.schedule_update()
                except:
                    if d == 0:
                        v.set(int(default))
                    else:
                        v.set(default)
            
            # Bindear eventos
            slider.bind("<ButtonRelease-1>", update_value)
            slider.bind("<B1-Motion>", update_value)
            entry.bind("<FocusOut>", update_value)
            entry.bind("<Return>", update_value)
            
            # Establecer valor inicial
            update_value()

    def schedule_update(self):
        """Programar actualización de probabilidad"""
        if hasattr(self, '_after_id') and self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(100, self.calculate_probability)

    def calculate_probability(self):
        """Calcular y mostrar probabilidad"""
        try:    
            crop = self.crop_var.get().lower()
            input_data = {}
            
            # Recoger valores en el orden del modelo
            for var_name in self.MODEL_VARIABLES[crop].keys():
                try:
                    decimales = self.MODEL_VARIABLES[crop][var_name][3]
                    value = self.input_widgets[var_name].get()
                    if decimales == 0:
                        value = int(value)
                    else:
                        value = round(float(value), decimales)
                    input_data[var_name] = value
                except:
                    input_data[var_name] = self.MODEL_VARIABLES[crop][var_name][2]
            
            # Crear DataFrame con el orden correcto
            df = pd.DataFrame([input_data])
            df = df[list(self.MODEL_VARIABLES[crop].keys())]
            
            # Escalar y predecir
            X = df.values
            scaled_input = self.scalers[crop].transform(X)
            probability = self.models[crop].predict_proba(scaled_input)[0][1] * 100
            
            self.update_display(probability)
            
        except Exception as e:
            print(f"Error en el cálculo: {str(e)}")

    def on_crop_change(self):
        """Manejar cambio de cultivo"""
        self.create_variable_inputs(self.crop_var.get())
        self.calculate_probability()

    def on_value_change(self, decimales):
        """Manejar cambios en cualquier variable"""
        # Formatear el valor según los decimales
        for name, var in self.input_widgets.items():
            try:
                value = float(var.get())
                dec = self.CROP_VARIABLES[self.crop_var.get()][name][3]
                if dec == 0:
                    var.set(int(value))
                else:
                    var.set(round(value, dec))
            except:
                pass
        
        # Programar la actualización
        if hasattr(self, '_after_id') and self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(100, self.calculate_probability)

    def draw_progress(self, percentage):
        """Dibujar círculo de progreso"""
        if not self.result_canvas:
            return
            
        self.result_canvas.delete("all")
        
        center = self.canvas_size // 2
        radius = (self.canvas_size - 40) // 2
        
        # Círculo de fondo
        self.result_canvas.create_arc(
            center - radius, center - radius,
            center + radius, center + radius,
            start=0, extent=359.999,
            fill="#f0f0f0"
        )
        
        if percentage > 0:
            extent = 359.999 * (percentage / 100)
            color = "#28a745" if percentage >= 75 else "#007bff" if percentage >= 50 else "#dc3545"
            
            self.result_canvas.create_arc(
                center - radius, center - radius,
                center + radius, center + radius,
                start=-90, extent=extent,
                fill=color
            )
        
        # Círculo interior
        inner_radius = radius - 20
        self.result_canvas.create_oval(
            center - inner_radius, center - inner_radius,
            center + inner_radius, center + inner_radius,
            fill="white"
        )

    # Modificar la función calculate_probability para usar el orden del modelo
    def calculate_probability(self):
        """Calcular y mostrar probabilidad"""
        try:    
            crop = self.crop_var.get().lower()
            input_data = {}
            
            # Usar el orden del modelo para los datos
            for var_name in self.MODEL_VARIABLES[crop].keys():
                try:
                    value = float(self.input_widgets[var_name].get())  # Aquí está el cambio
                    dec = self.MODEL_VARIABLES[crop][var_name][3]
                    if dec == 0:
                        value = int(value)
                    else:
                        value = round(value, dec)
                    input_data[var_name] = value
                except:
                    input_data[var_name] = self.MODEL_VARIABLES[crop][var_name][2]
            
            # Crear DataFrame con el orden correcto del modelo
            df = pd.DataFrame([input_data])
            df = df[list(self.MODEL_VARIABLES[crop].keys())]  # Asegurar orden correcto
            
            # Escalar datos
            X = df.values
            scaled_input = self.scalers[crop].transform(X)
            
            # Predecir
            probability = self.models[crop].predict_proba(scaled_input)[0][1] * 100
            
            # Actualizar display
            self.update_display(probability)
            
        except Exception as e:
            print(f"Error en el cálculo: {str(e)}")
            import traceback
            traceback.print_exc() 

    def update_display(self, probability):
        """Actualizar la visualización de los resultados"""
        if not all([self.result_canvas, self.probability_label, self.message_label]):
            return
            
        self.draw_progress(probability)
        self.probability_label.configure(text=f"{probability:.1f}%")
        
        if probability >= 75:
            message = "Alta probabilidad de compra"
            bootstyle = "success"
        elif probability >= 50:
            message = "Probabilidad moderada de compra"
            bootstyle = "primary"
        else:
            message = "Baja probabilidad de compra"
            bootstyle = "danger"
        
        self.message_label.configure(text=message, bootstyle=bootstyle)

    def on_crop_change(self):
        """Manejar cambio de cultivo"""
        self.create_variable_inputs(self.crop_var.get())
        self.calculate_probability()

    def setup_batch_tab(self):
        """Configurar pestaña de predicción masiva"""
        # Frame para botones
        buttons_frame = tb.Frame(self.batch_tab, padding=10)
        buttons_frame.pack(fill="x")
        
        tb.Button(
            buttons_frame,
            text="Cargar Excel",
            bootstyle="primary",
            command=self.load_excel
        ).pack(side="left", padx=5)
        
        tb.Button(
            buttons_frame,
            text="Descargar Plantilla",
            bootstyle="secondary",
            command=self.download_template
        ).pack(side="left", padx=5)

        # Frame para resultados
        self.batch_results_frame = tb.Frame(self.batch_tab)
        self.batch_results_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def show_batch_results(self, df):
        """Mostrar resultados de predicción masiva"""
        # Limpiar frame anterior
        for widget in self.batch_results_frame.winfo_children():
            widget.destroy()
        
        # Crear frame contenedor con scrollbars
        container = tb.Frame(self.batch_results_frame)
        container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Crear canvas para permitir scroll
        canvas = tk.Canvas(container)
        scrollbar_y = tb.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar_x = tb.Scrollbar(container, orient="horizontal", command=canvas.xview)
        
        # Frame dentro del canvas para la tabla
        table_frame = tb.Frame(canvas)
        
        # Configurar scrollbars
        canvas.configure(
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set
        )
        
        # Crear Treeview
        columns = list(df.columns)
        tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=15
        )
        
        # Configurar columnas
        for col in columns:
            tree.heading(col, text=col)
            if col.startswith('Probabilidad_'):
                tree.column(col, width=100, minwidth=100)
            else:
                tree.column(col, width=150, minwidth=100)
        
        # Insertar datos
        for i, row in df.iterrows():
            values = []
            for col in columns:
                if col.startswith('Probabilidad_'):
                    try:
                        val = float(row[col])
                        values.append(f"{val:.2f}%")
                    except:
                        values.append(str(row[col]))
                else:
                    values.append(str(row[col]))
            tree.insert("", "end", values=values)
        
        # Empaquetar todo
        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Crear ventana en el canvas para el frame
        canvas.create_window((0, 0), window=table_frame, anchor="nw")
        
        # Actualizar el scroll region cuando cambia el tamaño
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            width = table_frame.winfo_reqwidth()
            height = table_frame.winfo_reqheight()
            canvas.config(width=width, height=height)
        
        table_frame.bind("<Configure>", on_frame_configure)
        
        # Empaquetar la tabla
        tree.pack(fill="both", expand=True)
        
        # Botón de exportar
        export_button = tb.Button(
            self.batch_results_frame,
            text="Exportar Resultados",
            bootstyle="success",
            command=lambda: self.offer_save_results(df)
        )
        export_button.pack(pady=10)
        
        # Bind mousewheel para scroll
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            
        def on_shift_mousewheel(event):
            canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        canvas.bind_all("<Shift-MouseWheel>", on_shift_mousewheel)
        
        # Hacer scroll horizontal con las flechas
        def on_arrow_scroll(event):
            if event.keysym == 'Left':
                canvas.xview_scroll(-1, "units")
            elif event.keysym == 'Right':
                canvas.xview_scroll(1, "units")
                
        canvas.bind_all("<Left>", on_arrow_scroll)
        canvas.bind_all("<Right>", on_arrow_scroll)

    def load_excel(self):
        """Cargar archivo Excel"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("Excel files", "*.xlsx")]
            )
            if filename:
                # Leer el Excel
                df = pd.read_excel(filename)
                results = df.copy()  # Crear copia para mantener datos originales
                
                # Procesar cada cultivo
                for crop in ['colza', 'girasol', 'maiz']:
                    variables = list(self.MODEL_VARIABLES[crop].keys())  # Cambiado aquí
                    
                    # Verificar variables necesarias
                    if all(var in df.columns for var in variables):
                        try:
                            # Preparar datos para el modelo
                            X = df[variables].values
                            X_scaled = self.scalers[crop].transform(X)
                            
                            # Predecir y agregar al DataFrame de resultados
                            probs = self.models[crop].predict_proba(X_scaled)[:, 1] * 100
                            results[f'Probabilidad_{crop.title()}'] = probs
                        except Exception as e:
                            print(f"Error procesando {crop}: {str(e)}")
                            results[f'Probabilidad_{crop.title()}'] = None
                
                # Mostrar resultados
                self.show_batch_results(results)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar el archivo: {str(e)}")
            messagebox.showerror("Error", f"Error al procesar el archivo: {str(e)}")

    def offer_save_results(self, df):
        """Ofrecer guardar los resultados"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")]
            )
            if filename:
                df.to_excel(filename, index=False)
                messagebox.showinfo("Éxito", "Resultados guardados correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar resultados: {str(e)}")

    def download_template(self):
        """Descargar plantilla Excel"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")]
            )
            if filename:
                all_variables = []
                for crop_vars in self.MODEL_VARIABLES.values():  # Cambiado aquí
                    all_variables.extend(crop_vars.keys())
                    
                df_template = pd.DataFrame(columns=['ID_Cliente'] + all_variables)
                df_template.to_excel(filename, index=False)
                
                messagebox.showinfo("Éxito", "Plantilla descargada correctamente")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al crear plantilla: {str(e)}")
            messagebox.showerror("Error", f"Error al crear plantilla: {str(e)}")

if __name__ == "__main__":
    try:
        app = ModernCropPredictor()
        app.mainloop()
    except FileNotFoundError as e:
        messagebox.showerror("Error", f"No se encontraron los archivos necesarios: {str(e)}")
    except Exception as e:
        messagebox.showerror("Error", f"Error al iniciar la aplicación: {str(e)}")