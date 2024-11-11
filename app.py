import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import shap
import io

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Compra de Cultivos",
    page_icon="icono.ico",  # Cambia esto por la ruta de tu imagen
    layout="wide"
)

# Definir el directorio y cargar modelos/scalers al inicio
MODELS_DIR = Path(os.getcwd())

# Diccionario base para los valores de cada variable
# Definición de valores para cada variable, de modo que se usen en ambas configuraciones
VARIABLES_VALUES = {
    'Ventas_Colza_N-1': (0, 127, 1),
    'Visitas_Colza_N': (0, 14, 0),
    'Reclamaciones_Colza_N-1': ("No", "Sí", "Sí"),  # Cambiado a Sí/No
    'Jornada_Campo_Colza_N': ("No", "Sí", "No"),    # Cambiado a Sí/No
    'Potencial_Colza_has.': (0, 800, 5),
    'CuotaMercado_Zona_Colza': (0, 100, 0),
    'Rendimiento_Colza_N (kg/ha)': (0, 4666, 1600),
    'PrecAcum Septiembre 2024': (0, 300, 34),
    'Ventas_Girasol_N-1': (0, 487, 5),
    'Visitas_Girasol_N': (0, 48, 0),
    'Reclamaciones_Girasol_N-1': ("No", "Sí", "Sí"),  # Cambiado a Sí/No
    'Jornada_Campo_Girasol_N': ("No", "Sí", "No"),    # Cambiado a Sí/No
    'CuotaMercado_Zona_Girasol': (0, 100, 0),
    'Potencial_Girasol_has.': (1, 6200, 40),
    'Rendimiento_Girasol_N (kg/ha)': (0, 2431, 1050),
    'PrecAcum  (Andalucia:EneMar2024 Resto:MarMay2024)': (0, 3000, 350),
    'Ventas_Maiz_N-1': (0, 1033, 4),
    'Visitas_Maiz_N': (0, 46, 0),
    'Reclamaciones_Maiz_N-1': ("No", "Sí", "Sí"),  # Cambiado a Sí/No
    'Jornada_Campo_Maiz_N': ("No", "Sí", "No"),    # Cambiado a Sí/No
    'CuotaMercado_Zona_Maiz': (0, 100, 0),
    'Potencial_Maiz_has.': (1, 835, 15),
    'Rendimiento_Maiz_N (kg/ha)': (0, 55000, 20000),
    '% embalse abril N': (0, 100, 77)
}

# Añadir este diccionario después de VARIABLES_VALUES
VARIABLE_DISPLAY_NAMES = {
    'Ventas_Colza_N-1': 'Ventas Colza Año Anterior (dosis)',
    'Visitas_Colza_N': 'Visitas Colza Año Actual',
    'Reclamaciones_Colza_N-1': 'Reclamaciones Colza Año Anterior',
    'Jornada_Campo_Colza_N': 'Jornada Campo Colza Año Actual',
    'Potencial_Colza_has.': 'Potencial Colza (has)',
    'CuotaMercado_Zona_Colza': 'Cuota Mercado Zona Colza (%)',
    'Rendimiento_Colza_N (kg/ha)': 'Rendimiento Colza (kg/ha)',
    'PrecAcum Septiembre 2024': 'Precipitación Acumulada Sept. 2024 (mm)',
    
    'Ventas_Girasol_N-1': 'Ventas Girasol Año Anterior (dosis)',
    'Visitas_Girasol_N': 'Visitas Girasol Año Actual',
    'Reclamaciones_Girasol_N-1': 'Reclamaciones Girasol Año Anterior',
    'Jornada_Campo_Girasol_N': 'Jornada Campo Girasol Año Actual',
    'CuotaMercado_Zona_Girasol': 'Cuota Mercado Zona Girasol (%)',
    'Potencial_Girasol_has.': 'Potencial Girasol (has)',
    'Rendimiento_Girasol_N (kg/ha)': 'Rendimiento Girasol (kg/ha)',
    'PrecAcum  (Andalucia:EneMar2024 Resto:MarMay2024)': 'Precipitación Acumulada 2024 (mm)',
    
    'Ventas_Maiz_N-1': 'Ventas Maíz Año Anterior (dosis)',
    'Visitas_Maiz_N': 'Visitas Maíz Año Actual',
    'Reclamaciones_Maiz_N-1': 'Reclamaciones Maíz Año Anterior',
    'Jornada_Campo_Maiz_N': 'Jornada Campo Maíz Año Actual',
    'CuotaMercado_Zona_Maiz': 'Cuota Mercado Zona Maíz (%)',
    'Potencial_Maiz_has.': 'Potencial Maíz (has)',
    'Rendimiento_Maiz_N (kg/ha)': 'Rendimiento Maíz (kg/ha)',
    '% embalse abril N': 'Nivel Embalse Abril (%)'
}
# Configuración original para el cálculo
CROP_VARIABLES_ORIGINAL = {
    'colza': [
        'Ventas_Colza_N-1',
        'Visitas_Colza_N',
        'Reclamaciones_Colza_N-1',
        'Jornada_Campo_Colza_N',
        'Potencial_Colza_has.',
        'CuotaMercado_Zona_Colza',
        'Rendimiento_Colza_N (kg/ha)',
        'PrecAcum Septiembre 2024'
    ],
    'girasol': [
        'Ventas_Girasol_N-1',
        'Visitas_Girasol_N',
        'Reclamaciones_Girasol_N-1',
        'Jornada_Campo_Girasol_N',
        'CuotaMercado_Zona_Girasol',
        'Potencial_Girasol_has.',
        'Rendimiento_Girasol_N (kg/ha)',
        'PrecAcum  (Andalucia:EneMar2024 Resto:MarMay2024)'
    ],
    'maiz': [
        'Ventas_Maiz_N-1',
        'Visitas_Maiz_N',
        'Reclamaciones_Maiz_N-1',
        'Jornada_Campo_Maiz_N',
        'CuotaMercado_Zona_Maiz',
        'Potencial_Maiz_has.',
        'Rendimiento_Maiz_N (kg/ha)',
        '% embalse abril N'
    ]
}

# Configuración para la visualización en el orden de importancia
CROP_VARIABLES_IMPORTANCE_ORDER = {
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


def create_template_excel():
    """Crear plantilla Excel con las variables requeridas"""
    columns = ['ID_Cliente'] + list(VARIABLES_VALUES.keys())
    df_template = pd.DataFrame(columns=columns)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_template.to_excel(writer, index=False, sheet_name='Datos')
        worksheet = writer.sheets['Datos']
        
        workbook = writer.book
        format_colza = workbook.add_format({'bg_color': '#E8F1E8'})
        format_girasol = workbook.add_format({'bg_color': '#FFF2CC'})
        format_maiz = workbook.add_format({'bg_color': '#E6E6FA'})
        
        for i, col in enumerate(df_template.columns):
            if 'Colza' in col:
                worksheet.write(0, i, col, format_colza)
            elif 'Girasol' in col:
                worksheet.write(0, i, col, format_girasol)
            elif 'Maiz' in col:
                worksheet.write(0, i, col, format_maiz)
    
    return output.getvalue()

def process_excel_file(uploaded_file, models, scalers):
    """
    Procesar archivo Excel y calcular probabilidades para cada cultivo.
    Maneja correctamente la conversión y escalado de la cuota de mercado.
    """
    df = pd.read_excel(uploaded_file)
    
    cultivos_procesados = []
    cultivos_faltantes = []
    
    # Procesar cada cultivo
    for cultivo in ['colza', 'girasol', 'maiz']:
        # Obtener las variables necesarias para el cultivo
        variables = CROP_VARIABLES_ORIGINAL[cultivo]
        potential_var = f'Potencial_{cultivo.title()}_has.'
        market_share_var = f'CuotaMercado_Zona_{cultivo.title()}'
        
        # Verificar si todas las variables están presentes y tienen datos
        if all(var in df.columns for var in variables) and not df[variables].isna().any().any():
            # Crear una copia del DataFrame para las modificaciones
            df_temp = df.copy()
            
            # Convertir cuota de mercado de porcentaje a decimal ANTES del escalado
            if market_share_var in df_temp.columns:
                df_temp[market_share_var] = df_temp[market_share_var] / 100
            
            # Crear máscara para potencial 0
            zero_potential_mask = df[potential_var] == 0
            
            # Preparar datos para el modelo
            X = df_temp[variables]
            
            # Escalar datos
            X_scaled = scalers[cultivo].transform(X)
            
            # Predecir probabilidades
            probabilities = models[cultivo].predict_proba(X_scaled)[:, 1] * 100
            
            # Establecer probabilidad 0 donde el potencial es 0
            probabilities[zero_potential_mask] = 0.0
            
            # Añadir resultados al DataFrame original
            df[f'Probabilidad_{cultivo.title()}'] = probabilities
            cultivos_procesados.append(cultivo.title())
        else:
            cultivos_faltantes.append(cultivo.title())
            # Agregar columna de probabilidad vacía
            df[f'Probabilidad_{cultivo.title()}'] = np.nan
    
    return df, cultivos_procesados, cultivos_faltantes


@st.cache_resource
def load_models_and_scalers():
    models = {}
    scalers = {}
    
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
    
    for crop, filename in model_files.items():
        model_path = MODELS_DIR / filename
        if model_path.exists():
            models[crop] = joblib.load(str(model_path))
    
    for crop, filename in scaler_files.items():
        scaler_path = MODELS_DIR / filename
        if scaler_path.exists():
            scalers[crop] = joblib.load(str(scaler_path))
    
    return models, scalers

def calculate_probability(selected_crop, input_values, models, scalers):
    """
    Calcula la probabilidad de compra usando el orden original de las variables.
    Maneja correctamente la conversión y escalado de la cuota de mercado.
    """
    # Comprobar el valor de potencial según el cultivo
    potential_vars = {
        'colza': 'Potencial_Colza_has.',
        'girasol': 'Potencial_Girasol_has.',
        'maiz': 'Potencial_Maiz_has.'
    }
    
    # Si el potencial es 0, retornar 0% de probabilidad
    if input_values[potential_vars[selected_crop]] == 0:
        return 0.0
    
    # Crear una copia del diccionario de valores para no modificar el original
    modified_values = input_values.copy()
    
    # Convertir la cuota de mercado de porcentaje a decimal ANTES del escalado
    market_share_var = f'CuotaMercado_Zona_{selected_crop.title()}'
    if market_share_var in modified_values:
        # Si el usuario ingresa 8 (%), lo convertimos a 0.08 antes del escalado
        modified_values[market_share_var] = modified_values[market_share_var] / 100
    
    # Continuar con el cálculo normal usando los valores modificados
    variables_ordered = CROP_VARIABLES_ORIGINAL[selected_crop]
    input_df = pd.DataFrame([modified_values])[variables_ordered]
    
    # Escalar los datos y hacer la predicción
    scaler = scalers[selected_crop]
    scaled_input = scaler.transform(input_df)
    model = models[selected_crop]
    prob = model.predict_proba(scaled_input)[0][1] * 100
    return prob


def create_crop_inputs(crop_name, variable_names):
    """
    Crear inputs para un cultivo específico con nombres de visualización mejorados.
    """
    inputs = {}
    st.subheader(f"📊 Variables para {crop_name.title()}")

    with st.container():
        for var_name in variable_names:
            min_val, max_val, default = VARIABLES_VALUES[var_name]
            
            # Usar el nombre de visualización para la variable
            display_name = VARIABLE_DISPLAY_NAMES.get(var_name, var_name)
            
            col1, col2 = st.columns([1, 3])
            
            is_market_share = "CuotaMercado_Zona" in var_name
            
            if min_val == "No" and max_val == "Sí":
                with col1:
                    selected_value = st.selectbox(
                        display_name,  # Usar nombre de visualización
                        options=["No", "Sí"],
                        index=0 if default == "No" else 1,
                        key=f"select_{var_name}",
                        label_visibility="visible"
                    )
                    inputs[var_name] = 1 if selected_value == "Sí" else 0
            else:
                with col1:
                    if is_market_share:
                        value = st.number_input(
                            display_name,  # Usar nombre de visualización
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(default),
                            step=0.01,
                            format="%.2f",
                            key=f"num_{var_name}"
                        )
                    else:
                        value = st.number_input(
                            display_name,  # Usar nombre de visualización
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(default),
                            step=0.1,
                            key=f"num_{var_name}"
                        )
                with col2:
                    if is_market_share:
                        inputs[var_name] = st.slider(
                            f"Deslizante {display_name}",  # Usar nombre de visualización
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(value),
                            step=0.01,
                            format="%.2f%%",
                            key=f"slider_{var_name}",
                            label_visibility="collapsed"
                        )
                    else:
                        inputs[var_name] = st.slider(
                            f"Deslizante {display_name}",  # Usar nombre de visualización
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(value),
                            step=0.1,
                            key=f"slider_{var_name}",
                            label_visibility="collapsed"
                        )
    
    return inputs
    

def display_feature_importance(model, feature_names):
    """
    Visualiza la importancia de características usando nombres de visualización mejorados.
    """
    # Obtener la importancia de características del modelo
    feature_importances = model.get_booster().get_score(importance_type="weight")
    
    # Convertir los nombres de características a nombres de visualización
    display_names = [VARIABLE_DISPLAY_NAMES.get(name, name) for name in feature_names]
    
    # Convertir a un DataFrame para organizar los datos
    importance_df = pd.DataFrame({
        'Variable': display_names,
        'Importancia (%)': [feature_importances.get(f, 0) * 100 for f in feature_names]
    }).sort_values(by='Importancia (%)', ascending=True)
    
    fig = px.bar(
        importance_df,
        x="Importancia (%)",
        y="Variable",
        orientation="h",
        title="Importancia de Variables",
        text="Importancia (%)"
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis_title="Variable", xaxis_title="Importancia (%)", showlegend=False)
    
    st.plotly_chart(fig)

def display_shap_contributions_as_percentage(model, input_values, feature_names):
    """
    Calcula y visualiza la contribución de cada variable usando nombres de visualización mejorados.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_values)[0]
    
    # Convertir los nombres de características a nombres de visualización
    display_names = [VARIABLE_DISPLAY_NAMES.get(name, name) for name in feature_names]
    
    shap_abs_values = np.abs(shap_values)
    total_contribution = np.sum(shap_abs_values)
    shap_percentages = (shap_abs_values / total_contribution) * 100
    
    shap_df = pd.DataFrame({
        "Variable": display_names,
        "Contribución (%)": shap_percentages
    }).sort_values(by="Contribución (%)", ascending=True)

    fig_shap_percent = px.bar(
        shap_df,
        x="Contribución (%)",
        y="Variable",
        orientation="h",
        title="Contribución de cada variable a la predicción (en %)",
        text="Contribución (%)"
    )
    fig_shap_percent.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_shap_percent.update_layout(yaxis_title="Variable", xaxis_title="Contribución (%)", showlegend=False)
    
    st.plotly_chart(fig_shap_percent)

def main():
    # Contenedor vacío para forzar la alineación
    placeholder = st.empty()

    # Cargar y mostrar el icono
    with placeholder.container():
        st.image("icono.ico", width=200)  # Ajusta el tamaño según sea necesario

    # Título de la aplicación
    st.title("🌾 Predictor de Probabilidad de Compra de Cultivos")    

    # Cargar modelos y escaladores
    models, scalers = load_models_and_scalers()
    
    if not models or not scalers:
        st.error("❌ No se pudieron cargar los modelos y escaladores necesarios.")
        st.stop()
    

    # Crear pestañas para la predicción individual y masiva
    tab1, tab2 = st.tabs(["Predicción Individual", "Predicción Masiva"])

    with tab1:
        # Seleccionar cultivo
        selected_crop = st.sidebar.selectbox(
            "Seleccionar Cultivo",
            list(CROP_VARIABLES_ORIGINAL.keys()),
            format_func=lambda x: x.title()
        )
                
        # Crear dos columnas principales
        col1, col2 = st.columns([1, 1])

        with col1:
            # Generar los inputs en el orden de importancia para la visualización
            input_values = create_crop_inputs(selected_crop, CROP_VARIABLES_IMPORTANCE_ORDER[selected_crop])
            
            # Convertir "Sí/No" de Reclamaciones y Jornada_Campo a 1/0
            if "Reclamaciones" in input_values:
                input_values["Reclamaciones"] = 1 if input_values["Reclamaciones"] == "Sí" else 0
            if "Jornada_Campo" in input_values:
                input_values["Jornada_Campo"] = 1 if input_values["Jornada_Campo"] == "Sí" else 0

        with col2:
            st.subheader("🎯 Probabilidad de Compra")

            try:
                # Verificar si el potencial es 0
                potential_vars = {
                    'colza': 'Potencial_Colza_has.',
                    'girasol': 'Potencial_Girasol_has.',
                    'maiz': 'Potencial_Maiz_has.'
                }
                is_potential_zero = input_values[potential_vars[selected_crop]] == 0
                
                # Calcular probabilidad
                prob = calculate_probability(selected_crop, input_values, models, scalers)
                
                # Mostrar resultado
                st.metric(
                    label=f"Probabilidad de compra para {selected_crop}",
                    value=f"{prob:.1f}%"
                )
                
                st.progress(prob / 100)
                
                if is_potential_zero:
                    st.warning("Baja probabilidad de compra")
                    st.warning("Si este cliente no tiene hectáreas potenciales de este cultivo, su probabilidad será 0")
                elif prob >= 75:
                    st.success("Alta probabilidad de compra")
                elif prob >= 50:
                    st.info("Probabilidad moderada de compra")
                else:
                    st.warning("Baja probabilidad de compra")
                
                # Calcular la importancia de variables
                model = models[selected_crop]
                importance = model.feature_importances_
                variables = list(CROP_VARIABLES_ORIGINAL[selected_crop])

                # Mostrar valores SHAP para la predicción específica
                st.subheader("📊 Importancia de Variables para la Predicción Actual (Valores SHAP)")
                display_shap_contributions_as_percentage(model, pd.DataFrame([input_values]), list(CROP_VARIABLES_ORIGINAL[selected_crop]))            
                           
                

            except Exception as e:
                st.error(f"Error en la predicción: {str(e)}")
                
    with tab2:
        st.subheader("📈 Predicción Masiva desde Excel")
        
        # Botón para descargar la plantilla
        st.download_button(
            label="📑 Descargar Plantilla Excel",
            data=create_template_excel(),
            file_name="plantilla_prediccion_cultivos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Subida de archivo
        uploaded_file = st.file_uploader("Cargar archivo Excel con datos", type=['xlsx'])
        
        if uploaded_file is not None:
            try:
                # Procesar archivo
                results, cultivos_procesados, cultivos_faltantes = process_excel_file(
                    uploaded_file, models, scalers
                )
                
                # Mostrar información de procesamiento
                st.write("### Resultado del procesamiento:")
                if cultivos_procesados:
                    st.success(f"✅ Cultivos procesados: {', '.join(cultivos_procesados)}")
                if cultivos_faltantes:
                    st.warning(f"⚠️ Cultivos no procesados (datos faltantes): {', '.join(cultivos_faltantes)}")
                
                # Mostrar resultados
                st.write("### Resultados:")
                st.dataframe(results)
                
                # Botón para descargar resultados
                output = io.BytesIO()
                results.to_excel(output, index=False)
                st.download_button(
                    label="💾 Descargar Resultados",
                    data=output.getvalue(),
                    file_name="resultados_prediccion_cultivos.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            except Exception as e:
                st.error(f"Error al procesar el archivo: {str(e)}")
                st.error("Asegúrate de usar la plantilla proporcionada y que los datos estén en el formato correcto.")

if __name__ == "__main__":
    main()
