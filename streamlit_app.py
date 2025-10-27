"""
Clustering Espacial - Análisis de Datos Geológicos
Aplicación Streamlit para análisis interactivo con SKATER
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import plotly.graph_objects as go
import plotly.express as px

from skater_algorithm import run_skater

warnings.filterwarnings('ignore')

# Configuración de la página - DEBE SER LA PRIMERA LLAMADA A STREAMLIT
st.set_page_config(
    page_title="Clustering Espacial - SKATER",
    page_icon="🗺️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🗺️ Análisis de Clustering Espacial con SKATER")
st.markdown("**Spatial 'K'luster Analysis by Tree Edge Removal**")

# Sidebar para parámetros
st.sidebar.header("⚙️ Configuración de Parámetros")

# Información sobre el archivo
st.sidebar.markdown("### 📁 Configuración de Datos")

# Selector de archivo de datos
selected_file = st.sidebar.selectbox(
    "**Archivo de Datos**",
    options=["bd_Parametros cineticos_hom.csv", "1_recpeso.xlsx"],
    index=0,
    help="Selecciona el archivo de datos a analizar"
)

# Inicializar variables de datos en session_state
if 'selected_variable' not in st.session_state:
    st.session_state.selected_variable = None

# Intentar cargar preview de columnas según el archivo seleccionado
if selected_file == "bd_Parametros cineticos_hom.csv":
    try:
        df_preview = pd.read_csv('data/bd_Parametros cineticos_hom.csv', sep=';', encoding='latin-1', nrows=0)
        # Excluir columnas de coordenadas y metadatos
        available_vars = [col for col in df_preview.columns if col not in ['midx', 'midy', 'midz', 'compid', 'composito', 'holeid', 'SampleID', 'Campana']]
        if st.session_state.selected_variable not in available_vars:
            st.session_state.selected_variable = available_vars[0] if available_vars else None
        
        variable = st.sidebar.selectbox(
            "**Variable para Clustering**",
            options=available_vars,
            index=available_vars.index(st.session_state.selected_variable) if st.session_state.selected_variable in available_vars else 0,
            help="Selecciona la variable que se usará para el clustering"
        )
        st.session_state.selected_variable = variable
    except Exception as e:
        st.sidebar.error(f"Error al cargar columnas: {e}")
        variable = None

elif selected_file == "1_recpeso.xlsx":
    try:
        df_preview = pd.read_excel('data/1_recpeso.xlsx', sheet_name='recpeso', nrows=0)
        # Excluir columnas de coordenadas y metadatos
        available_vars = [col for col in df_preview.columns if col not in ['midx', 'midy', 'midz', 'compid']]
        if st.session_state.selected_variable not in available_vars:
            st.session_state.selected_variable = available_vars[0] if available_vars else None
        
        variable = st.sidebar.selectbox(
            "**Variable para Clustering**",
            options=available_vars,
            index=available_vars.index(st.session_state.selected_variable) if st.session_state.selected_variable in available_vars else 0,
            help="Selecciona la variable que se usará para el clustering"
        )
        st.session_state.selected_variable = variable
    except Exception as e:
        st.sidebar.error(f"Error al cargar columnas: {e}")
        variable = None

else:
    variable = None

# Guardar estado actual
prev_data_file = st.session_state.get('prev_data_file', None)
prev_variable_name = st.session_state.get('prev_variable_name', None)

st.session_state.data_file = selected_file
st.session_state.variable_name = variable

# Marcar que hay cambios si el archivo o variable cambió
if prev_data_file is not None:
    if prev_data_file != selected_file or prev_variable_name != variable:
        st.session_state.data_changed = True
    else:
        st.session_state.data_changed = False
else:
    st.session_state.data_changed = False

st.session_state.prev_data_file = selected_file
st.session_state.prev_variable_name = variable

# Parámetros configurables
st.sidebar.markdown("### 🔧 Parámetros SKATER")

# Número de clusters
n_clusters = st.sidebar.slider(
    "**Número de Clusters**",
    min_value=2,
    max_value=10,
    value=3,
    step=1,
    help="Número deseado de clusters para dividir los datos"
)

# K vecinos
k_vecinos = st.sidebar.slider(
    "**K Vecinos**",
    min_value=5,
    max_value=200,
    value=90,
    step=5,
    help="Número de vecinos más cercanos para crear la matriz de conectividad espacial"
)

# Floor (tamaño mínimo)
floor = st.sidebar.slider(
    "**Tamaño Mínimo por Cluster**",
    min_value=10,
    max_value=500,
    value=70,
    step=10,
    help="Número mínimo de puntos que debe tener cada cluster"
)

# Manejo de islas
islands = st.sidebar.selectbox(
    "**Manejo de Islas**",
    options=["ignore", "increase"],
    index=0,
    help="Cómo manejar puntos aislados"
)

# Función de disimilitud
dissimilarity_func = st.sidebar.selectbox(
    "**Función de Disimilitud**",
    options=["euclidean", "manhattan", "cosine"],
    index=0,
    help="Métrica para calcular distancias entre puntos"
)

# Modo de debugging
trace = st.sidebar.checkbox(
    "**Modo Debug**",
    value=False,
    help="Activa información detallada del proceso de clustering"
)

# Paleta de colores
st.sidebar.markdown("---")
color_palette = st.sidebar.selectbox(
    "**Paleta de Colores**",
    options=["dark", "deep", "muted", "bright", "pastel", "colorblind"],
    index=0,
    help="Paleta de colores para visualizar los diferentes clusters"
)

# Botón para ejecutar análisis
st.sidebar.markdown("---")
if st.sidebar.button("🚀 Ejecutar Análisis SKATER", type="primary"):
    st.session_state.run_analysis = True
    st.session_state.k_vecinos = k_vecinos
    st.session_state.floor = floor
    st.session_state.islands = islands
    st.session_state.dissimilarity_func = dissimilarity_func
    st.session_state.trace = trace

# Función para cargar datos
@st.cache_data
def load_data(data_file: str, variable_name: str):
    """Cargar datos desde archivo seleccionado
    
    Args:
        data_file: Nombre del archivo a cargar
        variable_name: Nombre de la variable a usar para clustering
    
    Returns:
        DataFrame con columnas x, y, z, variable
    """
    # Esta función se cacheará automáticamente por Streamlit
    # El cache se limpiará cuando cambien los parámetros
    try:
        # Cargar según el tipo de archivo
        if data_file == "bd_Parametros cineticos_hom.csv":
            df = pd.read_csv('data/bd_Parametros cineticos_hom.csv',
                            sep=';',
                            encoding='latin-1')
            
            # Verificar que la variable existe
            if variable_name not in df.columns:
                st.error(f"La variable '{variable_name}' no existe en el archivo")
                return None
                
        elif data_file == "1_recpeso.xlsx":
            df = pd.read_excel('data/1_recpeso.xlsx', sheet_name='recpeso')
            
            # Verificar que la variable existe
            if variable_name not in df.columns:
                st.error(f"La variable '{variable_name}' no existe en el archivo")
                return None
        else:
            st.error(f"Archivo '{data_file}' no reconocido")
            return None
        
        # Verificar que las columnas de coordenadas existen
        coord_cols = ['midx', 'midy', 'midz']
        if not all(col in df.columns for col in coord_cols):
            st.error("El archivo no contiene las columnas de coordenadas (midx, midy, midz)")
            return None
        
        # Seleccionar columnas necesarias
        df = df[coord_cols + [variable_name]].copy()
        df.columns = ['x', 'y', 'z', 'variable']
        
        # Eliminar filas con valores faltantes
        df = df.dropna()
        
        # Verificar que hay datos
        if len(df) == 0:
            st.warning("No hay datos válidos después de aplicar filtros")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None



# Función para crear gráficas 2D
def create_2d_plots(df, color_palette):
    """Crear gráficas 2D de proyecciones"""
    clusters = sorted(df['cluster'].unique())
    colors = sns.color_palette(color_palette, n_colors=len(clusters))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Proyección XY
    for i, cluster_id in enumerate(clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        axes[0].scatter(
            cluster_data['x']/1000, 
            cluster_data['y']/1000,
            color=colors[i],
            label=f'Cluster {cluster_id}',
            s=30,
            alpha=0.85
        )
    axes[0].set_xlabel('X (km)')
    axes[0].set_ylabel('Y (km)')
    axes[0].set_title('Proyección XY')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Proyección XZ
    for i, cluster_id in enumerate(clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        axes[1].scatter(
            cluster_data['x']/1000, 
            cluster_data['z']/1000,
            color=colors[i],
            label=f'Cluster {cluster_id}',
            s=30,
            alpha=0.85
        )
    axes[1].set_xlabel('X (km)')
    axes[1].set_ylabel('Z (km)')
    axes[1].set_title('Proyección XZ')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Proyección YZ
    for i, cluster_id in enumerate(clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        axes[2].scatter(
            cluster_data['y']/1000, 
            cluster_data['z']/1000,
            color=colors[i],
            label=f'Cluster {cluster_id}',
            s=30,
            alpha=0.85
        )
    axes[2].set_xlabel('Y (km)')
    axes[2].set_ylabel('Z (km)')
    axes[2].set_title('Proyección YZ')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    # Leyenda común
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=min(len(clusters), 8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    return fig

# Función para crear gráfica 3D interactiva con Plotly
def create_3d_plot(df, color_palette):
    """Crear gráfica 3D interactiva"""
    clusters = sorted(df['cluster'].unique())
    colors = px.colors.qualitative.Set1[:len(clusters)]
    
    fig = go.Figure()
    
    for i, cluster_id in enumerate(clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        fig.add_trace(go.Scatter3d(
            x=cluster_data['x'],
            y=cluster_data['y'],
            z=cluster_data['z'],
            mode='markers',
            marker=dict(
                size=5,
                color=colors[i],
                opacity=0.8
            ),
            name=f'Cluster {cluster_id}',
            text=[f'Cluster {cluster_id}<br>X: {x/1000:.2f}<br>Y: {y/1000:.2f}<br>Z: {z/1000:.2f}<br>Valor: {v:.2f}' 
                  for x, y, z, v in zip(cluster_data['x'], cluster_data['y'], cluster_data['z'], cluster_data['variable'])],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Clusters SKATER: Espacio 3D',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)'
        ),
        width=800,
        height=600
    )
    
    return fig

# Función para crear gráficas estadísticas
def create_statistical_plots(df, color_palette):
    """Crear gráficas de análisis estadístico"""
    clusters = sorted(df['cluster'].unique())
    colors = sns.color_palette(color_palette, n_colors=len(clusters))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Q-Q Plot lognormal
    for i, cluster_id in enumerate(clusters):
        cluster_data = df[df['cluster'] == cluster_id]['variable'].values
        cluster_data = cluster_data[np.isfinite(cluster_data)]
        cluster_data = cluster_data[cluster_data > 0]
        
        if len(cluster_data) > 0:
            data_sorted = np.sort(cluster_data)
            n = len(data_sorted)
            percentiles = (np.arange(1, n+1) - 0.5) / n * 100
            
            axes[0].scatter(data_sorted, percentiles, color=colors[i], 
                           label=f'Cluster {cluster_id}', 
                           alpha=0.5, edgecolors='k', s=40)
            
            # Línea de referencia lognormal
            mu = np.mean(np.log(data_sorted))
            sigma = np.std(np.log(data_sorted))
            x_ref = np.linspace(data_sorted.min(), data_sorted.max(), 100)
            y_ref = stats.norm.cdf(np.log(x_ref), mu, sigma) * 100
            
            axes[0].plot(x_ref, y_ref, color=colors[i], linestyle='-', 
                        linewidth=2, alpha=0.8)
    
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Variable (escala logarítmica)')
    axes[0].set_ylabel('Percentiles Lognormales')
    axes[0].set_title('Q-Q Plot Lognormal por Cluster')
    axes[0].set_ylim([0, 100])
    axes[0].grid(True, which='both', linestyle='--', alpha=0.3)
    
    # 2. Efecto Proporcional
    for i, cluster_id in enumerate(clusters):
        cluster_data = df[df['cluster'] == cluster_id]['variable']
        if len(cluster_data) > 0:
            mean = cluster_data.mean()
            std = cluster_data.std()
            axes[1].scatter(mean, std, color=colors[i], 
                          label=f'Cluster {cluster_id}', 
                          s=80, alpha=0.8, edgecolor='k')
            axes[1].annotate(f'n={len(cluster_data)}', 
                           (mean, std), 
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=9)
    
    # Línea de tendencia
    all_means = [df[df['cluster'] == c]['variable'].mean() for c in clusters]
    all_stds = [df[df['cluster'] == c]['variable'].std() for c in clusters]
    
    if len(all_means) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_means, all_stds)
        x_line = np.linspace(min(all_means) * 0.9, max(all_means) * 1.1, 100)
        y_line = slope * x_line + intercept
        axes[1].plot(x_line, y_line, 'k--', alpha=0.7, 
                    label=f'y = {slope:.2f}x + {intercept:.2f} (R² = {r_value**2:.2f})')
    
    axes[1].set_xlabel('Media')
    axes[1].set_ylabel('Desviación Estándar')
    axes[1].set_title('Efecto Proporcional por Cluster')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    # 3. Boxplot
    boxplot_data = [df[df['cluster'] == c]['variable'] for c in clusters]
    boxplot = axes[2].boxplot(boxplot_data, patch_artist=True)
    
    for i, box in enumerate(boxplot['boxes']):
        box.set_facecolor(colors[i])
        box.set_alpha(0.7)
        box.set_edgecolor('black')
    
    axes[2].set_xlabel('Cluster')
    axes[2].set_ylabel('Variable')
    axes[2].set_title('Boxplot por Cluster')
    axes[2].set_xticklabels([f'{c}' for c in clusters])
    axes[2].grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Leyenda común
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=min(len(clusters), 8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    return fig

# Función para crear histogramas
def create_histograms(df, color_palette):
    """Crear histogramas por cluster"""
    clusters = sorted(df['cluster'].unique())
    colors = sns.color_palette(color_palette, n_colors=len(clusters))
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    min_val = df['variable'].min()
    max_val = df['variable'].max()
    bins = np.linspace(min_val, max_val, 30)
    
    for i, cluster_id in enumerate(clusters):
        cluster_data = df[df['cluster'] == cluster_id]['variable']
        if len(cluster_data) > 0:
            mean = cluster_data.mean()
            std = cluster_data.std()
            ax.hist(cluster_data, bins=bins, alpha=0.6, color=colors[i], 
                    label=f'Cluster {cluster_id} (n={len(cluster_data)}, μ={mean:.2f}, σ={std:.2f})')
            ax.axvline(mean, color=colors[i], linestyle='--', linewidth=2)
    
    ax.set_title('Distribución de Valores por Cluster', fontsize=16, fontweight='bold')
    ax.set_xlabel('Valor de la Variable', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(title='Clusters', title_fontsize=12, fontsize=10, 
              loc='upper right', bbox_to_anchor=(1.15, 1), 
              frameon=True, facecolor='white', edgecolor='gray')
    
    plt.figtext(0.5, 0.01, 
                "Las líneas discontinuas representan la media de cada cluster. " +
                "La distribución muestra la variabilidad dentro de cada grupo.",
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    return fig

# Contenido principal
if 'run_analysis' in st.session_state and st.session_state.run_analysis:
    
    # Mostrar progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Cargar datos
    status_text.text("Cargando datos...")
    progress_bar.progress(20)
    
    # Obtener parámetros de configuración
    data_file = st.session_state.get('data_file', 'bd_Parametros cineticos_hom.csv')
    variable_name = st.session_state.get('variable_name', None)
    
    if variable_name is None:
        st.error("Por favor, selecciona una variable para el clustering")
        st.session_state.run_analysis = False
        st.stop()
    
    df = load_data(data_file, variable_name)
    
    if df is not None:
        # Ejecutar SKATER
        status_text.text("Ejecutando SKATER...")
        progress_bar.progress(60)
        
        # Obtener parámetros de SKATER desde session_state
        k_vec = st.session_state.get('k_vecinos', 90)
        fl = st.session_state.get('floor', 70)
        isl = st.session_state.get('islands', 'ignore')
        dissim = st.session_state.get('dissimilarity_func', 'euclidean')
        tr = st.session_state.get('trace', False)
        
        try:
            # Definir atributos para clustering (solo la variable)
            attrs_name = ['x',
            #  'y',
             'z', 'variable']
            
            # Ejecutar SKATER
            df_result, model = run_skater(
                df, 
                attrs_name=attrs_name,
                n_clusters=n_clusters,
                k_vecinos=k_vec,
                floor=fl,
                islands=isl,
                dissimilarity_func=dissim,
                trace=tr
            )
            algorithm_name = "SKATER"
        except Exception as e:
            st.error(f"Error al ejecutar SKATER: {e}")
            df_result, model = None, None
            algorithm_name = "SKATER"
        
        if df_result is not None:
            progress_bar.progress(100)
            status_text.text("¡Análisis completado!")
            
            # Mostrar estadísticas básicas
            st.header("📊 Resultados del Análisis")
            
            # Información de datos utilizados
            st.subheader("📁 Datos Analizados")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Archivo", st.session_state.get('data_file', 'N/A'))
            with col2:
                st.metric("Variable", st.session_state.get('variable_name', 'N/A'))
            st.divider()
            
            # Información de parámetros utilizados
            st.subheader("⚙️ Parámetros Utilizados")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("K Vecinos", k_vec)
                st.metric("Tamaño Mínimo", fl)
            with col2:
                st.metric("Clusters Deseados", n_clusters)
                st.metric("Manejo Islas", isl.title())
            with col3:
                st.metric("Función Disimilitud", dissim.title())
                st.metric("Modo Debug", "Sí" if tr else "No")
            with col4:
                st.metric("Algoritmo", algorithm_name)
                st.metric("Paleta Colores", color_palette.title())
            
            st.divider()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total de Puntos", len(df_result))
            with col2:
                st.metric("Clusters Encontrados", len(df_result['cluster'].unique()))
            with col3:
                st.metric("Variable Promedio", f"{df_result['variable'].mean():.2f}")
            with col4:
                st.metric("Desviación Estándar", f"{df_result['variable'].std():.2f}")
            
            # Distribución por clusters
            st.subheader("📈 Distribución por Clusters")
            cluster_counts = df_result['cluster'].value_counts().sort_index()
            st.bar_chart(cluster_counts)
            
            # Estadísticas detalladas
            st.subheader("📋 Estadísticas Detalladas por Cluster")
            stats_cluster = df_result.groupby('cluster')['variable'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ])
            stats_cluster['cv'] = (stats_cluster['std'] / stats_cluster['mean']) * 100
            st.dataframe(stats_cluster.round(3))
            
            # Visualizaciones
            st.header("🎨 Visualizaciones")
            
            # Gráficas 2D
            st.subheader("📐 Proyecciones 2D")
            fig_2d = create_2d_plots(df_result, color_palette)
            st.pyplot(fig_2d)
            
            # Gráfica 3D interactiva
            st.subheader("🌐 Visualización 3D Interactiva")
            fig_3d = create_3d_plot(df_result, color_palette)
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Gráficas estadísticas
            st.subheader("📊 Análisis Estadístico")
            fig_stats = create_statistical_plots(df_result, color_palette)
            st.pyplot(fig_stats)
            
            # Histogramas
            st.subheader("📈 Distribuciones por Cluster")
            fig_hist = create_histograms(df_result, color_palette)
            st.pyplot(fig_hist)
            
            # Resetear estado
            st.session_state.run_analysis = False
            
        else:
            st.error(f"Error al ejecutar el análisis {algorithm_name}")
    else:
        st.error("Error al cargar los datos")

else:
    # Página de inicio
    st.markdown("""
    ## 🎯 Bienvenido al Análisis de Clustering Espacial
    
    Esta aplicación permite realizar análisis de clustering espacial usando el algoritmo **SKATER**.
    
    **SKATER** (Spatial 'K'luster Analysis by Tree Edge Removal) es un algoritmo de regionalización espacial 
    basado en la poda de árboles de expansión. El algoritmo particiona el espacio en regiones contiguas 
    maximizando la homogeneidad interna y considerando las conexiones espaciales.
    
    ### 📋 Instrucciones:
    1. **Selecciona el archivo y variable** en la barra lateral
    2. **Configura los parámetros** del algoritmo SKATER
    3. **Haz clic en "Ejecutar Análisis SKATER"** para procesar los datos
    4. **Explora los resultados** en las diferentes secciones
    
    ### 🔧 Parámetros Explicados:
    
    **K Vecinos**: Define cuántos vecinos más cercanos se consideran para crear la matriz de conectividad espacial. 
    - Valores bajos (5-20): Conexiones más locales, clusters más pequeños y compactos
    - Valores medios (50-100): Balance entre conectividad local y global
    - Valores altos (150-200): Conexiones más amplias, clusters más grandes y dispersos
    
    **Número de Clusters**: Cantidad deseada de grupos espaciales.
    - El algoritmo intentará crear esta cantidad de clusters
    - Puede resultar en menos clusters si el tamaño mínimo no se cumple
    
    **Tamaño Mínimo por Cluster**: Número mínimo de puntos por cluster.
    - Clusters más pequeños serán fusionados con otros
    - Ayuda a evitar clusters muy pequeños o ruidosos
    - Reduce el número total de clusters si es muy restrictivo
    
    **Manejo de Islas**: Cómo tratar puntos aislados.
    - "ignore": Ignora puntos que no pueden conectarse al resto
    - "increase": Aumenta el número de clusters para incluir puntos aislados
    
    **Función de Disimilitud**: Métrica para calcular distancias entre atributos.
    - "euclidean": Distancia euclidiana estándar (recomendada para datos numéricos)
    - "manhattan": Distancia de Manhattan (suma de diferencias absolutas)
    - "cosine": Distancia coseno (útil para datos normalizados)
    
    **Modo Debug**: Activa información detallada del proceso.
    - Útil para entender cómo funciona el algoritmo internamente
    - Muestra el progreso de la poda del árbol de expansión
    
    ### 📊 Visualizaciones Incluidas:
    - **Proyecciones 2D**: Vistas XY, XZ, YZ de los clusters
    - **Visualización 3D**: Vista interactiva en espacio 3D
    - **Análisis Estadístico**: Q-Q plots, efecto proporcional, boxplots
    - **Distribuciones**: Histogramas por cluster con estadísticas
    
    ¡Comienza configurando los parámetros y ejecutando el análisis!
    """)
    
    # Mostrar información del dataset seleccionado
    st.subheader("📁 Información del Dataset")
    
    data_file = st.session_state.get('data_file', 'bd_Parametros cineticos_hom.csv')
    variable_name = st.session_state.get('variable_name', None)
    
    info_text = f"""
    **Archivo**: `data/{data_file}`
    **Variable**: `{variable_name}` (seleccionada para clustering)
    
    **Coordenadas**:
    - `midx`, `midy`, `midz`: Coordenadas espaciales (utilizadas para la conectividad)
    
    💡 **Nota**: Puedes cambiar el archivo y la variable en el menú lateral antes de ejecutar el análisis.
    """
    st.info(info_text)
