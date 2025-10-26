"""
Clustering Espacial - Análisis de Datos Geológicos
Aplicación Streamlit para análisis interactivo con SKATER y K-Means
"""

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from shapely.geometry import Point
from libpysal.weights import KNN
from spopt.region import Skater
from sklearn.metrics import pairwise
from sklearn.cluster import KMeans
from scipy import stats
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Intentar importar skfuzzy, si no está disponible, implementaremos nuestra propia versión
try:
    import skfuzzy as fuzz
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False
    st.warning("⚠️ scikit-fuzzy no está instalado. Se usará una implementación básica de Fuzzy C-Means.")

warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Clustering Espacial",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🗺️ Análisis de Clustering Espacial")
st.markdown("**Análisis de clustering espacial para datos geológicos usando SKATER y K-Means**")

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
st.sidebar.markdown("### 🔧 Parámetros del Algoritmo")

# Selector de algoritmo PRIMERO
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Selección de Algoritmo")
algorithm = st.sidebar.radio(
    "**Algoritmo de Clustering**",
    options=["SKATER", "K-Means", "Fuzzy C-Means"],
    index=0,
    help="Selecciona el algoritmo de clustering a utilizar"
)
st.sidebar.markdown("---")

# Parámetros comunes
st.sidebar.markdown("### ⚙️ Parámetros Comunes")

# Número de clusters (común a ambos algoritmos)
n_clusters = st.sidebar.slider(
    "**Número de Clusters**",
    min_value=2,
    max_value=10,
    value=3,
    step=1,
    help="Número deseado de clusters para dividir los datos"
)

# Paleta de colores (común)
color_palette = st.sidebar.selectbox(
    "**Paleta de Colores**",
    options=["dark", "deep", "muted", "bright", "pastel", "colorblind"],
    index=0,
    help="Paleta de colores para visualizar los diferentes clusters"
)

# Parámetros específicos según el algoritmo
if algorithm == "SKATER":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Parámetros SKATER")
    
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
        value=150,
        step=10,
        help="Número mínimo de puntos que debe tener cada cluster"
    )
    
    # Parámetro Alpha (control de espacialidad)
    alpha = st.sidebar.slider(
        "**Alpha - Control de Espacialidad**",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Balance entre similitud espacial y de atributos"
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
    
    # Valores por defecto para otros algoritmos
    include_spatial = False
    normalize_features = True
    init_method = "k-means++"
    n_init = 10
    fuzziness = 2.0
    max_iter = 300
    error = 0.005

elif algorithm == "K-Means":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Parámetros K-Means")
    
    include_spatial = st.sidebar.checkbox(
        "**Incluir Coordenadas Espaciales**",
        value=False,
        help="Si está marcado, usar coordenadas X,Y,Z además del atributo"
    )
    
    normalize_features = st.sidebar.checkbox(
        "**Normalizar Variables**",
        value=True,
        help="Normalizar las variables para mismo peso en clustering"
    )
    
    init_method = st.sidebar.selectbox(
        "**Método de Inicialización**",
        options=["k-means++", "random"],
        index=0,
        help="Método para inicializar los centroides"
    )
    
    n_init = st.sidebar.slider(
        "**Inicializaciones**",
        min_value=1,
        max_value=20,
        value=10,
        help="Número de veces que se ejecutará con diferentes semillas"
    )
    
    # Valores por defecto para otros algoritmos
    k_vecinos = 90
    floor = 150
    alpha = 0.5
    islands = "ignore"
    dissimilarity_func = "euclidean"
    trace = False
    
else:  # Fuzzy C-Means
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Parámetros Fuzzy C-Means")
    
    fuzziness = st.sidebar.slider(
        "**Parámetro de Difusidad (m)**",
        min_value=1.1,
        max_value=3.0,
        value=2.0,
        step=0.1,
        help="Controla el nivel de difusidad de la pertenencia (1.1 = muy difuso, 3.0 = menos difuso)"
    )
    
    include_spatial = st.sidebar.checkbox(
        "**Incluir Coordenadas Espaciales**",
        value=False,
        help="Si está marcado, usar coordenadas X,Y,Z además del atributo"
    )
    
    normalize_features = st.sidebar.checkbox(
        "**Normalizar Variables**",
        value=True,
        help="Normalizar las variables para mismo peso en clustering"
    )
    
    max_iter = st.sidebar.slider(
        "**Iteraciones Máximas**",
        min_value=50,
        max_value=500,
        value=300,
        step=50,
        help="Número máximo de iteraciones para convergencia"
    )
    
    error = st.sidebar.slider(
        "**Tolerancia de Convergencia**",
        min_value=0.0001,
        max_value=0.1,
        value=0.005,
        step=0.0001,
        format="%.4f",
        help="Condición de parada: diferencia mínima entre iteraciones"
    )
    
    # Valores por defecto para otros algoritmos
    k_vecinos = 90
    floor = 150
    alpha = 0.5
    islands = "ignore"
    dissimilarity_func = "euclidean"
    trace = False
    init_method = "k-means++"
    n_init = 10

# Botón para ejecutar análisis
st.sidebar.markdown("---")
if st.sidebar.button(f"🚀 Ejecutar Análisis {algorithm}", type="primary"):
    st.session_state.run_analysis = True
    
    # Determinar algoritmo y guardar parámetros
    if algorithm == "SKATER":
        st.session_state.algorithm = "SKATER"
        st.session_state.k_vecinos = k_vecinos
        st.session_state.floor = floor
        st.session_state.alpha = alpha
        st.session_state.islands = islands
        st.session_state.dissimilarity_func = dissimilarity_func
        st.session_state.trace = trace
    elif algorithm == "K-Means":
        st.session_state.algorithm = "KMeans"
        st.session_state.include_spatial = include_spatial
        st.session_state.normalize_features = normalize_features
        st.session_state.init_method = init_method
        st.session_state.n_init = n_init
    else:  # Fuzzy C-Means
        st.session_state.algorithm = "FuzzyCMeans"
        st.session_state.include_spatial = include_spatial
        st.session_state.normalize_features = normalize_features
        st.session_state.fuzziness = fuzziness
        st.session_state.max_iter = max_iter
        st.session_state.error = error

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

# Función para ejecutar SKATER
def run_skater(df, k_vecinos, n_clusters, floor, alpha, islands, dissimilarity_func, trace):
    """Ejecutar algoritmo SKATER con parámetros avanzados"""
    try:
        # Crear geometría de puntos
        geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        
        # Crear matriz de conectividad espacial
        w = KNN.from_dataframe(gdf, k=k_vecinos)
        
        # Seleccionar función de disimilitud
        if dissimilarity_func == "euclidean":
            dissimilarity = pairwise.euclidean_distances
        elif dissimilarity_func == "manhattan":
            dissimilarity = pairwise.manhattan_distances
        elif dissimilarity_func == "cosine":
            dissimilarity = pairwise.cosine_distances
        else:
            dissimilarity = pairwise.euclidean_distances
        
        # Función personalizada que implementa el parámetro alpha correctamente
        def custom_dissimilarity(X, Y=None):
            """Función de disimilitud que combina espacialidad y atributos según alpha"""
            
            # Si alpha es 0, usar solo disimilitud de atributos
            if alpha == 0.0:
                if Y is not None:
                    return dissimilarity(X, Y)
                else:
                    return dissimilarity(X)
            
            # Si alpha es 1, usar solo disimilitud espacial
            elif alpha == 1.0:
                # Usar coordenadas espaciales para calcular distancias
                spatial_data = gdf[['x', 'y']].values
                if Y is not None:
                    return pairwise.euclidean_distances(spatial_data, spatial_data)
                else:
                    return pairwise.euclidean_distances(spatial_data)
            
            # Caso intermedio: combinar ambas disimilitudes
            else:
                # Disimilitud de atributos
                if Y is not None:
                    attr_dist = dissimilarity(X, Y)
                else:
                    attr_dist = dissimilarity(X)
                
                # Disimilitud espacial
                spatial_data = gdf[['x', 'y']].values
                if Y is not None:
                    spatial_dist = pairwise.euclidean_distances(spatial_data, spatial_data)
                else:
                    spatial_dist = pairwise.euclidean_distances(spatial_data)
                
                # Asegurar que ambas matrices tengan las mismas dimensiones
                if attr_dist.shape != spatial_dist.shape:
                    # Si no coinciden, usar solo atributos para evitar errores
                    return attr_dist
                
                # Normalizar ambas matrices para que estén en la misma escala
                attr_max = np.max(attr_dist)
                spatial_max = np.max(spatial_dist)
                
                if attr_max > 0:
                    attr_dist_norm = attr_dist / attr_max
                else:
                    attr_dist_norm = attr_dist
                    
                if spatial_max > 0:
                    spatial_dist_norm = spatial_dist / spatial_max
                else:
                    spatial_dist_norm = spatial_dist
                
                # Combinar con peso alpha
                combined_dist = (1 - alpha) * attr_dist_norm + alpha * spatial_dist_norm
                
                return combined_dist
        
        # Configurar SKATER con parámetros avanzados
        spanning_forest_kwds = {
            'dissimilarity': custom_dissimilarity,
            'affinity': None,
            'reduction': np.sum,
            'center': np.mean,
            'verbose': 1 if trace else 0
        }
        
        # Crear y resolver modelo con parámetros adicionales
        model = Skater(
            gdf,
            w,
            ['variable'],
            n_clusters=n_clusters,
            floor=floor,
            trace=trace,
            islands=islands,
            spanning_forest_kwds=spanning_forest_kwds
        )
        
        model.solve()
        
        # Agregar clusters al dataframe
        df['cluster'] = model.labels_
        
        return df, model
        
    except Exception as e:
        st.error(f"Error en SKATER: {e}")
        return None, None

# Función para ejecutar K-Means
def run_kmeans(df, n_clusters, include_spatial, normalize_features, init_method, n_init):
    """Ejecutar algoritmo K-Means
    
    Args:
        df: DataFrame con columnas x, y, z, variable
        n_clusters: Número de clusters deseados
        include_spatial: Si True, incluye coordenadas espaciales en el clustering
        normalize_features: Si True, normaliza las características
        init_method: Método de inicialización ('k-means++' o 'random')
        n_init: Número de inicializaciones
        
    Returns:
        DataFrame con columna 'cluster' agregada
    """
    try:
        from sklearn.preprocessing import StandardScaler
        
        # Preparar características para clustering
        if include_spatial:
            # Incluir coordenadas espaciales y variable
            features = df[['x', 'y', 'z', 'variable']].values
        else:
            # Solo usar la variable
            features = df[['variable']].values
        
        # Normalizar si es necesario
        if normalize_features:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        
        # Aplicar K-Means
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init_method,
            n_init=n_init,
            random_state=42
        )
        
        clusters = kmeans.fit_predict(features)
        
        # Agregar clusters al dataframe
        df['cluster'] = clusters
        
        # Calcular inercia (sum of squared distances to centroids)
        inertia = kmeans.inertia_
        
        # Guardar el modelo
        model = {'kmeans': kmeans, 'scaler': scaler if normalize_features else None, 'inertia': inertia}
        
        return df, model
        
    except Exception as e:
        st.error(f"Error en K-Means: {e}")
        return None, None

# Función para ejecutar Fuzzy C-Means
def run_fuzzy_cmeans(df, n_clusters, fuzziness, include_spatial, normalize_features, max_iter, error):
    """Ejecutar algoritmo Fuzzy C-Means
    
    Args:
        df: DataFrame con columnas x, y, z, variable
        n_clusters: Número de clusters deseados
        fuzziness: Parámetro de difusidad (m)
        include_spatial: Si True, incluye coordenadas espaciales en el clustering
        normalize_features: Si True, normaliza las características
        max_iter: Número máximo de iteraciones
        error: Tolerancia de convergencia
        
    Returns:
        DataFrame con columna 'cluster' agregada y matriz de pertenencia
    """
    try:
        from sklearn.preprocessing import StandardScaler
        
        # Preparar características para clustering
        if include_spatial:
            features = df[['x', 'y', 'z', 'variable']].values
        else:
            features = df[['variable']].values
        
        # Normalizar si es necesario
        scaler = None
        if normalize_features:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        
        if SKFUZZY_AVAILABLE:
            # Usar skfuzzy si está disponible
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                features.T,
                n_clusters,
                fuzziness,
                error=error,
                maxiter=max_iter,
                init=None
            )
            
            # Asignar cluster al punto más probable
            cluster_assignments = np.argmax(u, axis=0)
            df['cluster'] = cluster_assignments
            
            # Guardar información adicional
            model = {
                'centers': cntr,
                'membership': u,
                'fpc': fpc,
                'scaler': scaler,
                'n_iterations': p,
                'inertia': jm[-1] if len(jm) > 0 else None
            }
        else:
            # Implementación básica de Fuzzy C-Means
            n_samples, n_features = features.shape
            
            # Inicializar matriz de pertenencia aleatoriamente
            np.random.seed(42)
            membership = np.random.random((n_clusters, n_samples))
            membership = membership / membership.sum(axis=0)
            
            # Calcular centroides iniciales
            centers = np.dot(membership ** fuzziness, features)
            centers = centers / (membership ** fuzziness).sum(axis=1)[:, np.newaxis]
            
            # Iterar hasta convergencia
            for iteration in range(max_iter):
                # Calcular distancias
                distances = np.zeros((n_clusters, n_samples))
                for i in range(n_clusters):
                    distances[i] = np.linalg.norm(features - centers[i], axis=1) ** 2
                
                # Actualizar matriz de pertenencia
                new_membership = np.zeros((n_clusters, n_samples))
                for j in range(n_samples):
                    for i in range(n_clusters):
                        sum_term = 0
                        for k in range(n_clusters):
                            sum_term += (distances[i, j] / distances[k, j]) ** (2 / (fuzziness - 1))
                        new_membership[i, j] = 1.0 / sum_term if distances[i, j] > 0 else 1.0
                
                # Actualizar centros
                centers = np.dot(new_membership ** fuzziness, features)
                centers = centers / (new_membership ** fuzziness).sum(axis=1)[:, np.newaxis]
                
                # Verificar convergencia
                diff = np.abs(new_membership - membership).max()
                membership = new_membership
                
                if diff < error:
                    break
            
            # Asignar cluster al punto más probable
            cluster_assignments = np.argmax(membership, axis=0)
            df['cluster'] = cluster_assignments
            
            # Guardar información adicional
            model = {
                'centers': centers,
                'membership': membership,
                'fpc': None,  # Fuzzy partition coefficient
                'scaler': scaler,
                'n_iterations': iteration + 1,
                'inertia': distances.max()
            }
        
        return df, model
        
    except Exception as e:
        st.error(f"Error en Fuzzy C-Means: {e}")
        return None, None

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
            x=cluster_data['x']/1000,
            y=cluster_data['y']/1000,
            z=cluster_data['z']/1000,
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
        # Determinar algoritmo a usar
        current_algorithm = st.session_state.get('algorithm', 'SKATER')
        
        # Inicializar variables para todos los algoritmos
        k_vec, fl, alp, isl, dissim, tr = None, None, None, None, None, None
        include_spatial, normalize_features, init_method, n_init = None, None, None, None
        fuzziness, max_iter, error = None, None, None
        
        if current_algorithm == 'SKATER':
            status_text.text("Ejecutando SKATER...")
            progress_bar.progress(60)
            
            # Obtener parámetros de SKATER desde session_state
            k_vec = st.session_state.get('k_vecinos', 90)
            fl = st.session_state.get('floor', 150)
            alp = st.session_state.get('alpha', 0.5)
            isl = st.session_state.get('islands', 'ignore')
            dissim = st.session_state.get('dissimilarity_func', 'euclidean')
            tr = st.session_state.get('trace', False)
            
            # Ejecutar SKATER
            df_result, model = run_skater(df, k_vec, n_clusters, fl, alp, isl, dissim, tr)
            algorithm_name = "SKATER"
            
        elif current_algorithm == 'KMeans':
            status_text.text("Ejecutando K-Means...")
            progress_bar.progress(60)
            
            # Obtener parámetros de K-Means
            include_spatial = st.session_state.get('include_spatial', False)
            normalize_features = st.session_state.get('normalize_features', True)
            init_method = st.session_state.get('init_method', 'k-means++')
            n_init = st.session_state.get('n_init', 10)
            
            # Ejecutar K-Means
            df_result, model = run_kmeans(df, n_clusters, include_spatial, normalize_features, init_method, n_init)
            algorithm_name = "K-Means"
            
        elif current_algorithm == 'FuzzyCMeans':
            status_text.text("Ejecutando Fuzzy C-Means...")
            progress_bar.progress(60)
            
            # Obtener parámetros de Fuzzy C-Means
            include_spatial = st.session_state.get('include_spatial', False)
            normalize_features = st.session_state.get('normalize_features', True)
            fuzziness = st.session_state.get('fuzziness', 2.0)
            max_iter = st.session_state.get('max_iter', 300)
            error = st.session_state.get('error', 0.005)
            
            # Ejecutar Fuzzy C-Means
            df_result, model = run_fuzzy_cmeans(df, n_clusters, fuzziness, include_spatial, normalize_features, max_iter, error)
            algorithm_name = "Fuzzy C-Means"
            
        else:
            st.error(f"Algoritmo desconocido: {current_algorithm}")
            df_result, model = None, None
            algorithm_name = "Desconocido"
        
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
            
            if algorithm_name == "SKATER":
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("K Vecinos", k_vec)
                    st.metric("Alpha", f"{alp:.1f}")
                with col2:
                    st.metric("Clusters Deseados", n_clusters)
                    st.metric("Tamaño Mínimo", fl)
                with col3:
                    st.metric("Función Disimilitud", dissim.title())
                    st.metric("Manejo Islas", isl.title())
                with col4:
                    st.metric("Modo Debug", "Sí" if tr else "No")
                    st.metric("Paleta Colores", color_palette.title())
            elif algorithm_name == "K-Means":
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Clusters", n_clusters)
                    st.metric("Inercia", f"{model['inertia']:.2f}")
                with col2:
                    st.metric("Espacial", "Sí" if include_spatial else "No")
                    st.metric("Normalizado", "Sí" if normalize_features else "No")
                with col3:
                    st.metric("Inicialización", init_method)
                    st.metric("N Inicializaciones", n_init)
                with col4:
                    st.metric("Algoritmo", algorithm_name)
                    st.metric("Paleta Colores", color_palette.title())
                    
            else:  # Fuzzy C-Means
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Clusters", n_clusters)
                    st.metric("Difusidad (m)", f"{fuzziness:.2f}")
                with col2:
                    st.metric("Espacial", "Sí" if include_spatial else "No")
                    st.metric("Normalizado", "Sí" if normalize_features else "No")
                with col3:
                    st.metric("Iteraciones", model.get('n_iterations', 'N/A'))
                    st.metric("Tolerancia", f"{error:.4f}")
                with col4:
                    st.metric("Algoritmo", algorithm_name)
                    st.metric("FPC", f"{model.get('fpc', 'N/A'):.3f}" if model.get('fpc') is not None else "N/A")
            
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
    
    Esta aplicación permite realizar análisis de clustering espacial usando tres algoritmos:
    
    - **SKATER** (Spatial 'K'luster Analysis by Tree Edge Removal) - Clustering espacial basado en conectividad
    - **K-Means** - Clustering basado en similitud de atributos, con opción de incluir coordenadas
    - **Fuzzy C-Means** - Clustering difuso que permite pertenencia parcial a múltiples clusters
    
    ### 📋 Instrucciones:
    1. **Selecciona el archivo y variable** en la barra lateral
    2. **Elige el algoritmo** (SKATER, K-Means o Fuzzy C-Means)
    3. **Configura los parámetros** según el algoritmo seleccionado
    4. **Haz clic en "Ejecutar Análisis"** para procesar los datos
    5. **Explora los resultados** en las diferentes secciones
    
    ### 🔧 Parámetros Explicados:
    
    **K Vecinos**: Define cuántos vecinos más cercanos se consideran para crear la matriz de conectividad espacial. 
    - Valores bajos (5-20): Conexiones más locales, clusters más pequeños y compactos
    - Valores altos (100-200): Conexiones más amplias, clusters más grandes y dispersos
    
    **Número de Clusters**: Cantidad deseada de grupos espaciales.
    - El algoritmo intentará crear esta cantidad de clusters
    - Puede resultar en menos clusters si el tamaño mínimo no se cumple
    
    **Tamaño Mínimo por Cluster**: Número mínimo de puntos por cluster.
    - Clusters más pequeños serán fusionados con otros
    - Ayuda a evitar clusters muy pequeños o ruidosos
    
    **Alpha - Control de Espacialidad**: Balance entre similitud espacial y de atributos.
    - 0.0: Solo considera similitud de atributos (ignora posición espacial)
    - 0.5: Balance equilibrado entre espacialidad y atributos
    - 1.0: Solo considera proximidad espacial (ignora valores de atributos)
    
    **Manejo de Islas**: Cómo tratar puntos aislados.
    - "ignore": Ignora puntos que no pueden conectarse
    - "increase": Aumenta el número de clusters para incluir puntos aislados
    
    **Función de Disimilitud**: Métrica para calcular distancias.
    - "euclidean": Distancia euclidiana estándar
    - "manhattan": Distancia de Manhattan (suma de diferencias absolutas)
    - "cosine": Distancia coseno (útil para datos normalizados)
    
    **Modo Debug**: Activa información detallada del proceso.
    - Útil para entender cómo funciona el algoritmo internamente
    
    ### 🔧 Parámetros K-Means:
    
    **Incluir Coordenadas Espaciales**: Si está marcado, el clustering considera las coordenadas X, Y, Z además del valor del atributo.
    - Marcado: Crea clusters espacialmente coherentes
    - Desmarcado: Clustering basado solo en valores del atributo
    
    **Normalizar Variables**: Normaliza las características para que tengan el mismo peso.
    - Importante cuando se incluyen coordenadas, ya que estas pueden tener diferentes escalas
    
    **Método de Inicialización**: 
    - "k-means++": Inicialización inteligente (recomendado)
    - "random": Inicialización aleatoria
    
    **Número de Inicializaciones**: Cuántas veces se ejecutará el algoritmo con diferentes semillas.
    - Mayor número = mejor resultado, pero más tiempo de ejecución
    
    ### 🔧 Parámetros Fuzzy C-Means:
    
    **Parámetro de Difusidad (m)**: Controla el grado de difusidad en la pertenencia a clusters.
    - Valores bajos (1.1-1.5): Muy difuso, alta incertidumbre en las fronteras
    - Valores medios (2.0): Balance entre difusidad y precisión
    - Valores altos (2.5-3.0): Menos difuso, clusters más definidos
    
    **Incluir Coordenadas Espaciales**: Si está marcado, el clustering considera las coordenadas X, Y, Z además del valor del atributo.
    
    **Iteraciones Máximas**: Número máximo de iteraciones antes de detener el algoritmo.
    
    **Tolerancia de Convergencia**: Criterio de parada cuando el cambio entre iteraciones es menor que este valor.
    
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
