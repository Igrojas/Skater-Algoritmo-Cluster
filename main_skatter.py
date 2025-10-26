"""
SKATER - ANÁLISIS DE CLUSTERING ESPACIAL PARA DATOS GEOLÓGICOS
================================================================
Código simple para cargar Excel y aplicar SKATER
"""
#%%
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import seaborn as sns
from shapely.geometry import Point
from libpysal.weights import KNN
from spopt.region import Skater
from sklearn.metrics import pairwise
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SKATER - CLUSTERING ESPACIAL GEOLÓGICO")
print("="*70)

# ==============================================================================
# CONFIGURACIÓN - MODIFICA AQUÍ TUS VARIABLES
# ==============================================================================

# 1. RUTA DEL ARCHIVO EXCEL
archivo_excel = 'data/1_recpeso.xlsx'  # Cambia por la ruta de tu archivo
hoja = 'recpeso'  # Nombre de la hoja (o 0 para primera hoja)

# 2. NOMBRES DE LAS COLUMNAS EN TU EXCEL
columna_x = 'midx'           # ← Nombre de tu columna X
columna_y = 'midy'           # ← Nombre de tu columna Y  
columna_z = 'midz'           # ← Nombre de tu columna Z
columna_variable = 'recpe'  # ← Nombre de tu variable a clusterizar

# 3. PARÁMETROS DE SKATER
k_vecinos = 90      # Número de vecinos para conectividad espacial
n_clusters = 3     # Número de clusters deseado
floor = 150          # Tamaño mínimo de puntos por cluster

print("\nCONFIGURACIÓN:")
print(f"  Archivo: {archivo_excel}")
print(f"  Hoja: {hoja}")
print(f"  Columnas: X={columna_x}, Y={columna_y}, Z={columna_z}")
print(f"  Variable: {columna_variable}")
print(f"  Parámetros: k_vecinos={k_vecinos}, n_clusters={n_clusters}, floor={floor}")

# ==============================================================================
# CARGAR DATOS DESDE EXCEL
# ==============================================================================

print("\n" + "="*70)
print("CARGANDO DATOS...")
print("="*70)

# Cargar el Excel
df = pd.read_excel(archivo_excel, sheet_name=hoja)

print(f"\n✓ Archivo cargado: {len(df)} filas")
print(f"  Columnas disponibles: {df.columns.tolist()}")

# df = df[df['midy'] < 25500]
# Verificar que las columnas existen
columnas_necesarias = [columna_x, columna_y, columna_z, columna_variable]
columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]

if columnas_faltantes:
    print(f"\n❌ ERROR: Columnas no encontradas: {columnas_faltantes}")
    print(f"   Columnas disponibles en el Excel: {df.columns.tolist()}")
    exit()

# Seleccionar solo las columnas necesarias y renombrar
df = df[[columna_x, columna_y, columna_z, columna_variable]].copy()
df.columns = ['x', 'y', 'z', 'variable']

# Eliminar filas con valores nulos
df = df.dropna()

print(f"\n✓ Datos procesados: {len(df)} puntos válidos")
print(f"\nEstadísticas de la variable '{columna_variable}':")
print(df['variable'].describe())

# ==============================================================================
# PREPARAR DATOS ESPACIALES
# ==============================================================================

print("\n" + "="*70)
print("PREPARANDO ANÁLISIS ESPACIAL...")
print("="*70)

# Crear geometría de puntos
geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Crear matriz de conectividad espacial
w = KNN.from_dataframe(gdf, k=k_vecinos)

print(f"\n✓ Matriz de conectividad creada (K={k_vecinos} vecinos)")

# ==============================================================================
# EJECUTAR SKATER
# ==============================================================================

print("\n" + "="*70)
print("EJECUTANDO SKATER...")
print("="*70)

# Configurar SKATER
spanning_forest_kwds = {
    'dissimilarity': pairwise.euclidean_distances,
    'affinity': None,
    'reduction': np.sum,
    'center': np.mean,
    'verbose': 1
}

# Crear y resolver modelo
model = Skater(
    gdf,
    w,
    ['variable'],
    n_clusters=n_clusters,
    floor=floor,
    trace=False,
    # islands='increase',
    spanning_forest_kwds=spanning_forest_kwds
)

model.solve()

# Agregar clusters al dataframe
gdf['cluster'] = model.labels_
df['cluster'] = model.labels_

print("\n✓ SKATER completado")
print(f"\nDistribución de puntos por cluster:")
print(df['cluster'].value_counts().sort_index())

# ==============================================================================
# CALCULAR ESTADÍSTICAS
# ==============================================================================

stats_cluster = df.groupby('cluster')['variable'].agg([
    'count', 'mean', 'std', 'min', 'max'
])
stats_cluster['cv'] = (stats_cluster['std'] / stats_cluster['mean']) * 100

print(f"\nEstadísticas por cluster:")
print(stats_cluster.round(3))
#%%
# ==============================================================================
# VISUALIZACIONES
# ==============================================================================

# Ahora puedes modificar esta variable para cambiar la paleta de colores fácilmente
color_palette_name = "dark"  # Usa "deep" la paleta por defecto de seaborn, o cambia por cualquier nombre válido

# Obtener los clusters realmente presentes en los datos
clusters = sorted(df['cluster'].unique())
actual_n_clusters = len(clusters)

# Generar paleta de colores adaptada al número real de clusters
cluster_palette = sns.color_palette(color_palette_name, n_colors=actual_n_clusters)

# Crear un diccionario para mapear cada cluster a su color
cluster_colors = {cluster_id: cluster_palette[i] for i, cluster_id in enumerate(clusters)}

# Colores para usar en las gráficas
colors = cluster_palette

# ----- GRÁFICAS 2D DE PROYECCIONES XY, XZ, YZ ------
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Proyección XY
for i, cluster_id in enumerate(clusters):
    cluster_data = df[df['cluster'] == cluster_id]
    axs[0].scatter(
        cluster_data['x']/1000, 
        cluster_data['y']/1000,
        color=cluster_colors[cluster_id],
        label=f'Cluster {cluster_id}',
        s=30,
        alpha=0.85
    )
axs[0].set_xlabel('X (km)')
axs[0].set_ylabel('Y (km)')
axs[0].set_title('Proyección XY')
axs[0].grid(True, linestyle='--', alpha=0.7)

# Proyección XZ
for i, cluster_id in enumerate(clusters):
    cluster_data = df[df['cluster'] == cluster_id]
    axs[1].scatter(
        cluster_data['x']/1000, 
        cluster_data['z']/1000,
        color=cluster_colors[cluster_id],
        label=f'Cluster {cluster_id}',
        s=30,
        alpha=0.85
    )
axs[1].set_xlabel('X (km)')
axs[1].set_ylabel('Z (km)')
axs[1].set_title('Proyección XZ')
axs[1].grid(True, linestyle='--', alpha=0.7)

# Proyección YZ
for i, cluster_id in enumerate(clusters):
    cluster_data = df[df['cluster'] == cluster_id]
    axs[2].scatter(
        cluster_data['y']/1000, 
        cluster_data['z']/1000,
        color=cluster_colors[cluster_id],
        label=f'Cluster {cluster_id}',
        s=30,
        alpha=0.85
    )
axs[2].set_xlabel('Y (km)')
axs[2].set_ylabel('Z (km)')
axs[2].set_title('Proyección YZ')
axs[2].grid(True, linestyle='--', alpha=0.7)

# Leyenda común para los tres gráficos
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=min(actual_n_clusters, 8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# ----- GRÁFICA 3D DE PUNTOS ------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar cada cluster por separado para usar colores específicos
for i, cluster_id in enumerate(clusters):
    cluster_data = df[df['cluster'] == cluster_id]
    ax.scatter(
        cluster_data['x']/1000, 
        cluster_data['y']/1000, 
        cluster_data['z']/1000,
        color=cluster_colors[cluster_id],
        label=f'Cluster {cluster_id}',
        s=30,
        alpha=0.85
    )

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Clusters SKATER: Espacio 3D')
ax.legend()
plt.tight_layout()
plt.show()

# ----- GRÁFICAS DE ANÁLISIS ESTADÍSTICO: QQ-PLOT, EFECTO PROPORCIONAL Y BOXPLOT ------
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 1. Q-Q Plot lognormal para cada cluster
for i, cluster_id in enumerate(clusters):
    # Obtener los datos del cluster actual
    cluster_data = df[df['cluster'] == cluster_id]['variable'].values
    
    # Eliminar NaN e infinitos
    cluster_data = cluster_data[np.isfinite(cluster_data)]
    cluster_data = cluster_data[cluster_data > 0]  # Lognormal requiere valores positivos
    
    if len(cluster_data) > 0:
        # Ordenar datos
        data_sorted = np.sort(cluster_data)
        n = len(data_sorted)
        
        # Calcular percentiles empíricos (estilo Vulcan)
        percentiles = (np.arange(1, n+1) - 0.5) / n * 100
        
        # Graficar puntos: X = valores originales, Y = percentiles
        axs[0].scatter(data_sorted, percentiles, color=colors[i], 
                   label=f'Cluster {cluster_id}', 
                   alpha=0.5, edgecolors='k', s=40)
        
        # Línea de referencia lognormal para este cluster
        # Ajustar parámetros lognormales
        mu = np.mean(np.log(data_sorted))
        sigma = np.std(np.log(data_sorted))
        
        # Generar línea de referencia
        x_ref = np.linspace(data_sorted.min(), data_sorted.max(), 100)
        y_ref = stats.norm.cdf(np.log(x_ref), mu, sigma) * 100
        
        axs[0].plot(x_ref, y_ref, color=colors[i], linestyle='-', 
                linewidth=2, alpha=0.8)

# Configurar el gráfico Q-Q Plot
axs[0].set_xscale('log')  # CRÍTICO: escala logarítmica en X
axs[0].set_xlabel('Variable (escala logarítmica)', fontsize=10)
axs[0].set_ylabel('Percentiles Lognormales', fontsize=10)
axs[0].set_title('Q-Q Plot Lognormal por Cluster', fontsize=12, fontweight='bold')
axs[0].set_ylim([0, 100])  # Percentiles de 0 a 100
axs[0].grid(True, which='both', linestyle='--', alpha=0.3)

# 2. Gráfico de Efecto Proporcional por Cluster
for i, cluster_id in enumerate(clusters):
    # Obtener los datos del cluster actual
    cluster_data = df[df['cluster'] == cluster_id]['variable']
    
    if len(cluster_data) > 0:
        # Calcular la media y desviación estándar
        mean = cluster_data.mean()
        std = cluster_data.std()
        
        # Graficar el punto
        axs[1].scatter(mean, std, color=colors[i], 
                  label=f'Cluster {cluster_id}', 
                  s=80, alpha=0.8, edgecolor='k')
        
        # Añadir etiqueta con el número de puntos
        axs[1].annotate(f'n={len(cluster_data)}', 
                   (mean, std), 
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=9)

# Calcular la línea de tendencia para todos los datos
all_means = [df[df['cluster'] == c]['variable'].mean() for c in clusters]
all_stds = [df[df['cluster'] == c]['variable'].std() for c in clusters]

# Añadir línea de tendencia si hay suficientes puntos
if len(all_means) > 1:
    # Ajustar una línea de regresión
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_means, all_stds)
    
    # Crear puntos x para la línea
    x_line = np.linspace(min(all_means) * 0.9, max(all_means) * 1.1, 100)
    y_line = slope * x_line + intercept
    
    # Graficar la línea de tendencia
    axs[1].plot(x_line, y_line, 'k--', alpha=0.7, 
            label=f'y = {slope:.2f}x + {intercept:.2f} (R² = {r_value**2:.2f})')

axs[1].set_xlabel('Media', fontsize=10)
axs[1].set_ylabel('Desviación Estándar', fontsize=10)
axs[1].set_title('Efecto Proporcional por Cluster', fontsize=12, fontweight='bold')
axs[1].grid(True, linestyle='--', alpha=0.3)

# 3. Boxplot para cada cluster
boxplot_data = [df[df['cluster'] == c]['variable'] for c in clusters]
boxplot = axs[2].boxplot(boxplot_data, patch_artist=True)

# Personalizar colores de los boxplots
for i, box in enumerate(boxplot['boxes']):
    box.set_facecolor(colors[i])
    box.set_alpha(0.7)
    box.set_edgecolor('black')

axs[2].set_xlabel('Cluster', fontsize=10)
axs[2].set_ylabel('Variable', fontsize=10)
axs[2].set_title('Boxplot por Cluster', fontsize=12, fontweight='bold')
axs[2].set_xticklabels([f'{c}' for c in clusters])
axs[2].grid(True, axis='y', linestyle='--', alpha=0.3)

# Leyenda común para los tres gráficos
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=min(actual_n_clusters, 8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# ==============================================================================
# GRÁFICO DE EFECTO PROPORCIONAL POR CLUSTER
# ==============================================================================
print("\n" + "="*70)
print("GENERANDO HISTOGRAMAS POR CLUSTER")
print("="*70)

# Crear figura para los histogramas
fig, ax = plt.subplots(figsize=(14, 10))

# Determinar el rango global para los histogramas
min_val = df['variable'].min()
max_val = df['variable'].max()
bins = np.linspace(min_val, max_val, 30)

# Para cada cluster, graficar su histograma
for i, cluster_id in enumerate(clusters):
    # Obtener los datos del cluster actual
    cluster_data = df[df['cluster'] == cluster_id]['variable']
    
    if len(cluster_data) > 0:
        # Calcular estadísticas
        mean = cluster_data.mean()
        std = cluster_data.std()
        
        # Graficar el histograma con transparencia
        ax.hist(cluster_data, bins=bins, alpha=0.6, color=colors[i], 
                label=f'Cluster {cluster_id} (n={len(cluster_data)}, μ={mean:.2f}, σ={std:.2f})')
        
        # Añadir línea vertical para la media
        ax.axvline(mean, color=colors[i], linestyle='--', linewidth=2)

# Configurar el gráfico
ax.set_title('Distribución de Valores por Cluster', fontsize=16, fontweight='bold')
ax.set_xlabel('Valor de la Variable', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.3)

# Mejorar la leyenda
ax.legend(title='Clusters', title_fontsize=12, fontsize=10, 
          loc='upper right', bbox_to_anchor=(1.15, 1), 
          frameon=True, facecolor='white', edgecolor='gray')

# Añadir texto explicativo
plt.figtext(0.5, 0.01, 
            "Las líneas discontinuas representan la media de cada cluster. " +
            "La distribución muestra la variabilidad dentro de cada grupo.",
            ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.show()

# %%
#