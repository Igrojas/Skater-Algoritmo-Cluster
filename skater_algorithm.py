"""
Algoritmo SKATER - Implementación básica
SKATER: Spatial 'K'luster Analysis by Tree Edge Removal
Documentación: https://pysal.org/spopt/notebooks/skater.html
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from libpysal.weights import KNN
from spopt.region import Skater
from sklearn.metrics import pairwise


def run_skater(
    df: pd.DataFrame,
    attrs_name: list,
    n_clusters: int,
    k_vecinos: int = 10,
    floor: int = 3,
    islands: str = "ignore",
    dissimilarity_func: str = "euclidean",
    trace: bool = False
) -> tuple[pd.DataFrame, Skater]:
    """
    Ejecutar algoritmo SKATER básico.
    
    Args:
        df: DataFrame con columnas 'x', 'y', 'z' y la variable de clustering
        attrs_name: Lista de nombres de atributos para el clustering
        n_clusters: Número deseado de clusters
        k_vecinos: Número de vecinos más cercanos para conectividad espacial
        floor: Número mínimo de unidades espaciales por región
        islands: Cómo manejar puntos aislados ('ignore' o 'increase')
        dissimilarity_func: Función de disimilitud ('euclidean', 'manhattan', 'cosine')
        trace: Activar información de debugging
        
    Returns:
        DataFrame con columna 'cluster' agregada y modelo SKATER
        
    Raises:
        Exception: Si ocurre error en la ejecución
    """
    try:
        # Crear geometría de puntos (usando x, y como coordenadas espaciales)
        geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        
        # Crear matriz de conectividad espacial (K vecinos más cercanos)
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
        
        # Configurar parámetros de spanning forest
        spanning_forest_kwds = {
            'dissimilarity': dissimilarity,
            'affinity': None,
            'reduction': np.sum,
            'center': np.mean,
        }
        
        # Crear modelo SKATER
        model = Skater(
            gdf,
            w,
            attrs_name,
            n_clusters=n_clusters,
            floor=floor,
            trace=trace,
            islands=islands,
            spanning_forest_kwds=spanning_forest_kwds
        )
        
        # Resolver el modelo
        model.solve()
        
        # Agregar clusters al dataframe
        df['cluster'] = model.labels_
        
        return df, model
        
    except Exception as e:
        raise Exception(f"Error en SKATER: {e}")

