# Análisis de Clustering Espacial con SKATER

Aplicación Streamlit para análisis de clustering espacial utilizando el algoritmo SKATER (Spatial 'K'luster Analysis by Tree Edge Removal).

## Estructura del Proyecto

```
├── streamlit_app.py       # Aplicación principal Streamlit (UI)
├── skater_algorithm.py     # Módulo con implementación básica de SKATER
├── data/                   # Datos de ejemplo
│   ├── bd_Parametros cineticos_hom.csv
│   └── 1_recpeso.xlsx
└── requirements.txt        # Dependencias del proyecto
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

Ejecutar la aplicación:

```bash
streamlit run streamlit_app.py
```

## Características

- **Clustering Espacial**: Uso exclusivo del algoritmo SKATER
- **Parámetros Configurables**:
  - K vecinos más cercanos
  - Número de clusters deseado
  - Tamaño mínimo por cluster
  - Manejo de islas
  - Función de disimilitud
  - Modo debug

- **Visualizaciones**:
  - Proyecciones 2D (XY, XZ, YZ)
  - Visualización 3D interactiva
  - Análisis estadístico (Q-Q plots, efecto proporcional, boxplots)
  - Histogramas por cluster

## Algoritmo SKATER

SKATER es un algoritmo de regionalización espacial basado en la poda de árboles de expansión mínima. El algoritmo:

1. Crea una matriz de conectividad espacial (K vecinos más cercanos)
2. Construye un árbol de expansión mínima
3. Particiona el árbol en regiones contiguas
4. Maximiza la homogeneidad interna de cada región

### Documentación

- https://pysal.org/spopt/notebooks/skater.html

## Módulos

### `skater_algorithm.py`

Módulo con la implementación básica de SKATER siguiendo la documentación oficial.

### `streamlit_app.py`

Aplicación Streamlit que proporciona:
- Carga de datos
- Configuración de parámetros
- Ejecución del algoritmo
- Visualización de resultados

