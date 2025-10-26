#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Lectura del CSV con separador correcto
df = pd.read_csv('../data/bd_Parametros cineticos_hom.csv',
             sep=';',
             encoding='latin-1')

# %%
list_columns = ["midx", "midy", "midz", "Starkey_min", "BWI_kwh_tc", "BWI_kwh_tm",
]
df[list_columns].hist()
#%%