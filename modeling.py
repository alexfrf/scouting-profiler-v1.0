# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 21:24:59 2023

@author: aleex
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import modeling_functions as mf
from pandas.api.types import is_numeric_dtype
import warnings
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings('ignore')
temporada='2021-2022'

seas = temporada[:4]
ruta_base = os.path.join('',Path(os.getcwd()))
ruta_datos = os.path.join(ruta_base,'Datos/{}'.format(seas))

#import Tmarkt_scraper



#for i in [2017,2018,2019,2020]:
    #a=i
    
def outlier_iqr(df, whis=1.5,columna_evaluar='score', columna_resultado='outlier',lim='both'):
    primer_cuartil = df[columna_evaluar].quantile(.25)
    tercer_cuartil = df[columna_evaluar].quantile(.75)
    whisker = whis
    iqr = tercer_cuartil - primer_cuartil
    
    print('iqr = ',round(iqr,3))

    limite_inferior = primer_cuartil - (iqr * whisker)
    limite_superior = tercer_cuartil + (iqr * whisker)
    
    limites = (round(limite_inferior,2), round(limite_superior,2))
    print(limites)

    if lim=='both':
        df.loc[df[columna_evaluar] < limite_inferior, columna_resultado] = True
        df.loc[df[columna_evaluar] > limite_superior, columna_resultado] = True
    else:
        if lim=='sup':
            df.loc[df[columna_evaluar] > limite_superior, columna_resultado] = True
        else:
            df.loc[df[columna_evaluar] < limite_inferior, columna_resultado] = True

    print(df[columna_resultado].value_counts())
    




# Empezamos filtrando el dataset de jugadores, eliminando todos aquellos que 
## no alcancen un mínimo de tiempo disputado.

def expanding_dfs():
    df_jug = pd.read_excel(ruta_datos+'/datos_jugadores_instat.xlsx')
    df_equipos = pd.read_excel(ruta_datos+'/datos_equipos_instat.xlsx')

    """DETECCION DE OUTLIERS"""
    # Metodo de rango intercuartil para dos columnas:
    ## Porcentaje de partidos jugados respecto al máximo de la competición
    ## De Sustitución
    
    gr_partidos = df_jug.groupby(by='league-id-instat',as_index=False)['Partidos jugados'].agg(['mean','max']).reset_index()
    
    df_jug = pd.merge(df_jug,gr_partidos[['league-id-instat','max']],how='left',on='league-id-instat')
    df_jug['PJ_maxleague'] = df_jug['Partidos jugados'] / df_jug['max']
    
    
    df_jug['outlier'] = False
    outlier_iqr(df_jug,0.3,columna_evaluar = "PJ_maxleague", columna_resultado='outlier', lim = "inf")
    plt.figure()
    sns.histplot(df_jug, x='PJ_maxleague', hue = 'outlier', bins=40)
    outlier_iqr(df_jug,1.5,columna_evaluar = "De sustitucion", columna_resultado='outlier', lim = "sup")    
    plt.figure()
    sns.histplot(df_jug, x='De sustitucion', hue = 'outlier',bins=40)
    df_jug_clean = df_jug[(df_jug['outlier']!=True) & (df_jug['Partidos jugados']>7)]
    
    
    team_cols = []
    for i in df_equipos.columns:
        if is_numeric_dtype(df_equipos[i])==True and 'Poses' in i:
            team_cols.append(i)
            
    df_jug_clean = pd.merge(df_jug_clean,df_equipos[['teamid','Posesiones de balón, cantidad',
                                                     'Posesión del balón, %']],how='left',on='teamid')
    
    df_equipos = mf.get_players_data(df_equipos,df_jug_clean,['Asistencias','Expected assists'])
    
    """NUEVAS COLUMNAS"""
    # OneHotEncoder para la pierna del jugador
    # gini en equipos para npxg+xa
    # id de escudo en equipos y foto en jugador
    # PosE
    logos = []
    for i in list(df_equipos.teamid.unique()):
        l= 'https://tmssl.akamaized.net/images/wappen/head/{}.png?lm=1467356331'.format(i)
        logos.append(l)
        
    df_equipos['logo'] = logos
    
    df_jug_clean = pd.merge(df_jug_clean,df_equipos[['teamid','logo']],on='teamid',how='left')
    
    
    
    
    df_jug_clean = mf.position_mapping(df_jug_clean)
    df_jug_clean = mf.metrics_players(df_jug_clean)
    df_equipos = mf.metrics_squads(df_equipos)
    
    ginix={}
    for i in df_jug_clean['teamid'].unique():
        p=mf.gini(df_jug_clean[df_jug_clean['teamid']==i],'xg+xa')
        ginix[i]=p   
    gini_chain = pd.DataFrame.from_dict(ginix,orient='index',columns=['Gini']).reset_index()
    gini_chain.rename({'index':'teamid'},inplace=True,axis=1)
    df_equipos= pd.merge(df_equipos,gini_chain,how='left',on='teamid')
    
    df_jug_clean = mf.possession_adj(df_jug_clean)
    df_equipos = mf.possession_adj(df_equipos)
    df_jug_clean = mf.actions_adj(df_jug_clean)
    
    
    df_equipos = df_equipos[df_equipos['league-id-instat'].isin(list(df_jug['league-id-instat'].unique()))]
    df_equipos = df_equipos[df_equipos['teamid'].isin(list(df_jug['teamid'].unique()))]
    
    encoder = OneHotEncoder(sparse=False)
    codif = encoder.fit_transform(df_jug_clean[['Pierna']])
    df_jug_clean[encoder.get_feature_names()] = codif
    pies_list = []
    for i in df_jug_clean.columns:
        if 'x0' in i:
            pies_list.append(i)
    
    df_jug_clean['Wing_Natural'] = 0
    df_jug_clean['Wing_Natural'] = np.where((df_jug_clean['x0_Derechа']==1) & (df_jug_clean['Posición'].isin(['LD','ED'])) | ((df_jug_clean['x0_Zurda']==1) & (df_jug_clean['Posición'].isin(['LI','EI']))),1,0)
    df_jug_clean['Wing_Natural'] = np.where((df_jug_clean['Posición'].isin(['LD','LI','ED','EI'])) & (df_jug_clean['x0_Ambidiestro']==1),1,df_jug_clean['Wing_Natural'])
    df_jug_clean['x0_Derechа'] = np.where(df_jug_clean['x0_Ambidiestro']==1,1,df_jug_clean['x0_Derechа'])      
      
    for i in pies_list:
        if i in df_jug_clean.columns and 'Derecha' not in i:
            df_jug_clean.drop(i,inplace=True,axis=1)
            
    replace_dict = {
    r"\bDC\b": "DFC",
    r"\bD\b": "DC"
    }
    
    for i in replace_dict:
        df_jug_clean['Posición'] = df_jug_clean['Posición'].str.replace(i, replace_dict[i], regex=True)
        
    dup = df_jug_clean[df_jug_clean.Nombre.duplicated(keep=False)]
    df_jug_clean['Nombre'] = np.where(df_jug_clean.index.isin(list(dup.index)),df_jug_clean.Players,df_jug_clean.Nombre)
    
    df_jug_clean.to_csv(ruta_base+'/Datos/Modeled/jugadores.csv',sep=';',decimal=',',
                        index=False)
    df_equipos.to_csv(ruta_base+'/Datos/Modeled/equipos.csv',sep=';',decimal=',',
                      index=False)
    
#expanding_dfs()