# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 20:18:48 2023

@author: aleex
"""

import pandas as pd
import numpy as np
import modeling
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from yellowbrick.cluster.elbow import kelbow_visualizer
from sklearn.preprocessing import MinMaxScaler,RobustScaler
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import math
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
import clustering_functions as cf

def elbow_method(X, max_range_for_elbow):
    return kelbow_visualizer(KMeans(random_state=0), X, k=(1, max_range_for_elbow)) 

np.random.seed(4)
ruta_datos = os.path.join(os.getcwd(),"Datos")
scaler=MinMaxScaler()
sns.set(style="whitegrid")

players = pd.read_csv(ruta_datos+'/Modeled/jugadores.csv',decimal=',',sep=';')
squad = pd.read_csv(ruta_datos+'/Modeled/equipos.csv',decimal=',',sep=';')

"""
A partir de datos de equipos construir un modelo de clusterización + clasificación
que pueda llegar a segmentar el modo de jugar de los equipos
(defending&pressing, buildup&passing, chance-creating actions, shot&finishing, organization)

""" 
system_cols = []
for i in squad.columns:
    if ('Use%_' in i and 'k' in i):
        system_cols.append(i)
        
        
def_cols = list(set(
    ['% presión efectiva','High pressing, %','Presiones/hPress','PAdj_Presión del equipo','Presiones/AtaquesPos',
     'Pases del rival por acción defensiva','PAdj_Recuperaciones en campo rival','Presiones/100Recuperaciones',
     'Recuperaciones_crival%','Recuperaciones_crival/perdidas','HighPress/100RecuperacionesCRival',
'PAdj_Disputas defensivas','PAdj_Disputas por arriba ganadas','Robos de balón con éxito, %',
 '% disputas defensivas ganadas','% disputas por arriba ganadas','Acciones defensivas/100Acciones',
 'Interceptaciones/100Acciones defensivas','PAdj_Balones recuperados','Interceptaciones/100Acciones defensivas',
 'Interceptaciones/100Balones recuperados','Rechaces/100Acciones defensivas','Rechaces/100Balones recuperados',
 'Disputas defensivas/100Acciones defensivas','Disputas defensivas/100Balones recuperados',
 'Presión del equipo/100Acciones defensivas','Presión del equipo/100Balones recuperados',
 'Balones recuperados/100Acciones defensivas','Entradas/100Acciones defensivas',
 'Entradas/100Balones recuperados','Faltas/100Acciones defensivas','Faltas/100Balones recuperados'
 ]))

buildup_cols = list(set(
    ['Juego por Banda, %','pace','Centros/100Pases','Juego por Banda Derecha, %','Contraataques/100Bups',
     'Posesión del balón, %','Contraataques/100Recuperaciones','Bups/100Posesiones','Contraataques/100Posesiones','PF/100Pases',
     'IncursionesARival/Pases','IncursionesUT/Regates','IncursionesCRival/Pases','IncursionesUT/Pases',
     'Centros/100IncARival','Presiones/100PerdidasCRival','Posesión del balón, seg',
     'Entrada al último tercio de campo','% de efectividad de pases']))

cca_cols = list(set(
    ['IncursionesARival/Pases','Entrada al área rival','KP/100Pases','KP_Ocasiones%',
     'Jugada_Gol/100Regates','Jugada_Gol/100Centros','Jugada_Gol/100IncursionesCRival','Jugada_Gol/100IncursionesUT',
     'Jugada_Gol/100PF','xA/CC','Gini','xA/PFe','xG/Jugada_Gol','Jugada_Gol/100ABP',
     'Jugada_Gol/100Contraataques','Jugada_Gol/100Bups',"Tiros/JugadaGol",'Tiros/100Acciones',
     'Centros/100AccionesBanda','Centros/100IncARival','KP/100IncARival','xG/100Acciones',
     'CC/100IncursionesUT','% disputas en ataque ganadas'
     ]))

cat_dict= {'disposicion_tactica':system_cols,
           'defensa':def_cols,
           'buildup':buildup_cols,
           'creacion_oportunidades':cca_cols}


cbs = list(set(['% disputas por arriba ganadas','PAdj_Disputas por arriba ganadas','PAdj_Interceptaciones+Entradas',
                'PAdj_Rechaces','PAdj_Entradas','Perdidas_crival%','Recuperaciones_crival%',
                'Entradas/100Acciones defensivas','Entradas, %','Robos de balón con éxito, %',
        '% disputas defensivas ganadas','PAdj_Disputas defensivas','Disputas defensivas/100Acciones defensivas',
        'PAdj_Faltas','Disputas aéreas/100Disputas','Disputas defensivas/100Disputas',
        'PAdj_Balones recuperados','PAdj_Acciones defensivas','Pases/100Acciones defensivas',
        '% de efectividad de pases','PAdj_Rechaces'
    ]))

mid = list(set([
    'KP/100Acciones no defensivas','Pases/100Acciones defensivas','Regates/100Acciones no defensivas','CC/100Acciones no defensivas','PAdj_Interceptaciones+Entradas',
    '% de efectividad de pases','PAdj_Disputas por arriba ganadas', 'Perdidas_crival%','Recuperaciones_crival%',
    'Disputas en ataque/100Disputas','Centros/100Pases','Robos de balón con éxito, %',
    'PAdj_Balones recuperados','PAdj_Disputas ganadas','% de efectividad de pases',
    'Entradas/100Acciones defensivas','PAdj_Interceptaciones','PAdj_Rechaces',
    'xG per shot','Tiros/100Acciones no defensivas','xA/PFe','PF/100Pases','Entradas, %',
    'Regates/100Acciones no defensivas', 'Acciones no defensivas','Pases de finalización efectivos'
    ]))


flb=list(set(['Regates/100Acciones no defensivas','Centros/100PF','Centros/100Pases',
              'Centros/100Acciones no defensivas','PF/100Acciones no defensivas',
     'Regates/100Centros','% de efectividad de pases','Entradas/100Acciones defensivas','% de efectividad de los centros',
     'Perdidas_crival%','Recuperaciones_crival%','PAdj_Interceptaciones+Entradas',
     'Acciones defensivas/100Acciones','KP/100Acciones no defensivas',
     'CC/100Centros','xA/PFe','PF/100Pases','Entradas, %','PAdj_Balones recuperados',
     'Expected assists','Disputas/100Acciones defensivas','Jugada_Gol/100Centros',
     'Pases/100Acciones defensivas','Tiros/100Acciones no defensivas','Wing_Natural']
             ))

attm=list(set(['Regates/100Acciones no defensivas','Centros/100PF','Centros/100Pases',
              'Centros/100Acciones no defensivas','PF/100Acciones no defensivas',
     'Regates/100Centros','% de efectividad de pases','xG per shot',
     'Perdidas_crival%','Recuperaciones_crival%','xG/Jugada_Gol','xg+xa/100Acciones',
     'Acciones defensivas/100Acciones','KP/100Acciones no defensivas','xA/CC',
     'CC/100Centros','CC/100Regates','xA/PFe','PF/100Pases','KP_Ocasiones%','CC/100PF','Disputas aéreas/100Disputas',
     'Disputas en ataque/100Disputas','Jugada_Gol/100Centros','% disputas por arriba ganadas',
     'Jugada_Gol/100Regates','Tiros/100Acciones no defensivas','Wing_Natural']))

fwd=list(set(['Regates/100Acciones no defensivas','Centros/100PF','Centros/100Pases',
              'Centros/100Acciones no defensivas','PF/100Acciones no defensivas',
     'Regates/100Centros','xG conversion','xG per shot','Ocasiones de gol, % conversión',
     'xG/Jugada_Gol','xg+xa/100Acciones',
     'Acciones defensivas/100Acciones','KP/100Acciones no defensivas','xA/CC',
     'CC/100Centros','CC/100Regates','xA/PFe','PF/100Pases','KP_Ocasiones%','CC/100PF','Disputas aéreas/100Disputas',
     'Disputas en ataque/100Disputas','Jugada_Gol/100Centros','% disputas por arriba ganadas',
     'Jugada_Gol/100Regates','Tiros/100Acciones no defensivas','Altura']))


pos_dict= {'Centre-Back':cbs,
           'Midfielder':mid,
           'Att. Midfield/Winger':attm,
           'Full-Back':flb,
           'Forward':fwd}




#columnas_e_sq = list(set(isna_check(squad,cat_dict)))
columnas_e_pl = list(set(cf.isna_check(players)))
for c in columnas_e_pl:
    players[c] = players[c].fillna(0)


df,features = cf.squad_clustering(list(cat_dict.keys()))

for i in df.columns:
    if '_cluster' in i:
        df[i] = df[i].astype(str)


cf.get_squad_features(features,df)


df,features,ks,pcas = cf.player_clustering(list(players.PosE.unique()))

#players2 = players[players['Min']>=min_minutes]

for i in df:
    k = df[i]
    i = i.replace('/','-')
    k.to_csv(ruta_datos+'/Modeled/{}_clustered.csv'.format(i),
             decimal=',',sep=';',index=False)
    
for i in features:
    k = features[i]
    k = k.reset_index()
    k.columns = ['feat','influence']
    i = i.replace('/','-')
    k.to_csv(ruta_datos+'/Modeled/{}_features.csv'.format(i),
             decimal=',',sep=';',index=False)
    



cf.get_player_features(features,df)
        