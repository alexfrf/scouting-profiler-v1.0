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
from clustering_functions import cat_dict

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
pcas_df = pd.DataFrame.from_dict(pcas,orient='index',columns = ['PCA']).reset_index()
pcas_df.rename({'index':'pos'},axis=1,inplace=True)
pcas_df.to_csv(ruta_datos+'/Modeled/pca_positions.csv',
               index=False,sep=';',decimal=',')
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
        