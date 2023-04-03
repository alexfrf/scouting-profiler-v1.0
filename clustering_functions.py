# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 23:33:30 2023

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
    'KP/100Acciones no defensivas','Pases/100Acciones defensivas','Regates/100Acciones no defensivas',
    'CC/100Acciones no defensivas','PAdj_Interceptaciones+Entradas',
    '% de efectividad de pases','PAdj_Disputas por arriba ganadas', 'Perdidas_crival%','Recuperaciones_crival%',
    'Disputas en ataque/100Disputas','Centros/100Pases','Robos de balón con éxito, %',
    'PAdj_Balones recuperados','PAdj_Disputas ganadas','% de efectividad de pases','Pases efectivos',
    'Entradas/100Acciones defensivas','PAdj_Interceptaciones','PAdj_Rechaces',
    'xG per shot','Tiros/100Acciones no defensivas','xA/PFe','PF/100Pases','Entradas, %',
    'Regates/100Acciones no defensivas', 'Pases/100Acciones','Pases de finalización efectivos'
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
     'Regates/100Centros','% de efectividad de pases','xG per shot','Regates',
     'Perdidas_crival%','Recuperaciones_crival%','xG/Jugada_Gol','xg+xa/100Acciones',
     'Acciones defensivas/100Acciones','KP/100Acciones no defensivas','xA/CC',
     'CC/100Centros','CC/100Regates','xA/PFe','PF/100Pases','KP_Ocasiones%','CC/100PF','Disputas aéreas/100Disputas',
     'Jugada_Gol/100Centros','% disputas por arriba ganadas',
     'Jugada_Gol/100Regates','Tiros/100Acciones no defensivas','Wing_Natural']))

fwd=list(set(['Regates/100Acciones no defensivas','Centros/100PF','Centros/100Pases',
              'Centros/100Acciones no defensivas','PF/100Acciones no defensivas',
     'Regates/100Centros','xG conversion','xG per shot','Ocasiones de gol, % conversión',
     'xG/Jugada_Gol','xg+xa/100Acciones',
     'Acciones defensivas/100Acciones','KP/100Acciones no defensivas','xA/CC',
     'CC/100Centros','CC/100Regates','xA/PFe','PF/100Pases','KP_Ocasiones%','CC/100PF','Disputas aéreas/100Disputas',
     'Jugada_Gol/100Centros','% disputas por arriba ganadas',
     'Jugada_Gol/100Regates','Tiros/100Acciones no defensivas','Altura']))


pos_dict= {'Centre-Back':cbs,
           'Midfielder':mid,
           'Att. Midfield/Winger':attm,
           'Full-Back':flb,
           'Forward':fwd}


def isna_check(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    cols_na = []
    for i in df.columns:
        if df[i].isna().any()==True:
            cols_na.append(i)
            df[i] = df[i].fillna(0)
    
    return cols_na

#columnas_e_sq = list(set(isna_check(squad,cat_dict)))
columnas_e_pl = list(set(isna_check(players)))
for c in columnas_e_pl:
    players[c] = players[c].fillna(0)

def pca_opt():
    for i in list(cat_dict.keys()):
        name = i
        cols = cat_dict[i]
        data = squad[cols]
        data_norm = scaler.fit_transform(data)
        sel = VarianceThreshold(threshold=0.01)
        sel.fit(data_norm)
        print('Num. of variables that are not constant or cuasi-constant: ', sum(sel.get_support()), ' out of {}'.format(data_norm.shape[1]))
        data_norm =  sel.transform(data_norm)
         # 2D PCA for the plot
        for p in range(2,10):
            pca = PCA(n_components = p)
            print('PCA({}) for {}'.format(p,name))
            pca.fit(data_norm)
            #reduced = pd.DataFrame(pca.fit_transform(data_norm))
            print('Explained var = {}'.format(round(sum(pca.explained_variance_ratio_),2)))
            plt.figure()
            plt.plot(range(0,p), list(pca.explained_variance_ratio_))
            plt.ylabel('Explained Variance')
            plt.xlabel('Principal Components')
            plt.xticks(range(0,p),rotation=60)
            plt.title('Explained Variance Ratio PCA{}'.format(p))
            plt.show()


def squad_clustering(categories):
    #np.random.seed(0)
    sns.set(style="whitegrid")
    squad_clustering = pd.DataFrame(squad[['Equipo']],columns=['Equipo'])
    for i in squad.columns:
        if 'cluster' in i:
            squad.drop(i,inplace=True,axis=1)
    fs=[]
    for i in categories:
        name = i
        print('-----Starting clustering for {}-----'.format(name))
        cols = cat_dict[i]
        data = squad[cols]
        
        sel = VarianceThreshold(threshold=0)
        sel.fit(data)
        print('Num. of variables that are not constant: ', sum(sel.get_support()), ' out of {}'.format(data.shape[1]))
        if sum(sel.get_support())>0:
            for x in data.columns:
                if x not in data.columns[sel.get_support()]:
                    print(x)
        data_norm = scaler.fit_transform(data)
        data_norm =  sel.transform(data_norm)
         # 2D PCA for the plot
        pca_app = 0
        for p in range(2,10):
            pca = PCA(n_components = p,random_state=0)
            print('PCA({}) for {}'.format(p,name))
            #reduced = pd.DataFrame(pca.fit_transform(data_norm))
            pca.fit(data_norm)
            print('Explained var = {}'.format(round(sum(pca.explained_variance_ratio_),2)))
            if sum(pca.explained_variance_ratio_)>=0.9:
                pca_app = p
                print('Selected PCA: {}'.format(p))
                break
            else:
                continue

        plt.plot(range(0,p), list(pca.explained_variance_ratio_))
        plt.ylabel('Explained Variance')
        plt.xlabel('Principal Components')
        plt.xticks(range(0,p),rotation=60)
        plt.title('Explained Variance Ratio PCA{}'.format(pca_app))
        plt.show()
        
        pca = PCA(n_components = pca_app,random_state=0) 
        reduced = pd.DataFrame(pca.fit_transform(data_norm))
        # We intend to have, as min, four clusters
        view = elbow_method(reduced,10)
        if view.elbow_value_<4:
            n = 4
        else:
            if view.elbow_value_>6:
                n = 6
            else:
                n = view.elbow_value_
                
        
#view.show()
        k = KMeans(n_clusters = n,random_state=0)

        reduced['cluster'] = k.fit_predict(reduced)
        reduced['cluster'] = reduced['cluster'] +1
        data = pd.merge(data,reduced[['cluster']],how='left',left_index=True,right_index=True)
        corr_matrix = data.corr()
        
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        upper  
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        data= data.drop(to_drop,axis=1)
        data.drop('cluster',axis=1,inplace=True)
        # Based on important information
        
        
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(scaler.fit_transform(data), reduced['cluster'])
        feature_scores = pd.Series(clf.feature_importances_, index=data.columns).sort_values(ascending=False)
        
        f, ax = plt.subplots(figsize=(30, 20))
        ax = sns.barplot(x=feature_scores, y=feature_scores.index)
        ax.set_title("Feature scores of {} in squads".format(name),size=30)
        ax.set_yticklabels(feature_scores.index,size=22)
        ax.set_xlabel("Feature importance score",size=20)
        ax.set_ylabel("Features",size=20)
        plt.savefig(os.path.join(os.getcwd(),'Model','Teams')+"/Feature_scores_{}".format(name),dpi=100)
        plt.show()
        
        feature_scores_to_drop = feature_scores[feature_scores.cumsum()>=.9]
        data.drop(list(feature_scores_to_drop.index),inplace=True,axis=1)
  
        cl=[]
        for i in reduced.columns:
            if type(i)==int:
                i+=1
                cl.append('pc'+str(i))
            else:
                cl.append(i)
        reduced.columns = cl
        #reduced.columns=['pc1','pc2','cluster']
        #centroids = k.cluster_centers_
        reduced = pd.merge(reduced,squad[['Equipo','last_coach','league-instat']],how='left',left_index=True,right_index=True)
        
        sns.set(style="white")
        ax= sns.lmplot( x="pc1", y="pc2", hue='cluster', data = reduced, legend=True,
         size = 8, scatter_kws={"s": 100},aspect=1.5,height=4, palette='muted')
        ax=plt.gca()
        ax.set_title('PCA({}) Clustering Distribution - {} in squads // Two Principal Components'.format(pca_app,name.title()),size=12)
        texts = []
        reduced_sampled = reduced.sample(frac=1).head(100)
        for x, y, s in zip(reduced_sampled.pc1, reduced_sampled.pc2, reduced_sampled.Equipo):
            texts.append(plt.text(x, y, s))
        print(reduced_sampled[['Equipo','last_coach','league-instat','cluster']].head(10))
            
            
        
        #ax.set(ylim=(-5, 5))
        #plt.scatter(centroids[:,0], centroids[:,1], c=range(centroids.shape[0]), s=1000)
        plt.tick_params(labelsize=10)
        plt.xlabel("PC1", fontsize = 15)
        plt.ylabel("PC2", fontsize = 15)
        plt.tight_layout()
        #ax.legend(frameon =True,shadow=True,loc="upper right")
        plt.savefig(os.path.join(os.getcwd(),'Model','Teams')+"/PCA{}_{}clusters_{}.png".format(pca_app,n,name),dpi=100)
        plt.show()
        
        #data_gr = data.groupby(by='cluster',as_index=False).mean()
        data = pd.merge(data,reduced[['cluster']],how='left',left_index=True,right_index=True)
        squad_clustering[name+'_cluster'] = data[['cluster']]
        fs.append(pd.DataFrame(feature_scores,columns=[name]))
        
    squad_clustering['cluster_clas'] = squad_clustering['disposicion_tactica_cluster'].astype(str).str.replace('.0','') + squad_clustering['defensa_cluster'].astype(str).str.replace('.0','') + squad_clustering['buildup_cluster'].astype(str).str.replace('.0','') + squad_clustering['creacion_oportunidades_cluster'].astype(str).str.replace('.0','')
    squad_clustering.rename({'Squad':'Equipo'},inplace=True,axis=1)
    squad_merged = pd.merge(squad,squad_clustering,how='left',on='Equipo')
    squad_merged.to_csv(ruta_datos+'/Modeled/equipos.csv',sep=';',decimal=',',index=False)
    return squad_merged,fs

"""
df,features = squad_clustering(list(cat_dict.keys()))

for i in df.columns:
    if '_cluster' in i:
        df[i] = df[i].astype(str)

"""
def get_squad_features(feat_team,data):

    for f in feat_team:
        clu = list(f.columns)[0]
        print('-----CATEGORY: {}-----'.format(clu))
        perf = data.groupby(by=clu+'_cluster')['Puntos esperados'].mean()
        perf =round(perf,3)
        means = list(perf.values)
        f = f[:5]
        n=0
        for k in f.index:
            n+=1
            col=k
            val = round(f[clu][col],5)*100
            fig = plt.figure(figsize = (10, 6))
            ax = fig.add_subplot(111)
            
            sns.boxplot(x = "{}_cluster".format(clu),
                        y = col,
                        orient = "v",
                        data = data,
                        ax = ax,
                        palette='muted')
            
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)
            
            ax.set_title("{} - Clusterized Squads\nExplained: {:.2f}% // Avg League Pos.: {}".format(col,val,means))
            ax.set_xlabel("Cluster - based on {} criteria".format(clu.upper()))
            ax.set_ylabel(col)
            sns.despine()
            plt.grid(True, alpha = 0.4)
            plt.savefig(os.path.join(os.getcwd(),'Model','Teams')+"/{}_cluster_metric{}.png".format(clu,str(n)),dpi=100)
            plt.show();

#get_squad_features(features)

def player_clustering(position):
    #np.random.seed(0)
    fs={}
    frames={}
    ks = {}
    pcas={}
    positions=[]
    if type(position)==list:
        positions=position
    else:
        positions.append(position)
    for position in positions:
        
        #pl = players[players.Min>=min_minutes]
        #pl_clustering = pd.DataFrame(players[['Player']],columns=['Player'])
        positioner = players[players.PosE==position]
        name = positioner.PosE.unique()[0]
        print('-----Starting clustering for {}s-----'.format(name))
        data_pos= players[players['PosE']==name]
        data_pos.reset_index(inplace=True)
        data_pos.drop('index',inplace=True,axis=1)
        data = data_pos[pos_dict[name]]
    
        
        sel = VarianceThreshold(threshold=0.01)
        sel.fit(data)
        print('Num. of variables that are not constant or cuasi-constant: ', sum(sel.get_support()), ' out of {}'.format(data.shape[1]))
        
        if sum(sel.get_support())>0:
            for x in data.columns:
                if x not in data.columns[sel.get_support()]:
                    print(x)
         # 2D PCA for the plot
        data_norm = scaler.fit_transform(data)
        data_norm =  sel.transform(data_norm)
        
        pca_app = 0
        print('Setting PCA for training')
        for p in range(2,100):
            pca = PCA(n_components = p,random_state=0,svd_solver='arpack')
            #print('PCA({}) for {}'.format(p,name))
            #reduced = pd.DataFrame(pca.fit_transform(data_norm))
            pca.fit(data_norm)
            #print('Explained var = {}'.format(round(sum(pca.explained_variance_ratio_),2)))
            if sum(pca.explained_variance_ratio_)>=0.9:
                pca_app = p
                print('Selected PCA: {}'.format(p))
                break
            else:
                continue
    
        reduced = pd.DataFrame(pca.transform(data_norm))
        # We intend to have, as min, four clusters
        plt.figure(figsize=(10,5))
        view = elbow_method(reduced,10)

        if view.elbow_value_>=5:
            n = 4
        elif view.elbow_value_<3:
            n=3
        else:
            n = view.elbow_value_
        
        if 'Full-' in position:
            n = 4    
        #view.show()
        k = KMeans(n_clusters = n,random_state=0)
        pcas[position] = pca_app
        # We intend to have, as min, four clusters
        
        ks[position] = n
        reduced['cluster'] = k.fit_predict(reduced)
        reduced['cluster'] = reduced['cluster'] +1
        data = pd.merge(data,reduced[['cluster']],how='left',left_index=True,right_index=True)
        corr_matrix = data.corr()
        plt.figure(figsize=(30,6))
        a = sns.heatmap(corr_matrix, square=True, annot=True, fmt='.2f', linecolor='black')
        a.set_xticklabels(a.get_xticklabels(), rotation=30,size=8)
        a.set_yticklabels(a.get_yticklabels(), rotation=30,size=8)           
        plt.show()   
        
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        upper  
        to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
        data= data.drop(to_drop,axis=1)
        if 'cluster' in data.columns:
            data.drop('cluster',axis=1,inplace=True)
        # Based on important information
        
        
        clf = RandomForestClassifier(n_estimators=100, random_state=0)

        #data = pd.merge(data,reduced[['cluster']],how='left',left_index=True,right_index=True)

        clf.fit(scaler.fit_transform(data), reduced['cluster'])
        feature_scores = pd.Series(clf.feature_importances_, index=data.columns).sort_values(ascending=False)
        
        f, ax = plt.subplots(figsize=(30, 20))
        ax = sns.barplot(x=feature_scores, y=feature_scores.index)
        ax.set_title("Feature scores of {} in Players clusterization".format(name.replace('/','_')),size=30)
        ax.set_yticklabels(feature_scores.index,size=22)
        ax.set_xlabel("Feature importance score",size=20)
        ax.set_ylabel("Features",size=20)
        plt.savefig(os.path.join(os.getcwd(),'Model','Players')+"/Feature_scores_{}".format(name.replace('/','-').replace('.','')),dpi=100)
        plt.show()
        
        feature_scores_to_drop = feature_scores[feature_scores.cumsum()>=.9]
        data.drop(list(feature_scores_to_drop.index),inplace=True,axis=1)
        
        feature_scores = feature_scores[~feature_scores.index.isin(list(feature_scores_to_drop.index))]


        plt.plot(range(0,p), list(pca.explained_variance_ratio_))
        plt.ylabel('Explained Variance')
        plt.xlabel('Principal Components')
        plt.xticks(range(0,p),rotation=60)
        plt.title('Explained Variance Ratio PCA{}'.format(pca_app))
        plt.show()


        
        cl=[]
        for i in reduced.columns:
            if type(i)==int:
                i+=1
                cl.append('pc'+str(i))
            else:
                cl.append(i)
        reduced.columns = cl
        if 'Nombre' in cl:
            reduced.drop('Nombre',inplace=True,axis=1)
        #reduced.columns=['pc1','pc2','cluster']
        #centroids = k.cluster_centers_
        reduced = pd.merge(reduced,data_pos[['Nombre']],how='left',left_index=True,right_index=True)
        
        sns.set(style="white")
        ax= sns.lmplot( x="pc1", y="pc2", hue='cluster', data = reduced, legend=True,
         size = 8, scatter_kws={"s": 100},aspect=1.5,height=4, palette='muted')
        ax=plt.gca()
        ax.set_title('PCA({}) Clustering Distribution - {}s // Two Principal Components'.format(pca_app,name.title()),size=12)
        """
        texts = []
        for x, y, s in zip(reduced.pc1, reduced.pc2, reduced.Player):
            if abs(x)>abs(reduced['pc1'].mean()) or abs(y)>abs(reduced['pc2'].mean()):
                texts.append(plt.text(x, y, s))
        """
        #texts = []
        reduced_sampled = reduced.sample(frac=1).head(100)
        """
        for x, y, s in zip(reduced_sampled.pc1, reduced_sampled.pc2, reduced_sampled.Nombre):
            texts.append(plt.text(x, y, s))
        """
        print(reduced_sampled[['Nombre','cluster']].head(10))
        
        #ax.set(ylim=(-5, 5))
        #plt.scatter(centroids[:,0], centroids[:,1], c=range(centroids.shape[0]), s=1000)
        plt.tick_params(labelsize=10)
        plt.xlabel("PC1", fontsize = 15)
        plt.ylabel("PC2", fontsize = 15)
        plt.tight_layout()
        #ax.legend(frameon =True,shadow=True,loc="upper right")
        #plt.savefig(os.path.join(os.getcwd(),'Model','Players')+"/PCA{}_{}clusters_{}.png".format(pca_app,n,name.upper()),dpi=100)
        plt.show();
        
        data = pd.merge(data,reduced[['cluster']],how='left',left_index=True,right_index=True)
        #data_gr = data.groupby(by='cluster',as_index=False).mean()
        player_clustering = data_pos
        player_clustering['cluster'] = data[['cluster']]
        
        frames[position]= reduced
        fs[position]=pd.DataFrame(feature_scores,columns=[name])    
        

    return frames, fs, ks, pcas
"""
df,features,ks,pcas = player_clustering(list(players.PosE.unique()))

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
    

"""


def get_player_features(feat,data):
    for f in feat.keys():
        clu = f
        print('-----POSITION: {}-----'.format(clu))
    
        d = data[clu]
        #pl = players.set_index('Nombre')
        ls=list(feat[clu].index)
        ls.append('Nombre')
        d = pd.merge(d,players[ls],how='left',on='Nombre')
        f = feat[clu][:8]
        n=0
        for k in f.index:
            n+=1
            col=k
            val = round(f[clu][col],5)*100
            fig = plt.figure(figsize = (10, 6))
            ax = fig.add_subplot(111)
            
            sns.boxplot(x = "cluster",
                        y = col,
                        orient = "v",
                        data = d,
                        ax = ax,
                        palette='muted')
            
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)
            
            ax.set_title("{} - Clusterized Players\nExplained: {:.2f}%".format(col,val))
            ax.set_xlabel("Cluster - based on position: {}".format(clu.upper()))
            ax.set_ylabel(col)
            sns.despine()
            plt.grid(True, alpha = 0.4)
            plt.savefig(os.path.join(os.getcwd(),'Model','Players')+"/{}_cluster_metric{}.png".format(clu.replace('/','-').replace('.',''),str(n)),dpi=100)
            plt.show();

#get_player_features(features)


def player_similarities(p, data):
    np.random.seed(4)
    pos = data[data['Nombre']==p].PosE.values[0]
    pos2 = pos.replace('/','-')
    print('Looking for similar profiles to {}'.format(p.upper()))
    player_name = p.title()
    df = pd.read_csv(ruta_datos+'/Modeled/{}_clustered.csv'.format(pos2),sep=';',decimal=',')
    data_player = data[data['Nombre']==p]
    idx = data_player.ID.unique()[0]
    name = data_player.PosE.unique()[0]
    name2 = data_player['Posición'].unique()[0]
    pl = data[(data.PosE==name)]
    
    data_pos = df
    pl = pl.reset_index()
    pl = pl.drop('index',axis=1)

    reduced=data_pos

    reduced = pd.merge(reduced,pl[['ID','Posición','Edad','Equipo','league-instat','Nacionalidad']],how='left',left_index=True,right_index=True)
    y = reduced[reduced['ID']==idx]
    y.drop(['ID','Nombre','Posición','Edad','Equipo','league-instat','Nacionalidad','cluster'],inplace=True,axis=1)
    pl = list(reduced.Nombre)
    pos = list(reduced['Posición'])
    eda = list(reduced.Edad)
    eq = list(reduced.Equipo)
    le = list(reduced['league-instat'])
    nac = list(reduced.Nacionalidad)
    idx = list(reduced.ID)
    cl = list(reduced.cluster)
    reduced.drop(['ID','Nombre','Posición','Edad','Equipo','league-instat','Nacionalidad','cluster'],inplace=True,axis=1)
    euc = []
    for i in reduced.values:
        euc.append(euclidean_distances(np.array(y),[i])[0][0])
    simil = pd.DataFrame(euc,index=[idx,pos,eda,eq,le,nac,cl],columns=['Similarity_Score'])
    simil = simil.reset_index()
    simil.columns = ['ID','Posición','Edad','Equipo','league-instat','Nacionalidad','cluster','Similarity_Score']
    #simil = simil[simil.Nombre.str.title()!=player_name.title()]
    if name2[0] == 'L':
        simil = simil[simil['Posición']==name2]
        
    simil = simil.sort_values(by='Similarity_Score',ascending=True)
    simil = pd.merge(simil,data[['ID','Nombre']],how='left',on='ID')
    simil = simil[simil.Nombre!=p]
    simil = simil.set_index('Nombre')
    simil.drop('ID',axis=1,inplace=True)



    return simil



def team_mapping(team,position, data_team, data_player, cats):
    pos = data_player[data_player['Posición']==position].PosE.values[0]
    pos2 = pos.replace('/','-')
    #players,features,ks,pcas = player_clustering(position=klist, data = data_player, cat_input = cats)
    
    players = pd.read_csv(ruta_datos+'/Modeled/{}_clustered.csv'.format(pos2),sep=';',decimal=',')
    cluster_cols=[]
    
    for i in data_team.columns:
        if 'cluster' in i:
            cluster_cols.append(i)
    data_team[cluster_cols] = data_team[cluster_cols].astype(str)
    cluster_cols.append('teamid')
    pl = pd.merge(data_player,data_team[cluster_cols],how='left',on='teamid')
    clus = cluster_cols
    for i in ['teamid','cluster_clas']:
        if i in clus:
            clus.remove(i)
    if (position == 'DFC' or position[0]=='L' or position=='MCD') and 'creacion_oportunidades_cluster' in clus:
        clus.remove('creacion_oportunidades_cluster')


    cluster_comb = data_team[data_team['Equipo']==team].set_index(clus)
    cluster_comb.index=cluster_comb.index.map(''.join).str.replace('.0','')
    cluster_comb = cluster_comb.index[0]
    grouped = pl[(pl['Posición']==position)]
    name = grouped.PosE.unique()[0]
    grouped = grouped.groupby(by=clus).mean()
    data_cluster = grouped[cats[name]]
    data_cluster.index = data_cluster.index.map(''.join).str.replace('.0','')
    data_cluster = data_cluster[data_cluster.index==cluster_comb]
    pcas = pd.read_csv(ruta_datos+'/Modeled/pca_positions.csv',sep=';',decimal=',')
    pca_app = pcas[pcas.pos==name].PCA.values[0]
    #pl = players[players['Min']>=min_minutes]
    #players = players[name]
    data_pos= data_player[data_player['PosE']==name]
    data_pos.reset_index(inplace=True)
    data_pos.drop('index',inplace=True,axis=1) 
    data = data_pos[cats[name]]

    data = pd.concat([data,data_cluster])
    data.reset_index(inplace=True)
    data.drop('index',inplace=True,axis=1) 

    
    data_norm = scaler.fit_transform(data)
    
    
    pca = PCA(n_components = pca_app,random_state=0)
    reduced = pd.DataFrame(pca.fit_transform(data_norm))

    y = reduced.tail(1)
    #y.drop('cluster',axis=1,inplace=True)
    reduced = reduced.head(data_pos.shape[0])
    data_pos = pd.merge(data_pos,players[['cluster','Nombre']],how='left',on='Nombre')
    reduced = pd.merge(reduced,data_pos[['Nombre','ID','Posición','teamid','Índice InStat','Values','Edad','Pierna','Equipo','Nacionalidad','cluster']],how='left',left_index=True,right_index=True)
    #reduced['idx'] = reduced['idx'].fillna('{}-{}'.format(team,position))
    #reduced['Player'] = reduced['Player'].fillna('{}-{}'.format(team,position))
    if position[0]=='E':
        reduced = reduced[reduced['Posición'].str.startswith('E')]
    else:
        reduced = reduced[reduced['Posición']==position]
    pl = list(reduced.Nombre)
    idx = list(reduced.ID)
    sq = list(reduced.teamid)
    pw = list(reduced['Índice InStat'])
    vl=list(reduced.Values)
    ag=list(reduced.Edad)
    ft=list(reduced.Pierna)
    cl = list(reduced.cluster)
    eq = list(reduced.Equipo)
    nc = list(reduced.Nacionalidad)
    reduced.drop(['Nombre','ID','Posición','teamid','Índice InStat','Values','Edad','Pierna','Equipo','Nacionalidad','cluster'],inplace=True,axis=1)
    euc = []
    for i in reduced.values:
        euc.append(euclidean_distances(np.array(y),[i])[0][0])
    simil = pd.DataFrame(euc,index=[pl,idx,sq,ag,ft,vl,pw,eq,nc,cl],columns=['Team_Similarity_Index'])
    
    simil = simil.sort_values(by='Team_Similarity_Index',ascending=True)

    simil = simil.reset_index()
    simil.columns = ['Nombre','ID','teamid','Edad','Pierna','Values','Índice InStat','Equipo','Nacionalidad','cluster','Team_Similarity_Index']
    simil['Team_Similarity_Index'] = round(simil['Team_Similarity_Index'],3)

    #simil = simil[simil['Team_Similarity_Index']!=0]
    simil['Values'] = round(simil.Values,0).astype(int)
    simil['Edad'] = simil.Edad.astype(int)
    simil['Índice InStat'] = simil['Índice InStat'].astype(int)
    simil['Nacionalidad'] = simil.Nacionalidad.apply(lambda x: x.split(',')[0].strip())
    #simil = simil[(simil.Values<=value) & (simil.Edad<=age)]
    
    
    return simil


