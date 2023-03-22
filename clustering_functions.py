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

ruta_datos = os.path.join(os.getcwd(),"Datos")
np.random.seed(4)

def isna_check(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    cols_na = []
    for i in df.columns:
        if df[i].isna().any()==True:
            cols_na.append(i)
            df[i] = df[i].fillna(0)
    
    return cols_na


def pca_opt(df,cat_input):
    for i in list(cat_input.keys()):
        name = i
        cols = cat_input[i]
        data = df[cols]
        data_norm = scaler.fit_transform(data)
        sel = VarianceThreshold(threshold=0.01)
        sel.fit(data_norm)
        #print('Num. of variables that are not constant or cuasi-constant: ', sum(sel.get_support()), ' out of {}'.format(data_norm.shape[1]))
        data_norm =  sel.transform(data_norm)
         # 2D PCA for the plot
        for p in range(2,10):
            pca = PCA(n_components = p)
            #print('PCA({}) for {}'.format(p,name))
            pca.fit(data_norm)
            #reduced = pd.DataFrame(pca.fit_transform(data_norm))
            #print('Explained var = {}'.format(round(sum(pca.explained_variance_ratio_),2)))
            plt.figure()
            plt.plot(range(0,p), list(pca.explained_variance_ratio_))
            plt.ylabel('Explained Variance')
            plt.xlabel('Principal Components')
            plt.xticks(range(0,p),rotation=60)
            plt.title('Explained Variance Ratio PCA{}'.format(p))
            plt.show()


def squad_clustering(dataf,categories, cat_input):
    #np.random.seed(0)
    squad_clustering = pd.DataFrame(dataf[['Equipo']],columns=['Equipo'])
    fs=[]
    for i in categories:
        name = i
        #print('-----Starting clustering for {}-----'.format(name))
        cols = cat_input[i]
        data = dataf[cols]
        
        sel = VarianceThreshold(threshold=0)
        sel.fit(data)
        #print('Num. of variables that are not constant: ', sum(sel.get_support()), ' out of {}'.format(data.shape[1]))
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
        
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        
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
        dataf.reset_index(inplace=True)
        dataf.drop('index',axis=1,inplace=True)
        reduced = pd.merge(reduced,dataf[['Equipo','last_coach','league-instat']],how='left',left_index=True,right_index=True)
        
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
    squad_merged = pd.merge(dataf,squad_clustering,how='left',on='Equipo')
    return squad_merged,fs

def get_squad_data(input_data, dest):
    squad,features = squad_clustering(input_data)
    squad.to_excel(dest,index=False)
    
    return squad

#squad = get_squad_data(list(cat_dict.keys()))


def get_squad_features(feat_team, cat_input):
    squad,features = squad_clustering(list(cat_input.keys()))
    for f in features:
        clu = list(f.columns)[0]
        print('-----CATEGORY: {}-----'.format(clu))
        perf = squad.groupby(by=clu+'_cluster')['Puntos esperados'].mean()
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
                        data = squad,
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


def player_clustering(position, data, cat_input):
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
        positioner = data[data.PosE==position]
        name = positioner.PosE.unique()[0]
        print('-----Starting clustering for {}s-----'.format(name))
        data_pos= data[data['PosE']==name]
        data_pos.reset_index(inplace=True)
        data_pos.drop('index',inplace=True,axis=1)
        data = data_pos[cat_input[name]]
    
        
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
        #print('Setting PCA for training')
        for p in range(2,100):
            pca = PCA(n_components = p,random_state=0,svd_solver='arpack')
            #print('PCA({}) for {}'.format(p,name))
            #reduced = pd.DataFrame(pca.fit_transform(data_norm))
            pca.fit(data_norm)
            #print('Explained var = {}'.format(round(sum(pca.explained_variance_ratio_),2)))
            if sum(pca.explained_variance_ratio_)>=0.9:
                pca_app = p
                #print('Selected PCA: {}'.format(p))
                break
            else:
                continue
    
        reduced = pd.DataFrame(pca.transform(data_norm))
        # We intend to have, as min, four clusters
        plt.figure(figsize=(10,5))
        view = elbow_method(reduced,10)
        if view.elbow_value_>=5:
            n = 4
        else:
            n = view.elbow_value_
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
        
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
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
        if 'ID' in cl:
            reduced.drop('ID',inplace=True,axis=1)
        #reduced.columns=['pc1','pc2','cluster']
        #centroids = k.cluster_centers_
        reduced = pd.merge(reduced,data_pos[['ID']],how='left',left_index=True,right_index=True)
        """
        sns.set(style="white")
        ax= sns.lmplot( x="pc1", y="pc2", hue='cluster', data = reduced, legend=True,
         fontsize = 8, scatter_kws={"s": 100},aspect=1.5,height=4, palette='muted')
        ax=plt.gca()
        ax.set_title('PCA({}) Clustering Distribution - {}s // Two Principal Components'.format(pca_app,name.title()),size=12)
        """

        reduced_sampled = reduced.sample(frac=1).head(100)
        reduced_sampled = pd.merge(reduced_sampled, data_pos[['ID','Nombre']],how='left',on='ID')
        #print(reduced_sampled[['Nombre','cluster']].head(10))
        
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
        
        frames[position]= player_clustering
        fs[position]=pd.DataFrame(feature_scores,columns=[name])    
        

    return frames, fs, ks, pcas

#players,features,ks,pcas = player_clustering(position=list(players_df.PosE.unique()), data = players_df, cat_input = pos_dict)

#players2 = players[players['Min']>=min_minutes]

def get_player_features(feat, data):
    df,features,ks,pcas = player_clustering(list(data.PosE.unique()))
    for f in feat.keys():
        clu = f
        print('-----POSITION: {}-----'.format(clu))
    
        d = df[clu]
        pl = data.set_index('Nombre')
        ls=list(features[clu].index)
        ls.append('Nombre')
        d = pd.merge(d,data[ls],how='left',on='Nombre')
        f = features[clu][:8]
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
    y.drop(['ID','Nombre','Posición','Edad','Equipo','league-instat','Nacionalidad'],inplace=True,axis=1)
    pl = list(reduced.Nombre)
    pos = list(reduced['Posición'])
    eda = list(reduced.Edad)
    eq = list(reduced.Equipo)
    le = list(reduced['league-instat'])
    nac = list(reduced.Nacionalidad)
    reduced.drop(['ID','Nombre','Posición','Edad','Equipo','league-instat','Nacionalidad'],inplace=True,axis=1)
    euc = []
    for i in reduced.values:
        euc.append(euclidean_distances(np.array(y),[i])[0][0])
    simil = pd.DataFrame(euc,index=[pl,pos,eda,eq,le,nac],columns=['Similarity_Score'])
    simil = simil.reset_index()
    simil.columns = ['Nombre','Posición','Edad','Equipo','league-instat','Nacionalidad','Similarity_Score']
    #simil = simil[simil.Nombre.str.title()!=player_name.title()]
    if name2[0] == 'L':
        simil = simil[simil['Posición']==name2]
        
    simil = simil.sort_values(by='Similarity_Score',ascending=True)
    
    simil = simil[simil.Nombre!=p]
    #simil.drop('Posición',axis=1,inplace=True)



    return simil

        


def team_mapping(team,position, data_team, data_player, cats):
    klist = list(data_player[data_player['Posición']==position].PosE.unique())
    players,features,ks,pcas = player_clustering(position=klist, data = data_player, cat_input = cats)
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
    if position == 'DFC' or position[0]=='L' or position=='MCD':
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
    players = players[name]
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
    reduced = reduced.head(reduced.shape[0])
    reduced = pd.merge(reduced,data_pos[['Nombre','ID','Posición','teamid','Índice InStat','Values','Edad','Pierna']],how='left',left_index=True,right_index=True)
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
    reduced.drop(['Nombre','ID','Posición','teamid','Índice InStat','Values','Edad','Pierna'],inplace=True,axis=1)
    euc = []
    for i in reduced.values:
        euc.append(euclidean_distances(np.array(y),[i])[0][0])
    simil = pd.DataFrame(euc,index=[pl,idx,sq,ag,ft,vl,pw],columns=['Team_Similarity_Index'])
    
    simil = simil.sort_values(by='Team_Similarity_Index',ascending=True)

    simil = simil.reset_index()
    simil.columns = ['Nombre','ID','teamid','Edad','Pierna','Values','Power','Team_Similarity_Index']
    simil['Team_Similarity_Index'] = round(simil['Team_Similarity_Index'],3)

    simil = simil[simil['Team_Similarity_Index']!=0]
   
    return simil


