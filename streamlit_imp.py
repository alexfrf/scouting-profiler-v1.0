# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 02:51:34 2023

@author: aleex
"""

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from PIL import Image
from clustering_functions import isna_check
from clustering_functions import cbs,mid,attm,flb,fwd,pos_dict
from Players_plotting import sradar,radar_comp,plot_percentiles
import math
from scipy.stats import percentileofscore

scaler = MinMaxScaler()

np.random.seed(4)

image = Image.open(os.getcwd()+"/Documentacion/logomin.png")
#image_comp = Image.open(os.getcwd()+"/Documentacion/logo_comp.png")
sns.set(style="whitegrid")

ruta_datos = os.path.join(os.getcwd(),"Datos")

players_df = pd.read_csv(ruta_datos+'/Modeled/jugadores.csv',sep=';',decimal=',')
players_df.drop('Equipo',inplace=True,axis=1)
squad_df = pd.read_csv(ruta_datos+'/Modeled/equipos.csv',sep=';',decimal=',')
players_df = pd.merge(players_df,squad_df[['teamid','Equipo']],how='left',on='teamid')



#players_df = pd.merge(players_df,squad_df[['teamid','Equipo']],how='left',on='teamid')

columnas_e_pl = list(set(isna_check(players_df)))

for c in columnas_e_pl:
    players_df[c] = players_df[c].fillna(0)

st.set_page_config(layout="wide")
cola, colb, colc = st.columns([0.25,0.25,1.5]) 
cola.image(image)
#colb.image(image_comp)

st.title('Perfilación de Futbolistas mediante Clusterización, Métricas Avanzadas y Análisis de Modelos de Juego')
st.write("""
        Esta aplicación emplea un modelo analítico para segmentar las características de
        los jugadores en las competiciones de fútbol más importantes del mundo, 
        utilizando métricas avanzadas. Al considerar el modelo de juego de los equipos,
        la aplicación puede asociar a cada club los jugadores que mejor se adapten a su
        sistema para cada posición en el campo. El valor añadido principal se centra, pues,
        en agregar el estilo de juego de los equipos como factor clave, midiendo y
        cuantificándolo para identificar a los jugadores que se acerquen más las métricas
        objetivo del conjunto que se está analizando. Estos jugadores son los que potencialmente
        mejor se adaptarán a las necesidades del equipo al acudir al mercado, ya que requerirán 
        un proceso de adaptación más corto y conocerán mecanismos de juego similares, al provenir
        de equipos que muestran similitudes tácticas.

         """)

st.markdown("""
            * **Datos procedentes de [InStat](https://www.instatsport.com/en/) y  [Transfermarkt](https://www.transfermarkt.com/), correspondientes a la temporada 2021/22**.
            * **Desarrollo del Modelo: [Alex Fernández](https://alexfrf.github.io/)**
            * **Repositorio y Documentación del Proyecto en [Github](https://github.com/alexfrf/scouting-profiler)**
            """)

          
st.sidebar.header('Introducción de Filtros')



option = st.sidebar.selectbox(
        "Posición",
        ('DFC','LD','LI','MCD','MC','MCO','ED','EI','DC'))

pose = players_df[players_df['Posición']==option].PosE.values[0]
pose = pose.replace('/','-')
clu = pd.read_csv(ruta_datos+'/Modeled/{}_clustered.csv'.format(pose),decimal=',',sep=';')

sorted_unique_team = sorted(squad_df.Equipo.unique())
select_team = st.sidebar.selectbox(
        "Equipo",
        tuple(squad_df['Equipo'].unique()))

select_unt = st.sidebar.slider(
        "Q",5,200,100)

players_df['Values'] = players_df.Values.astype(int)
players_df['Edad'] = players_df.Edad.astype(int)

selected_value = st.sidebar.slider('Valor de Mercado', 0,int(players_df.Values.max()+1),10)
selected_age = st.sidebar.slider('Edad', int(players_df.Edad.min()),int(players_df.Edad.max()),30)
select_pierna = st.sidebar.multiselect(
        "Pierna Buena",
        list(players_df.Pierna.unique()[:3]),list(players_df.Pierna.unique()[:3]))
selected_instat = st.sidebar.slider('Performance Index', 0,100,20)

st.write("""
         ***
         """)
         
st.markdown("""
              #### Mostrando los {} jugadores más adecuados a la posición de {} en {}\n
              """.format(select_unt,option,select_team))

col2, col3 = st.columns([1,1])  

@st.cache_resource
def get_features(position):
    pose = players_df[players_df['Posición']==position].PosE.values[0]
    pose = pose.replace('/','-')
    features = pd.read_csv(ruta_datos+'/Modeled/{}_features.csv'.format(pose),
                           decimal=',',sep=';')
    
    return features

@st.cache_resource
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
    
    simil['Similarity_Score'] = simil['Similarity_Score'].apply(lambda x: 100-percentileofscore(simil['Similarity_Score'],x, kind='strict'))
    simil['Similarity_Score'] = simil['Similarity_Score'].apply(lambda x:round(x,2))
    simil = simil.sort_values(by='Similarity_Score',ascending=False)
    simil = pd.merge(simil,data[['ID','Nombre']],how='left',on='ID')
    simil = simil[simil.Nombre!=p]
    simil = simil.set_index('Nombre')
    simil.drop('ID',axis=1,inplace=True)



    return simil    


@st.cache_resource
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
    reduced = pd.merge(reduced,data_pos[['Nombre','ID','Posición','teamid','Performance','Values','Edad','Pierna','Equipo','Nacionalidad','cluster']],how='left',left_index=True,right_index=True)
    #reduced['idx'] = reduced['idx'].fillna('{}-{}'.format(team,position))
    #reduced['Player'] = reduced['Player'].fillna('{}-{}'.format(team,position))
    if position[0]=='E':
        reduced = reduced[reduced['Posición'].str.startswith('E')]
    else:
        reduced = reduced[reduced['Posición']==position]
    pl = list(reduced.Nombre)
    idx = list(reduced.ID)
    sq = list(reduced.teamid)
    pw = list(reduced['Performance'])
    vl=list(reduced.Values)
    ag=list(reduced.Edad)
    ft=list(reduced.Pierna)
    cl = list(reduced.cluster)
    eq = list(reduced.Equipo)
    nc = list(reduced.Nacionalidad)
    reduced.drop(['Nombre','ID','Posición','teamid','Performance','Values','Edad','Pierna','Equipo','Nacionalidad','cluster'],inplace=True,axis=1)
    euc = []
    for i in reduced.values:
        euc.append(euclidean_distances(np.array(y),[i])[0][0])
    simil = pd.DataFrame(euc,index=[pl,idx,sq,ag,ft,vl,pw,eq,nc,cl],columns=['Team_Similarity_Index'])
    
    simil = simil.sort_values(by='Team_Similarity_Index',ascending=True)

    simil = simil.reset_index()
    simil.columns = ['Nombre','ID','teamid','Edad','Pierna','Values','Performance','Equipo','Nacionalidad','cluster','Team_Similarity_Index']
    
    simil['Team_Similarity_Index'] = ((simil.Team_Similarity_Index - simil.Team_Similarity_Index.mean()) / simil.Team_Similarity_Index.std())
    #simil = simil[simil['Team_Similarity_Index']!=0]
    simil['Team_Similarity_Index'] = simil['Team_Similarity_Index'] + math.ceil(abs(simil.Team_Similarity_Index.min()))
    simil['Team_Similarity_Index'] = round(simil['Team_Similarity_Index'],3)
    maxs = simil['Team_Similarity_Index'].values[0]
    #simil['Team_Similarity_Index'] = simil['Team_Similarity_Index'].apply(lambda x: 100-maxs-percentileofscore(simil['Team_Similarity_Index'],x, kind='strict'))
    #simil['Team_Similarity_Index'] = simil['Team_Similarity_Index'].apply(lambda x:round(x,2))
    
    simil['Values'] = round(simil.Values,0).astype(int)
    simil['Edad'] = simil.Edad.astype(int)
    #simil['Índice InStat'] = simil['Índice InStat'].astype(int)
    simil['Nacionalidad'] = simil.Nacionalidad.apply(lambda x: x.split(',')[0].strip())
    
    s = pd.DataFrame(simil['Team_Similarity_Index'].describe()).T
    s = s[['count', 'mean','25%', '50%', '75%']]
    s.rename({'count':'N','mean':'Media','25%':'PCT25','50%':'PCT50','75%':'PCT75'},axis=1,
             inplace=True)
    #simil = simil[(simil.Values<=value) & (simil.Edad<=age)]
    
    
    return simil,s

df = team_mapping(select_team,option, squad_df, players_df, pos_dict)[0]
describe = team_mapping(select_team,option, squad_df, players_df, pos_dict)[1]

sorted_unique_lg = sorted(squad_df['league-instat'].unique())
select_league = st.sidebar.multiselect('Competición',sorted_unique_lg,sorted_unique_lg)
sorted_unique_team = sorted(squad_df.Equipo.unique())

teams = []
for i in select_league:
    for t in list(squad_df[squad_df['league-instat']==i].Equipo.unique()):
        teams.append(t)
select_team_l = st.sidebar.multiselect('Equipo',teams,teams)

df = df[(df.Equipo.isin(select_team_l))]
df = df[(df.Edad<=selected_age) & (df.Values<=selected_value) & (df['Performance']>=selected_instat) & (df.Pierna.isin(select_pierna))]

cols = ['Nombre','Equipo','Nacionalidad','Edad','Values','Team_Similarity_Index','Performance','cluster']

plf = players_df[(players_df['Posición']==option)]
#kdf1 = plf[plf.ID==df.head(1).ID.values[0]]
#kdf = plf[plf.ID!=df.head(1).ID.values[0]]
#kdf = pd.concat([kdf1,kdf])
#kdf = pd.merge(kdf,df[['ID','Team_Similarity_Index']],how='left',on='ID')
plf = pd.merge(plf,clu[['cluster','Nombre']],how='left',on='Nombre')
#kdf = kdf.sort_values(by='Team_Similarity_Index',ascending=True)
df = df[cols]
df.rename({'Values':'Valor',
           'Team_Similarity_Index':'Similarity',
           'cluster':'Cluster_Jugador'},
          inplace=True,axis=1)

df = df.set_index('Nombre')
tab = df[['Similarity']].copy()
tab['Ranking'] = tab['Similarity'].rank()
tab['Ranking'] = tab['Ranking'].apply(lambda x: "{:.0f}º".format(x))





col2.dataframe(df.head(select_unt))


feat = get_features(option)
fig,ax = plt.subplots(figsize=(16, 12))

ax.barh(feat['feat'],feat['influence'])
ax.invert_yaxis()
sns.despine()
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=25)
for item in ax.yaxis.get_ticklabels():
    item.set_fontsize(22)
    
for item in ax.xaxis.get_ticklabels():
    item.set_fontsize(14)
ax.set_title('Variables más Influyentes en el Modelo para Segmentar {}'.format(option), size= 20, weight='bold')

col3.pyplot(fig)

describe.rename({'Team_Similarity_Index':'Similarity'},axis=0,inplace=True)
col2.dataframe(describe)



st.write('***')

col4, col5= st.columns([1,1]) 
col4.markdown("""#### Dashboard""")

st.set_option('deprecation.showPyplotGlobalUse', False)
col6, col7= st.columns([1,1]) 

select_pl = col6.selectbox(
        "Mostrar Datos de:",
        tuple(df.index.unique()))

#col5.dataframe(tab[tab.index==select_pl][['Similarity','Ranking']])





plf_team = plf[plf.Equipo==select_team]

plf = plf[~plf.index.isin(plf_team.index)]
plf = pd.concat([plf_team,plf])

tab['Rank'] = np.where(tab.index.isin(list(tab.head(select_unt).index.values)),"Top{:.0f}".format(select_unt),'Resto')
col6.pyplot(sradar(select_pl))

fig,ax=plt.subplots(figsize=(12,2.5))
sns.histplot(data=tab, x="Similarity", hue="Rank", multiple="stack",kde=True,palette='seismic')
ax.scatter(x=tab[tab.index==select_pl].Similarity.values[0],y=8,s=300,marker='X',color='purple')
ax.axvline(tab.Similarity.mean(),0,color='blue')
ax.axvline(tab.Similarity.quantile(.25),0,color='blue')
ax.annotate(tab[tab.index==select_pl].index[0]+' ({})'.format(tab[tab.index==select_pl].Ranking.values[0]),(tab[tab.index==select_pl].Similarity.values[0],20),ha='center',weight='bold',size=16)
ax.annotate('Pct25',(tab.Similarity.quantile(.25),100),ha='right',size=10,weight='bold')
ax.annotate('Media',(tab.Similarity.mean(),100),ha='right',size=10,weight='bold')
#ax.set_title('Distribución de Similitud', size= 18, weight='bold')
#ax.set_xlabel(None)
#ax.set_ylabel(None)
sns.despine()
col7.pyplot(fig)
col7.pyplot(plot_percentiles(select_pl,select_team_l))

col8,col9,col10 = st.columns([0.5,0.5,1]) 

select_pl1 = col8.selectbox(
        "Mostrar Radar Comparativo de:",
        tuple(df.index.unique()))
select_pl2 = col9.selectbox(
        "y:",
        tuple(plf.Nombre.unique()))

plf.rename({'cluster':'Cluster_Jugador'},axis=1,inplace=True)
plf = plf.set_index('Nombre')

col10,col11 = st.columns([1,1]) 
col10.pyplot(radar_comp(select_pl1,select_pl2))

ps = player_similarities(select_pl, players_df)
#ps = ps.set_index('Nombre')
col11.dataframe(plf[plf.index.isin([select_pl1,select_pl2])][['Performance','Cluster_Jugador']])

col11.markdown('**Jugadores similares a {}**'.format(select_pl))

col11.dataframe(ps.head(50))

gl= squad_df
gl = gl.sort_values(by='Puntos esperados',ascending=False)
gl = gl.set_index('Equipo')
for i in gl.columns:
    if 'cluster' in i:
        gl[i] = pd.to_numeric(gl[i])
st.write('***')
col12,col13 = st.columns([1,1]) 
col12.markdown("""#### Clusters de Modelos de Juego de Equipos Explicados, por categorías""")

col12.markdown("""
**Disposición Táctica**
- **C1**: Cuatro defensas, doble pivote, un delantero - 4-2-3-1
- **C2**: Cuatro defensas, dos delanteros - 4-3-1-2, 4-4-2
- **C3**: Tres centrales, no juegan con un delantero - 3-4-3, 3-5-2
- **C4**: Cuatro defensas, tres delanteros - 4-3-3
               """)

col12.markdown("""*{} se encuentra en el Cluster {}*""".format(select_team,
                                                               squad_df[squad_df['Equipo']==select_team]['disposicion_tactica_cluster'].values[0]))
means = []
for i in range(1,1+gl['disposicion_tactica_cluster'].max()):
    means.append(gl[gl['disposicion_tactica_cluster']==i]['Puntos esperados'].mean())
    
media = np.mean(means)
select_clu1= col12.selectbox(
        "Mostrar Equipos de Cluster Táctico:",
        tuple([i for i in range(1,5)]))               
col12.dataframe(gl[gl['disposicion_tactica_cluster']==select_clu1][['last_coach','league-instat','Puntos esperados']])
col12.metric("Media de Puntos Esperados para el Cluster Seleccionado",
             "{:.2f}".format(gl[gl['disposicion_tactica_cluster']==select_clu1]['Puntos esperados'].mean()),
             "{:.2f}".format(gl[gl['disposicion_tactica_cluster']==select_clu1]['Puntos esperados'].mean() - media))


col12.write('***')
col12.markdown("""
**Defensa**
- **C1**: Menor tendencia a presión, más repliegue bajo y más tiempo defendiendo.
- **C2**: Más intensidad defensiva en en zonas intermedias.
- **C3**: Variación de alturas y eficiencia cuando deciden defender alto.
- **C4**: Más tendencia a presionar, defensa alta y agresividad en campo contrario.
""")

col12.markdown("""*{} se encuentra en el Cluster {}*""".format(select_team,
                                                               squad_df[squad_df['Equipo']==select_team]['defensa_cluster'].values[0]))

means = []
for i in range(1,1+gl['defensa_cluster'].max()):
    means.append(gl[gl['defensa_cluster']==i]['Puntos esperados'].mean())
    
media = np.mean(means)
select_clu2= col12.selectbox(
        "Mostrar Equipos de Cluster Defensivo:",
        tuple([i for i in range(1,5)]))               
col12.dataframe(gl[gl['defensa_cluster']==select_clu2][['last_coach','league-instat','Puntos esperados']])
col12.metric("Media de Puntos Esperados para el Cluster Seleccionado",
             "{:.2f}".format(gl[gl['defensa_cluster']==select_clu2]['Puntos esperados'].mean()),
             "{:.2f}".format(gl[gl['defensa_cluster']==select_clu2]['Puntos esperados'].mean() - media))

col12.write('***')
col12.markdown("""
**Buildup**
- **C1**: Posesiones largas, bajo volumen de juego en tercio final, transiciones largas y desde lejos, poca incidencia al contragolpe.
- **C2**: Progresión hacia campo rival más directa, tendencia alta al contragolpe, ritmo alto en construcción y llegada a área con pocos pases.
- **C3**: Salida de balón más elaborada, ritmo bajo en construcción y posesiones largas, poco recurso al contragolpe.
- **C4**: Transiciones cortas en campo contrario, salida de balón relativamente elaborada, alto volumen de juego en tercio final, ritmo muy elevado y verticalidad.
""")

col12.markdown("""*{} se encuentra en el Cluster {}*""".format(select_team,
                                                               squad_df[squad_df['Equipo']==select_team]['buildup_cluster'].values[0]))

means = []
for i in range(1,1+gl['buildup_cluster'].max()):
    means.append(gl[gl['buildup_cluster']==i]['Puntos esperados'].mean())
    
media = np.mean(means)
select_clu3= col12.selectbox(
        "Mostrar Equipos de Cluster de Buildup:",
        tuple([i for i in range(1,5)]))              
col12.dataframe(gl[gl['buildup_cluster']==select_clu3][['last_coach','league-instat','Puntos esperados']])
col12.metric("Media de Puntos Esperados para el Cluster Seleccionado",
             "{:.2f}".format(gl[gl['buildup_cluster']==select_clu3]['Puntos esperados'].mean()),
             "{:.2f}".format(gl[gl['buildup_cluster']==select_clu3]['Puntos esperados'].mean() - media))

col12.write('***')
col12.markdown("""
**Creación de Oportunidades y Finalización**
- **C1**: Bajo índice de transformación (tiro u ocasión) al llegar al área rival, reducida asociación para generar -mayor dependencia de acción individual-.
- **C2**: Poca capacidad de rentabilizar sus posesiones pero relativamente alto número de llegadas y de posibilidades de pase al área/centro, bajo nivel de asociación en metros finales.
- **C3**: Alto índice de transformación (tiro u ocasión) al llegar al área rival, calidad individual para encontrar una buena oportunidad, mayor asociación entre jugadores de ataque.
- **C4**: Buen ratio de transformación por cada ataque, menor asociación entre atacantes para encontrar la oportunidad.
""")

col12.markdown("""*{} se encuentra en el Cluster {}*""".format(select_team,
                                                               squad_df[squad_df['Equipo']==select_team]['creacion_oportunidades_cluster'].values[0]))

means = []
for i in range(1,1+gl['creacion_oportunidades_cluster'].max()):
    means.append(gl[gl['creacion_oportunidades_cluster']==i]['Puntos esperados'].mean())
    
media = np.mean(means)
select_clu4= col12.selectbox(
        "Mostrar Equipos de Cluster de Creación de Oportunidades:",
        tuple([i for i in range(1,5)]))              
col12.dataframe(gl[gl['creacion_oportunidades_cluster']==select_clu4][['last_coach','league-instat','Puntos esperados']])
col12.metric("Media de Puntos Esperados para el Cluster Seleccionado",
             "{:.2f}".format(gl[gl['buildup_cluster']==select_clu4]['Puntos esperados'].mean()),
             "{:.2f}".format(gl[gl['buildup_cluster']==select_clu4]['Puntos esperados'].mean() - media))



players_df = players_df.sort_values(by='Performance', ascending=False)
players_clus = pd.merge(players_df,clu,how='left',on='Nombre')   
players_clus = players_clus[players_clus.cluster.isna()!=True] 
players_clus = players_clus.set_index('Nombre')
players_clus.rename({'Performance':'IDX',
                     'Values':'Valor'},axis=1,inplace=True)
col13.write("""#### Clusters de {} Explicados
            """.format(pose))
            
dict_explicacion = {'Centre-Back':"""
                                    - **C1**: Dominio de juego aéreo, más disputas defensivas, mayor capacidad de imponerse individualmente, menor participación con balón.
                                    - **C2**: Menor propensión a duelos y a entradas, menos faltas, mayor participación con balón.
                                    - **C3**: Mayor tendencia a realizar entradas y a salir fuera de zona -mayor índice de comisión de faltas-, menor propensión a medirse por alto.
                                    """,
           'Midfielder':"""
                           - **C1**: Pivote, alto volumen de tareas defensivas, poca participación en tercio final, alta tasa de disputas y recepciones en la base de la jugada.
                           - **C2**: Perfil más ofensivo, menos acciones defensivas, mayor participación en creación de oportunidades, caída a banda y recepciones cercanas al área.  
                           - **C3**: Perfil mixto y de amplio recorrido, con y sin balón. Organizadores itinerantes en equipos que aparecen para construir en distintas alturas y centrocampistas con alta capacidad de llegada a área para rematar.
                           """,
           'Att. Midfield-Winger':"""
                           - **C1**: Jugadores autosuficientes, siempre por el centro o en banda a pierna cambiada. Alta incidencia en area (valores altos de último pase y oportunidades propias disfrutadas).
                           - **C2**: Extremo a pie natural, encarador y regateador. Alta tasa de centros en línea de fondo. Se incluyen carrileros que juegan muy alto.
                           - **C3**: Extremos inversos o mediapuntas con alto valor generado a través del pase y mucha participación en construcción. Algunos, acostumbrados a jugar en la posición de 10 o como interiores en 4-3-3.
                           """,
           'Full-Back':"""           
                           - **C1**: Lateral a pie cambiado, incorporación al ataque hacia zonas interiores.
                           - **C2**: Profundidad y centros, alta participación directa en la generación de ocasiones y menor participación en la construcción.
                           - **C3**: Organizador desde la banda, menor intensidad defensiva pero en altura no elevada -no gana línea de fondo en ataque-. Mucha participación en salida y elaboración.
                           - **C4**: Perfil conservador. Menor tasa de centros, mayor volumen de actividad defensiva, poca llegada a zonas de último pase.
                           """,
           'Forward':"""
                           - **C1**: Mayor influencia en la generación de oportunidades, propias y para sus compañeros. Nueve puro, móvil y autosuficiente.                
                           - **C2**: Más disputas aéreas y dominio del juego por alto, estático pero con participación en el juego -recibiendo de espaldas, ganando duelos que propician segunda jugada-.
                           - **C3**: Perfil de área, regateador en espacios cortos, menor movimiento sin balón, menor participación en posesión, rematador pero no dominador por alto. 
                           - **C4**: Más esfuerzo defensivo y participación en la elaboración, perfil segundo delantero. Cae a banda, regatea y genera para los demás. 
                           """}
            
col13.write("""
               {}
               """.format(dict_explicacion[pose]))
               

select_clu_pl= col13.selectbox(
        "Mostrar Jugadores para el cluster de {}:".format(pose),
        tuple([i for i in range(1,int(players_clus.cluster.max()+1))]))              
col13.dataframe(players_clus[(players_clus['cluster']==select_clu_pl) & (players_clus.Edad<=selected_age) & (players_clus.Valor<=selected_value) & (players_clus['IDX']>=selected_instat) & (players_clus.Pierna.isin(select_pierna)) & (players_clus.Equipo.isin(select_team_l))][['Equipo','Posición','Nacionalidad','Pierna','Edad','Valor','IDX']])