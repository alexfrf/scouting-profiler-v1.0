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
from sklearn.preprocessing import MinMaxScaler
from Players_plotting import sradar,radar_comp,plot_percentiles
from PIL import Image
from clustering_functions import isna_check


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



#players_df = pd.merge(players_df,squad_df[['teamid','Equipo']],how='left',on='teamid')

columnas_e_pl = list(set(isna_check(players_df)))

for c in columnas_e_pl:
    players_df[c] = players_df[c].fillna(0)

st.set_page_config(layout="wide")
cola, colb, colc = st.columns([0.25,0.25,1.5]) 
cola.image(image)
#colb.image(image_comp)

st.title('Perfilación de Futbolistas en base a Métricas Avanzadas y Modelo de Juego')
st.write("""
        Esta aplicación ilustra un modelo que busca segmentar las
características de los jugadores de las competiciones más importantes del fútbol
mundial según métricas avanzadas y,
segmentando el modelo de juego de los equipos, poder asociar, a cada club, los futbolistas que,
para cada posición en el campo, mejor encajen en su sistema. El propósito es añadir
la variante del estilo de juego de los equipos -el propio y los demás-, midiéndolo y
cuantificándolo para poder encontrar a aquellos jugadores que más se aproximen, basándonos
en métricas avanzadas, a los números del equipo que estemos analizando. Serán esos futbolistas
los que, potencialmente, más se adecúen a las necesidades por las que dicho equipo acude al
mercado, pues serán piezas que requerirán un proceso de adaptación más corto y conocerán
mecanismos similares de juego, pues vendrán de conjuntos que, tácticamente, muestran
simitudes.

         """)

st.markdown("""
            * **Datos procedentes de [InStat](https://www.instatsport.com/en/) y  [Transfermarkt](https://www.transfermarkt.com/), correspondientes a la temporada 2021/22**.
            * **Desarrollo del Modelo: [Alex Fernández](https://alexfrf.github.io/)**
            """)

          
st.sidebar.header('Introducción de Filtros')



option = st.sidebar.selectbox(
        "Posición",
        ('DFC','LD','LI','MCD','MC','MCO','ED','EI','DC'))

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
selected_instat = st.sidebar.slider('Índice InStat', int(players_df['Índice InStat'].min()),int(players_df['Índice InStat'].max()),int(players_df['Índice InStat'].min()))

st.write("""
         ***
         """)
         
st.markdown("""
              #### Mostrando los {} jugadores más adecuados a la posición de {} en {}\n
              """.format(select_unt,option,select_team))

col2, col3 = st.columns([1,1])  

@st.cache
def get_features(position):
    pose = players_df[players_df['Posición']==position].PosE.values[0]
    pose = pose.replace('/','-')
    features = pd.read_csv(ruta_datos+'/Modeled/{}_features.csv'.format(pose),
                           decimal=',',sep=';')
    
    return features

@st.cache
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
    idx = list(reduced.ID)
    reduced.drop(['ID','Nombre','Posición','Edad','Equipo','league-instat','Nacionalidad'],inplace=True,axis=1)
    euc = []
    for i in reduced.values:
        euc.append(euclidean_distances(np.array(y),[i])[0][0])
    simil = pd.DataFrame(euc,index=[idx,pos,eda,eq,le,nac],columns=['Similarity_Score'])
    simil = simil.reset_index()
    simil.columns = ['ID','Posición','Edad','Equipo','league-instat','Nacionalidad','Similarity_Score']
    #simil = simil[simil.Nombre.str.title()!=player_name.title()]
    if name2[0] == 'L':
        simil = simil[simil['Posición']==name2]
        
    simil = simil.sort_values(by='Similarity_Score',ascending=True)
    simil = pd.merge(simil,data[['ID','Nombre']],how='left',on='ID')
    simil = simil[simil.Nombre!=p]
    simil = simil.set_index('Nombre')
    simil.drop('ID',axis=1,inplace=True)



    return simil


@st.cache
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
    #data_pos = pd.merge(data_pos,players,how='left',on='Nombre')
    reduced = pd.merge(reduced,data_pos[['Nombre','ID','Posición','teamid','Índice InStat','Values','Edad','Pierna','Equipo','Nacionalidad']],how='left',left_index=True,right_index=True)
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
    eq = list(reduced.Equipo)
    nc = list(reduced.Nacionalidad)
    reduced.drop(['Nombre','ID','Posición','teamid','Índice InStat','Values','Edad','Pierna','Equipo','Nacionalidad'],inplace=True,axis=1)
    euc = []
    for i in reduced.values:
        euc.append(euclidean_distances(np.array(y),[i])[0][0])
    simil = pd.DataFrame(euc,index=[pl,idx,sq,ag,ft,vl,pw,eq,nc],columns=['Team_Similarity_Index'])
    
    simil = simil.sort_values(by='Team_Similarity_Index',ascending=True)

    simil = simil.reset_index()
    simil.columns = ['Nombre','ID','teamid','Edad','Pierna','Values','Índice InStat','Equipo','Nacionalidad','Team_Similarity_Index']
    simil['Team_Similarity_Index'] = round(simil['Team_Similarity_Index'],3)

    #simil = simil[simil['Team_Similarity_Index']!=0]
    simil['Values'] = round(simil.Values,0).astype(int)
    simil['Edad'] = simil.Edad.astype(int)
    simil['Índice InStat'] = simil['Índice InStat'].astype(int)
    simil['Nacionalidad'] = simil.Nacionalidad.apply(lambda x: x.split(',')[0].strip())
    #simil = simil[(simil.Values<=value) & (simil.Edad<=age)]
    
    
    return simil

df = team_mapping(select_team,option, squad_df, players_df, pos_dict)

sorted_unique_lg = sorted(squad_df['league-instat'].unique())
select_league = st.sidebar.multiselect('Competición',sorted_unique_lg,sorted_unique_lg)
sorted_unique_team = sorted(squad_df.Equipo.unique())

teams = []
for i in select_league:
    for t in list(squad_df[squad_df['league-instat']==i].Equipo.unique()):
        teams.append(t)
select_team_l = st.sidebar.multiselect('Equipo',teams,teams)

df = df[(df.Equipo.isin(select_team_l))]
df = df[(df.Edad<=selected_age) & (df.Values<=selected_value) & (df['Índice InStat']>=selected_instat) & (df.Pierna.isin(select_pierna))]

cols = ['Nombre','Equipo','Nacionalidad','Edad','Values','Team_Similarity_Index','Índice InStat']
df = df[cols]
df.rename({'Índice InStat':'IDX',
           'Values':'Valor',
           'Team_Similarity_Index':'Similarity'},
          inplace=True,axis=1)
df = df.set_index('Nombre')
df = df.head(select_unt)
col2.dataframe(df)


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


st.write('***')

st.set_option('deprecation.showPyplotGlobalUse', False)
col4,col5 = st.columns([1,1])  
col4.markdown("""#### Dashboard""")
select_pl = col4.selectbox(
        "Mostrar Datos de:",
        tuple(df.index.unique()))




plf = players_df[(players_df['Posición']==option)]
plf_team = plf[plf.Equipo==select_team]

plf = plf[~plf.index.isin(plf_team.index)]
plf = pd.concat([plf_team,plf])

col6, col7= st.columns([1,1])
col6.pyplot(sradar(select_pl))
col7.pyplot(plot_percentiles(select_pl,select_team_l))

col8,col9,col10 = st.columns([0.5,0.5,1]) 

select_pl1 = col8.selectbox(
        "Mostrar Radar Comparativo de:",
        tuple(df.index.unique()))
select_pl2 = col9.selectbox(
        "y:",
        tuple(plf.Nombre.unique()))


col10,col11 = st.columns([1,1]) 
col10.pyplot(radar_comp(select_pl1,select_pl2))

ps = player_similarities(select_pl, players_df)
#ps = ps.set_index('Nombre')
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

select_clu1= col12.selectbox(
        "Mostrar Equipos de Cluster Táctico:",
        tuple([i for i in range(1,5)]))               
col12.dataframe(gl[gl['disposicion_tactica_cluster']==select_clu1][['last_coach','league-instat','Puntos esperados']])

col12.write('***')
col12.markdown("""
**Defensa**
- **C1**: Menor tendencia a presión, más repliegue bajo y más tiempo defendiendo.
- **C2**: Más intensidad defensiva en en zonas intermedias.
- **C3**: Variación de alturas y eficiencia cuando deciden defender alto.
- **C4**: Más tendencia a presionar, defensa alta y agresividad en campo contrario.
""")

select_clu2= col12.selectbox(
        "Mostrar Equipos de Cluster Defensivo:",
        tuple([i for i in range(1,5)]))               
col12.dataframe(gl[gl['defensa_cluster']==select_clu2][['last_coach','league-instat','Puntos esperados']])
col12.write('***')
col12.markdown("""
**Buildup**
- **C1**: Posesiones largas, bajo volumen de juego en tercio final, transiciones largas y desde lejos, poca incidencia al contragolpe.
- **C2**: Progresión hacia campo rival más directa, tendencia alta al contragolpe, ritmo alto en construcción y llegada a área con pocos pases.
- **C3**: Salida de balón más elaborada, ritmo bajo en construcción y posesiones largas, poco recurso al contragolpe.
- **C4**: Transiciones cortas en campo contrario, salida de balón relativamente elaborada, alto volumen de juego en tercio final, ritmo muy elevado y verticalidad.
""")

select_clu3= col12.selectbox(
        "Mostrar Equipos de Cluster de Buildup:",
        tuple([i for i in range(1,5)]))              
col12.dataframe(gl[gl['buildup_cluster']==select_clu3][['last_coach','league-instat','Puntos esperados']])

col12.write('***')
col12.markdown("""
**Creación de Oportunidades y Finalización**
- **C1**: Bajo índice de transformación (tiro u ocasión) al llegar al área rival, reducida asociación para generar -mayor dependencia de acción individual-.
- **C2**: Poca capacidad de rentabilizar sus posesiones pero relativamente alto número de llegadas y de posibilidades de pase al área/centro, bajo nivel de asociación en metros finales.
- **C3**: Alto índice de transformación (tiro u ocasión) al llegar al área rival, calidad individual para encontrar una buena oportunidad, mayor asociación entre jugadores de ataque.
- **C4**: Buen ratio de transformación por cada ataque, menor asociación entre atacantes para encontrar la oportunidad.
""")

select_clu4= col12.selectbox(
        "Mostrar Equipos de Cluster de Creación de Oportunidades:",
        tuple([i for i in range(1,5)]))              
col12.dataframe(gl[gl['creacion_oportunidades_cluster']==select_clu4][['last_coach','league-instat','Puntos esperados']])


pose = players_df[players_df['Posición']==option].PosE.values[0]
pose = pose.replace('/','-')
clu = pd.read_csv(ruta_datos+'/Modeled/{}_clustered.csv'.format(pose),decimal=',',sep=';')

players_df = players_df.sort_values(by='Índice InStat', ascending=False)
players_clus = pd.merge(players_df,clu,how='left',on='Nombre')   
players_clus = players_clus[players_clus.cluster.isna()!=True] 
players_clus = players_clus.set_index('Nombre')
players_clus.rename({'Índice InStat':'IDX',
                     'Values':'Valor'},axis=1,inplace=True)
col13.write("""#### Clusters de {} Explicados
            """.format(pose))
            
dict_explicacion = {'Centre-Back':"""
                                    - **C1**: Mayor tendencia a realizar entradas y a salir fuera de zona -mayor índice de comisión de faltas-, menor propensión a medirse por alto.
                                    - **C2**: Dominio de juego aéreo, más disputas defensivas, mayor capacidad de imponerse individualmente, menor participación con balón.
                                    - **C3**: Menor propensión a duelos y a entradas, menos faltas, mayor participación con balón.
                                  """,
           'Midfielder':"""
                           - **C1**: Perfil mixto, jugador de base de la jugada o llegador.
                           - **C2**: Pivote, alto volumen de tareas defensivas, poca participación en tercio final, alta tasa de disputas.
                           - **C3**: Perfil más ofensivo, menos acciones defensivas, mayor participación en creación de oportunidades, caída a banda.
                           """,
           'Att. Midfield-Winger':"""
                           - **C1**: Jugadores autosuficientes, siempre por el centro o en banda a pierna cambiada. Alta incidencia en area (valores altos de último pase y oportunidades propias disfrutadas).
                           - **C2**: Extremo a pie natural, encarador y regateador. Alta tasa de centros en línea de fondo. Se incluyen carrileros que juegan muy alto.
                           - **C3**: Extremos inversos o mediapuntas con alto valor generado a través del pase. Algunos, acostumbrados a jugar en la posición de 10 o como interiores en 4-3-3.
                           """,
           'Full-Back':"""
                           - **C1**: Perfil conservador. Menor tasa de centros, mayor volumen de actividad defensiva, poca llegada a zonas de último pase.            
                           - **C2**: Lateral a pie cambiado, incorporación al ataque hacia zonas interiores.
                           - **C3**: Organizador desde la banda, menor intensidad defensiva pero en altura no elevada -no gana línea de fondo en ataque-. Mucha participación en salida y elaboración.
                           - **C4**: Profundidad y centros, alta participación directa en la generación de ocasiones y menor participación en la construcción.
                           """,
           'Forward':"""
                           - **C1**: Más esfuerzo defensivo y participación en la elaboración, perfil segundo delantero. Cae a banda, regatea y genera para los demás.
                           - **C2**: Más disputas aéreas y dominio del juego por alto, menor participación en la generación de ocasiones.
                           - **C3**: Mayor influencia en la generación de oportunidades, propias y para sus compañeros. Nueve puro, móvil y autosuficiente.
                           """}
            
col13.write("""
               {}
               """.format(dict_explicacion[pose]))
               

select_clu_pl= col13.selectbox(
        "Mostrar Jugadores para el cluster de {}:".format(pose),
        tuple([i for i in range(1,int(players_clus.cluster.max()+1))]))              
col13.dataframe(players_clus[(players_clus['cluster']==select_clu_pl) & (players_clus.Edad<=selected_age) & (players_clus.Valor<=selected_value) & (players_clus['IDX']>=selected_instat) & (players_clus.Pierna.isin(select_pierna)) & (players_clus.Equipo.isin(select_team_l))][['Equipo','Posición','Nacionalidad','Pierna','Edad','Valor','IDX']])