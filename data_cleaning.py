# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 19:09:52 2023

@author: aleex
"""

import pandas as pd
import numpy as np
import utils_foot
import glob
import os
from pathlib import Path
from fuzzywuzzy import process, fuzz

temporada='2021-2022'

#for i in [2017,2018,2019,2020]:
    #a=i
seas = temporada[:4]
ruta_base = os.path.join('',Path(os.getcwd()))
ruta_datos = os.path.join(ruta_base,'Datos/{}'.format(seas))

df_equipos = pd.DataFrame()
df_jug = pd.DataFrame()


for f in os.listdir(ruta_datos):
    file = os.path.join(ruta_datos,f)
    if '.xls' in f and 'agregado' not in f and '.depr' not in file:
        df = pd.read_excel(file)
        if 'teams' in f:
            df.rename({'Equipos':'Equipo'},axis=1,inplace=True)
            df['Equipo'] = df['Equipo'].str.replace('amp;','')
            df = df[df.Equipo!='Average per team']
            df = utils_foot.clean_df(df)
            ids = f.split('_')[-1]
            ids = int(ids.replace('.xlsx',''))
            df['league'] = ids
            df_equipos = pd.concat([df_equipos,df])
        else:
            df.rename({'Unnamed: 0':'Num','Unnamed: 1':'Nombre'},axis=1,inplace=True)
            df = utils_foot.clean_df(df)
            df = df[(df['Partidos jugados']>=df['Partidos jugados'].quantile(0.1))]
            #df['Nombre'] = df.Nombre.str.replace('0','-')
            ids = f.split('_')[-1]
            ids = ids.replace('.xlsx','')
            df['league'] = ids
            df_jug = pd.concat([df_jug,df])
            

       
base = pd.read_csv(ruta_base+'/tmarkt_scraping/leagues_{}.csv'.format(temporada),
                   sep=';',decimal=',')

leagues = list(base['league-id-instat'].unique())
df_equipos['tmarkt_name'] = df_equipos.Equipo

for l in leagues:
    newlist_ud=[]
    instat_tm_transform={}
    equipos_tmarkt = list(base[base['league-id-instat']==l].team.unique())
    equipos_instat = list(df_equipos[df_equipos['league']==l].Equipo.unique())
    for i in equipos_tmarkt:
        x=process.extract(i, equipos_instat, scorer=fuzz.token_set_ratio)[0]
        if x[1]>=0:
            instat_tm_transform[x[0]]=i
            newlist_ud.append(x[0])
            
    
    df_equipos['tmarkt_name'].replace(instat_tm_transform,inplace=True)

df_equipos['comprobacion'] = np.where(df_equipos['Equipo']==df_equipos['tmarkt_name'],0,1)    


df_equipos = pd.merge(df_equipos,base[['team','league-id-instat','teamid']],
                      left_on=['tmarkt_name','league'],right_on=['team','league-id-instat'],how='left')

df_equipos.drop(['team','league-id-instat'],inplace=True,axis=1)
df_equipos = df_equipos[df_equipos.teamid.isna()==False]
#df_equipos.to_excel(ruta_datos+'/agregado.xlsx',index=False)

base_u = base.drop_duplicates(subset=['league-instat','league-id-instat','team'])
tmarkt = pd.read_csv(ruta_base+'/tmarkt_scraping/tmarkt_{}.csv'.format(temporada),
                   sep=';',decimal=',')

tmarkt = pd.merge(tmarkt,base[['team','league-id-instat','teamid']],
                  how='inner',right_on='team',left_on='Team')
tmarkt = tmarkt.drop_duplicates()
tmarkt.drop('team',inplace=True,axis=1)
tmarkt = tmarkt[tmarkt.teamid.isna()==False]

tmarkt['Num'] = tmarkt['Num'].str.replace('-',"0")
tmarkt['Num'] = pd.to_numeric(tmarkt['Num'])
tmarkt = tmarkt.sort_values(by=['ID','Num'],ascending=False)
tmarkt = tmarkt.drop_duplicates(subset='ID',keep='first')


df_jug = df_jug.drop_duplicates(subset=['Nombre','Equipo',
                                        'National team (last match date, mm.yy)'],
                                keep=False)
dup_instat = pd.read_excel(ruta_base+'/Datos/Comprobacion_instat.xlsx')
dup_instat = dup_instat.sort_values(by=['Nombre','Equipo',
                                        'National team (last match date, mm.yy)','Partidos jugados'],ascending=False)
dup_instat = dup_instat.drop_duplicates(subset=['Nombre','Equipo',
                                        'National team (last match date, mm.yy)'],keep='first')

df_jug = pd.concat([df_jug,dup_instat])
df_jug['tmarkt_jug'] = df_jug.Nombre

base_u = base.drop_duplicates(subset=['league-instat','league-id-instat'])
df_jug = pd.merge(df_jug,base_u[['league-instat','league-id-instat']],
                  how='left',right_on='league-instat',left_on='league')
df_jug = df_jug[df_jug['Índice InStat']!=0]

jug_out = pd.DataFrame()
for l in leagues:
    score = []
    d = df_jug[df_jug['league-id-instat']==l]
    newlist_ud=[]
    instat_tm_transform={}
    jug_tmarkt = list(tmarkt[tmarkt['league-id-instat']==l].Players)
    jug_instat = list(d.Nombre)
    for i in jug_instat:
        x=process.extract(i, jug_tmarkt, scorer=fuzz.token_sort_ratio)[0]
        if x[1]>=0:
            instat_tm_transform[i]=x[0]
            newlist_ud.append(x[0])
            score.append(x[1])
    
    d['tmarkt_jug'].replace(instat_tm_transform,inplace=True)
    d['score'] = score
    jug_out = pd.concat([jug_out,d])


    
jug_out_ok = jug_out[jug_out.score>=70]
dupl = jug_out_ok[jug_out_ok.duplicated(subset=['tmarkt_jug','league-id-instat'],keep=False)] 
dupl = dupl.sort_values(by=['tmarkt_jug','score'],ascending=False)
jug_out_ok = jug_out_ok.drop_duplicates(subset=['tmarkt_jug','league-id-instat'],keep=False)
qs = dupl.groupby('Nombre',as_index=False).Num.count()
qs_list = qs[qs.Num>1].Nombre.unique()
dupl_ok = dupl[(~dupl.Nombre.isin(qs_list))]
dupl_ok = dupl_ok.drop_duplicates(subset='tmarkt_jug',keep='first')
jug_out_ok = pd.concat([jug_out_ok,dupl_ok])

dupl = dupl[~dupl.index.isin(dupl_ok.index)]
bad_score = jug_out[~jug_out.index.isin(jug_out_ok.index)]
duplicado = dupl
tmarkt_out = tmarkt[~tmarkt.Players.isin(list(jug_out_ok.tmarkt_jug.unique()))]

bad_score['tmarkt_jug'] = bad_score['Nombre']
jug_out_n = pd.DataFrame()
for l in leagues:
    score = []
    d = bad_score[bad_score['league-id-instat']==l]
    newlist_ud=[]
    instat_tm_transform={}
    jug_tmarkt = list(tmarkt_out[tmarkt_out['league-id-instat']==l].Players)
    jug_instat = list(d.Nombre)
    for i in jug_instat:
        x=process.extract(i, jug_tmarkt, scorer=fuzz.token_set_ratio)[0]
        if x[1]>=0:
            instat_tm_transform[i]=x[0]
            newlist_ud.append(x[0])
            score.append(x[1])
    
    d['tmarkt_jug'].replace(instat_tm_transform,inplace=True)
    d['score'] = score
    jug_out_2 = pd.concat([jug_out_n,d])


jug_out_2_ok = jug_out_2[jug_out_2.score>=80]
dupl = jug_out_2_ok[jug_out_2_ok.duplicated(subset=['tmarkt_jug','league-id-instat'],keep=False)] 
dupl = dupl.sort_values(by=['tmarkt_jug','score'],ascending=False)
jug_out_2_ok = jug_out_2_ok.drop_duplicates(subset=['tmarkt_jug','league-id-instat'],keep=False)
qs = dupl.groupby('Nombre',as_index=False).Num.count()
qs_list = qs[qs.Num>1].Nombre.unique()

bad_score_2 = jug_out_2[~jug_out_2.index.isin(jug_out_2_ok.index)]
duplicado_2 = dupl

df_jug = pd.concat([jug_out_ok,jug_out_2_ok])
dupl = pd.concat([duplicado,duplicado_2])
bad = bad_score_2


badf = bad[(bad['Partidos jugados']>=10) & (bad['Starting lineup appearances']!=0)]
duplf = dupl[(dupl['Partidos jugados']>=10) & (dupl['Starting lineup appearances']!=0)]

duplf.to_excel(ruta_base+'/Datos/Revisiones/duplicados.xlsx',index=False)
badf.to_excel(ruta_base+'/Datos/Revisiones/mal_score.xlsx',index=False)

df_jug_ids = pd.merge(df_jug,tmarkt[['ID','Players','league-id-instat']],
                      how='left',left_on=['tmarkt_jug','league-id-instat'],
                      right_on=['Players','league-id-instat'])



bad_ok = utils_foot.google_scrap(ruta_base+'/Datos/Revisiones','duplicados') 
bad_ok = bad_ok.drop_duplicates(keep='first')
bad_ok = bad_ok[bad_ok.Values.isna()==False]
if bad_ok.ID.isna().any()==False and bad_ok[bad_ok.duplicated(subset='ID',keep=False)].shape[0]==0:
    print('data ok')
    df_jug_ids = pd.concat([df_jug_ids,bad_ok])
else:
    print('data must be double-checked')

dupl_ok = utils_foot.google_scrap(ruta_base+'/Datos/Revisiones','mal_score')
dupl_ok = dupl_ok.drop_duplicates(keep='first')
dupl_ok = dupl_ok[dupl_ok.Values.isna()==False]
if dupl_ok.ID.isna().any()==False and dupl_ok[dupl_ok.duplicated(subset='ID',keep=False)].shape[0]==0:
    print('data ok')
    df_jug_ids = pd.concat([df_jug_ids,dupl_ok])
else:
    print('data must be double-checked')

for i in df_jug_ids.columns:
    if 'Players' or 'league-id-instat' in i:
        df_jug_ids.drop(i,inplace=True,axis=1)

if df_jug_ids.ID.isna().any()==False and df_jug_ids[df_jug_ids.duplicated(subset='ID',keep=False)].shape[0]==0:
    print('data ok')
    df_jug_tot = pd.merge(df_jug_ids,tmarkt[['Players','Values', 'POS', 'Team', 'ID','teamid']],
                      how='left',on='ID')
else:
    print('data must be double-checked')

df_jug_tot = df_jug_tot.drop_duplicates(keep='first')

df_jug_tot = df_jug_tot[df_jug_tot.Values.isna()==False]

"""
dup_clubs = df_jug_tot[df_jug_tot.duplicated(subset=['ID','league-id-instat'],keep=False)]
df_jug_tot = df_jug_tot[~df_jug_tot.index.isin(dup_clubs.index)]

dup_clubs.to_excel(ruta_base+'/Datos/Revisiones/dup_clubs.xlsx',index=False)
duplclubs = pd.read_excel(ruta_base+'/Datos/Revisiones/as1_clubs.xlsx')
df_jug_ids = pd.concat([df_jug_tot,duplclubs])

badf = pd.read_excel(ruta_base+'/Datos/Revisiones/mal_score_ok.xlsx')
badf['ID'] = badf.ID.astype(str)
for i,j in badf.iterrows():
    nid = j['ID']
    if '/' in nid:
        nid = nid.split('/')[-1]
        
    badf.loc[i,'ID'] = nid
    
badf['check'] = badf.ID.apply(lambda x: len(x))

badf_dup = badf[badf.duplicated(subset='ID',keep=False)]
badf['ID'] = pd.to_numeric(badf['ID'])
bad_team = badf[badf.ID.isin(list(df_equipos.teamid.unique()))]
badf.to_excel(ruta_base+'/Datos/Revisiones/mal_score_ok.xlsx',index=False)

badf_tot = pd.merge(badf,tmarkt[['league-id-instat','Values', 'POS', 'Team', 'ID','teamid']],
                      how='left',on='ID')

badf_dup = badf_tot[badf_tot.duplicated(subset='ID',keep=False)]
badf_tot = badf_tot.drop_duplicates(subset=['ID','Nombre'],keep='first')
badna = badf_tot[badf_tot.Values.isna()==True]
#badna.to_excel(ruta_base+'/Datos/Revisiones/mal_id.xlsx',index=False)
badf_tot = badf_tot[badf_tot.Values.isna()==False]
badna = pd.read_excel(ruta_base+'/Datos/Revisiones/mal_id.xlsx')
badna.drop(['Values', 'POS', 'Team','teamid'],axis=1,inplace=True)
for i in badna.columns:
    if 'league-id-instat' in i:
        badna.drop(i,axis=1,inplace=True)
badna = pd.merge(badna,tmarkt[['league-id-instat','Values', 'POS', 'Team', 'ID','teamid']],
                      how='left',on='ID')
badna = badna[badna.Values.isna()==False]
badf_tot = pd.concat([badf_tot,badna])



df_jug_tot_bad = pd.concat([df_jug_tot,badf_tot])
badftot_dup = df_jug_tot_bad[df_jug_tot_bad.duplicated(subset='ID',keep=False)]
badftot_dup = badftot_dup.sort_values(by='ID')
badftot_dup.to_excel(ruta_base+'/Datos/Revisiones/duplicados_id.xlsx',index=False)

df_jug_tot_bad = df_jug_tot_bad[~df_jug_tot_bad.index.isin(badftot_dup.index)]
badftot_dup = pd.read_excel(ruta_base+'/Datos/Revisiones/duplicados_id_ok.xlsx')
badftot_dup.drop(['Values', 'POS', 'Team','teamid'],axis=1,inplace=True)
badftot_dup = pd.merge(badftot_dup,tmarkt[['league-id-instat','Players','Values', 'POS', 'Team', 'ID','teamid']],
                      how='left',on='ID')

df_jug_tot_bad.reset_index(inplace=True, drop=True)

for i in badftot_dup.columns:
    if 'league-id-instat' in i:
        badftot_dup.drop(i,inplace=True,axis=1)
df_jug_tot_bad = pd.concat([df_jug_tot_bad,badftot_dup])
df_jug_tot_bad['Players'] = np.where(df_jug_tot_bad.Players.isna()==True,df_jug_tot_bad.Players_y,df_jug_tot_bad.Players)
df_jug_tot_bad['Players'] = np.where(df_jug_tot_bad.Players.isna()==True,df_jug_tot_bad.Players_x,df_jug_tot_bad.Players)
df_jug_tot_bad['Players'] = np.where(df_jug_tot_bad.Players.isna()==True,df_jug_tot_bad.Nombre,df_jug_tot_bad.Players)
df_jug_tot_bad = df_jug_tot_bad[df_jug_tot_bad.teamid.isna()==False]


dupf = pd.read_excel(ruta_base+'/Datos/Revisiones/duplicados_ok.xlsx')
dupf = dupf[~dupf.ID.isin(df_jug_tot_bad.ID.unique())]
dupf = pd.merge(dupf,tmarkt[['league-id-instat','Players','Values', 'POS', 'Team', 'ID','teamid']],
                      how='left',on='ID')


df_jug_tot_bad_dup = pd.concat([df_jug_tot_bad,dupf])
df_jug_tot_bad_dup = df_jug_tot_bad_dup.drop_duplicates()
dupid = df_jug_tot_bad_dup[df_jug_tot_bad_dup.duplicated(subset='ID',keep=False)]
dupid.to_excel(ruta_base+'/Datos/Revisiones/last_dup.xlsx')
df_jug_tot_bad_dup = df_jug_tot_bad_dup.drop_duplicates(subset='ID',keep=False)

dupid.drop(['league-id-instat','Players','Values', 'POS', 'Team', 'ID','teamid'],inplace=True,
           axis=1)
dupid = pd.read_excel(ruta_base+'/Datos/Revisiones/last_dup_ok.xlsx')
dupid = pd.merge(dupid,tmarkt[['league-id-instat','Players','Values', 'POS', 'Team', 'ID','teamid']],
                      how='left',on='ID')
df_jug_tot_bad_dup_ok = pd.concat([df_jug_tot_bad_dup,dupid])


"""
df_jug_ids = df_jug_ids[df_jug_ids.Players.isna()==False]



equipos_sist = pd.read_csv(ruta_base+'/tmarkt_scraping/equipos_sistemas_{}.csv'.format(temporada),
                           sep=';',decimal=',')
df_equipos_full = pd.merge(df_equipos,equipos_sist,how='left',on='teamid')

df_equipos_full['Tiempo medio de las posesiones'] = df_equipos_full['Tiempo medio de las posesiones'].apply(lambda x: int(x.split(':')[-1]))
df_equipos_full['Posesión del balón, seg'] = df_equipos_full['Posesión del balón, seg'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[-1]))

df_equipos_full = pd.merge(df_equipos_full,base[['league-id-tmarkt', 'league-instat',
       'league-id-instat']],how='left',left_on='league',right_on='league-id-instat')
df_equipos_full = df_equipos_full.drop_duplicates()
df_equipos_full.drop('league',inplace=True,axis=1)
df_equipos_full.to_excel(ruta_datos+'/datos_equipos_instat.xlsx',index=False)

"""
to_drop = ['corto', 'league',
       'tmarkt_jug', 'league-instat', 'league-id-instat', 'score','league-id-instat_x',
       'link', 'check', 'league-id-instat_y', 'Players_x', 'Players_y']

df_jug_tot_bad_dup_ok.drop(to_drop,inplace=True,axis=1)
gk = df_jug_tot_bad_dup_ok[df_jug_tot_bad_dup_ok.POS=='Goalkeeper']
gk.drop(['Players',
       'Values', 'POS', 'Team', 'teamid'],inplace=True,axis=1)

gk.to_excel(ruta_base+'/Datos/Revisiones/gks.xlsx',index=False)
df_jug_tot_bad_dup_ok = df_jug_tot_bad_dup_ok[df_jug_tot_bad_dup_ok.POS!='Goalkeeper']

gk = pd.read_excel(ruta_base+'/Datos/Revisiones/gks_ok.xlsx')
gk = pd.merge(gk,tmarkt[['Players','Values', 'POS', 'Team', 'ID','teamid']],
                      how='left',on='ID')

gk = gk.drop_duplicates(keep='first')
gk = gk[gk.Values.isna()==False]
if gk.ID.isna().any()==False and gk[gk.duplicated(subset='ID',keep=False)].shape[0]==0 and gk.Values.isna().any()==False:
    print('data ok')
    df_jug_tot_bad_dup_ok_gk = pd.concat([df_jug_tot_bad_dup_ok,gk])

dup = df_jug_tot_bad_dup_ok_gk[df_jug_tot_bad_dup_ok_gk.duplicated(subset='ID',keep=False)]
df_jug_tot_bad_dup_ok_gk = df_jug_tot_bad_dup_ok_gk.drop_duplicates(subset='ID',keep=False) 
dup.to_excel(ruta_base+'/Datos/Revisiones/dup.xlsx',index=False)

gk = pd.read_excel(ruta_base+'/Datos/Revisiones/dup_ok.xlsx')
gk = pd.merge(gk,tmarkt[['Players','Values', 'POS', 'Team', 'ID','teamid']],
                      how='left',on='ID')

gk = gk.drop_duplicates(keep='first')
gk = gk[gk.Values.isna()==False]
if gk.ID.isna().any()==False and gk[gk.duplicated(subset='ID',keep=False)].shape[0]==0 and gk.Values.isna().any()==False:
    print('data ok')
    df_jug_tot_bad_dup_ok_gk = pd.concat([df_jug_tot_bad_dup_ok_gk,gk])

"""
df_jug_tot = df_jug_tot.drop_duplicates(keep='first')
df_jug_tot = pd.merge(df_jug_tot,df_equipos_full[['teamid','league-id-instat','league-instat']],
                                 how='left',on='teamid')

if df_jug_tot.ID.isna().any()==False and df_jug_tot[df_jug_tot.duplicated(subset='ID',keep=False)].shape[0]==0:
    df_jug_tot.to_excel(ruta_datos+'/datos_jugadores_instat.xlsx',index=False)

