# -*- coding: utf-8 -*-
"""
Created on Fri May 27 18:17:34 2022

@author: aleex
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
import warnings
import shutil
from pathlib import Path
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import re
from googlesearch import search
import random
import time


#s = str(2021)



 # 1 si deseamos datos actuales (por defecto), 0 si queremos una exportación del pasado
warnings.filterwarnings('ignore')
#def get_season(month)
temporada='2021-2022'
CWD = os.getcwd()
DIR = os.path.join(CWD,"tmarkt_scraping")

def get_season(s=0):
    if s==0:
        datos_actualizados=1
    else:
        datos_actualizados=0
    k='{}-{}'
    ruta_base = os.path.join('',Path(os.getcwd()))
    
    listfiles = []
    if datos_actualizados!=1:
        season=s[:4]
        ruta=ruta_base + '/' + season
        
        
    else:
        month = int(dt.datetime.today().strftime('%m'))
        if month<=8:
            season = k.format(str(int(dt.datetime.today().strftime('%Y'))-1),dt.datetime.today().strftime('%Y'))
        else:
            season = k.format(dt.datetime.today().strftime('%Y'),str(int(dt.datetime.today().strftime('%Y'))+1))
        try:      
            os.makedirs(ruta_base+'/'+season+'/{}/Output'.format(dt.datetime.today().strftime('%Y_%m')))
        except:
            #print('File already exists')
            pass
            
        for (dirpath, dirnames, filenames) in os.walk(ruta):    
            listfiles += [os.path.join(dirpath, file) for file in filenames]
            for i in listfiles:
                k = season+'/{}'.format(dt.datetime.today().strftime('%Y_%m'))
                if 'Output' in i and 'png' not in i:
                    k+='/Output'
                    shutil.copy(i,i.replace('Current',k))
    return season,ruta_base


def import_image(ax,logo,x,y):
    imagebox = OffsetImage(logo,zoom=0.12)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, [x,y], pad=0, frameon=False)
    return ax.add_artist(ab)

def clean_df(df):
    '''
    función que limpia los dataframes de datos agregados (ya sean equipos, jugadores o porteros) procedentes de instat.
    Toma el dataframe previamente importado desde excel
    '''
    df = df.replace('%', '', regex = True)

    for i in df.columns:
        if i!='Nombre' and i!= 'Equipo':
            
            df[i] = df[i].replace('-', '0', regex = True)
            if pd.to_numeric(df[i], errors='coerce').notnull().all()==True and 'fecha' not in i and 'Nombre' not in i and 'Equipo' not in i:
                df[i] = df[i].apply(pd.to_numeric)
        else:
            df[i] = df[i].astype(str)
            
    df['corto'] = df.Equipo.apply(lambda x: re.sub(r'\b\w{1,3}\b', '', x))
    df['corto'] = df['corto'].str.replace('.','').str.strip()
    df['corto'] = df['corto'].str.replace('  ',' ').str.strip()

    return df
       
USER_AGENTS = [
   #Chrome
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    #Firefox
    'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
    'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)'
]

def google_scrap(ruta,xlsx):    

    df = pd.read_excel('{}/{}.xlsx'.format(ruta,xlsx))
    df['ID'] = df['ID'].fillna("")
    
    
    #desc = df[(df['Partidos jugados']<10) | (df['Starting lineup appearances']==0)]
    #df = df[~df.index.isin(desc.index)]
    #list_ids = []
    for i,row in df.iterrows():
        l = "{} {} {}".format(row['Nombre'],row['Equipo'],"TRANSFERMARKT")
        col = row['ID']
        if col=="":
            #idx= df[df.link==i].index[0]
            url= l+ ' Perfil del jugador'
            url = url.replace('https://www.google.com/search?q=','')
            
            for j in search(url, tld="co.in", num=1, stop=1, pause=2):
                print(j)
                #ids = str(j.split('/')[-1])
            
                df.loc[i,'ID'] = j
                #list_ids.append(ids)
                time.sleep(random.randint(20,30))
    
    df['ID'] = df.ID.apply(lambda x: int(x.split('/')[-1]))
    #df.to_excel('{}/{}_ok.xlsx'.format(ruta,xlsx),index=False)
    return df    
    

#ids_tmarkt = google_scrap(CWD+'/Datos/Revisiones','asignacion_manual')         