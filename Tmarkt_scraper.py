# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 17:16:42 2021

@author: aleex
"""

import csv
from fuzzywuzzy import process, fuzz
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time
import os
from unidecode import unidecode
import warnings
import zipfile
import datetime as dt

warnings.filterwarnings('ignore')


#s = str(2021)

CWD = os.getcwd()
DIR = os.path.join(CWD,"tmarkt_scraping")

def download_tmarkt_files():   
    if os.path.exists(os.path.join(DIR,"player-scores.zip")):
        with zipfile.ZipFile(os.path.join(DIR,"player-scores.zip"), 'r') as zip_ref:
            zip_ref.extractall(DIR)
        #os.remove(os.path.join(DIR,"player-scores.zip"))
        print("New transfermarkt data {}".format(dt.datetime.today().strftime('%d/%m/%Y')))
    else:
        print('Zip file does not exist - Getting the newest available players.csv file')
        pass

headers = {'User-Agent': 
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

leagues_url = ["https://www.transfermarkt.com/laliga/startseite/wettbewerb/ES1/plus/?saison_id={}",
               "https://www.transfermarkt.com/bundesliga/startseite/wettbewerb/L1/plus/?saison_id={}",
               "https://www.transfermarkt.com/serie-a/startseite/wettbewerb/IT1/plus/?saison_id={}",
               "https://www.transfermarkt.com/ligue-1/startseite/wettbewerb/FR1/plus/?saison_id={}",
               "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1/plus/?saison_id={}",
"https://www.transfermarkt.com/premier-liga/startseite/wettbewerb/RU1/plus/?saison_id={}",
"https://www.transfermarkt.com/jupiler-pro-league/startseite/wettbewerb/BE1/plus/?saison_id={}",
"https://www.transfermarkt.com/championship/startseite/wettbewerb/GB2/plus/?saison_id={}",
"https://www.transfermarkt.com/ligue-2/startseite/wettbewerb/FR2/plus/?saison_id={}",
"https://www.transfermarkt.com/liga-portugal/startseite/wettbewerb/PO1/plus/?saison_id={}",
"https://www.transfermarkt.com/pko-ekstraklasa/startseite/wettbewerb/PL1/plus/?saison_id={}",
"https://www.transfermarkt.com/super-liga-srbije/startseite/wettbewerb/SER1/plus/?saison_id={}",
"https://www.transfermarkt.com/prva-liga/startseite/wettbewerb/SL1/plus/?saison_id={}",
"https://www.transfermarkt.com/laliga2/startseite/wettbewerb/ES2/plus/?saison_id={}",
"https://www.transfermarkt.com/super-lig/startseite/wettbewerb/TR1/plus/?saison_id={}",
"https://www.transfermarkt.com/professional-football-league/startseite/wettbewerb/AR1N/plus/?saison_id={}",
"https://www.transfermarkt.com/bundesliga/startseite/wettbewerb/A1/plus/?saison_id={}",
"https://www.transfermarkt.com/campeonato-brasileiro-serie-a/startseite/wettbewerb/BRA1/plus/?saison_id={}",
"https://www.transfermarkt.com/supersport-hnl/startseite/wettbewerb/KR1/plus/?saison_id={}",
"https://www.transfermarkt.com/2-bundesliga/startseite/wettbewerb/L2/plus/?saison_id={}",
"https://www.transfermarkt.com/super-league-1/startseite/wettbewerb/GR1/plus/?saison_id={}",
"https://www.transfermarkt.com/nemzeti-bajnoksag/startseite/wettbewerb/UNG1/plus/?saison_id={}",
"https://www.transfermarkt.com/serie-b/startseite/wettbewerb/IT2/plus/?saison_id={}",
"https://www.transfermarkt.com/eliteserien/startseite/wettbewerb/NO1/plus/?saison_id={}",
"https://www.transfermarkt.com/superliga/startseite/wettbewerb/RO1/plus/?saison_id={}",
"https://www.transfermarkt.com/scottish-premiership/startseite/wettbewerb/SC1/plus/?saison_id={}",
"https://www.transfermarkt.com/allsvenskan/startseite/wettbewerb/SE1/plus/?saison_id={}",
"https://www.transfermarkt.com/super-league/startseite/wettbewerb/C1/plus/?saison_id={}",
"https://www.transfermarkt.com/fortuna-liga/startseite/wettbewerb/TS1/plus/?saison_id={}",
"https://www.transfermarkt.com/efbet-liga/startseite/wettbewerb/BU1/plus/?saison_id={}"
 ]


temporada = '2021-2022'
def transfermarkt_scrap(season):
    
    if type(season)==str:
        if '-' in season:
            season = str(season[:4])
    else:
        season = str(season)
    print('Getting Transfermarkt players data for the {}-{} season'.format(season,str(int(season)+1)))
    
    
    
    team_lg = pd.DataFrame()
    for url_l in leagues_url:
        teams_url = []
        #print(url)
        league=url_l.split('https://www.transfermarkt.com/')[-1]
        league = league.split('/')[0].title()
        sub = url_l.split(league.lower()+'/startseite/wettbewerb/')[-1]
        sub = sub.split('/')[0].title()
        #sub = sub[:4].upper()
        sub = sub.replace('/','')
        reqs = requests.get(url_l.format(season),headers=headers) 
        soup = BeautifulSoup(reqs.content, 'html.parser')
        links = soup.select("tbody")[1]
        teamLinks = []
        
        for link in soup.select("tbody")[1].find_all('a'):
            l = link.get("href")
            if 'startseite' in l:
                teamLinks.append(l)
        
        
        for i in range(len(teamLinks)):
            teamLinks[i] = "https://www.transfermarkt.com"+teamLinks[i]
        
        teamLinks = list(set(teamLinks))
        
        for i in teamLinks:
            teams_url.append(i)
    
       
        poslist = list(pd.read_csv(DIR+'/POS.csv').Pos.unique())
    
    
    
        x=[]
        xid =[]
        rep=0
        list_df=[]
        Players=''
        for url in teams_url:
            rep +=1
            
            team = url.split('https://www.transfermarkt.com/')
            #print(url)
            team=team[1].split('/')[0].replace('-',' ').title()
            x.append(team)
            teamid = url.split('/startseite/verein/')
            #print(url)
            teamid=teamid[1].split('/')[0].replace('-',' ').title()
            xid.append(teamid)
        
            page = url
            print(page)
            pageTree = requests.get(page, headers=headers)
            pageSoup = BeautifulSoup(pageTree.content, 'html.parser')
            Players = pageSoup.find_all("td", {"itemprop": "athlete"})
            Values = pageSoup.find_all("td", {"class": "rechts hauptlink"})
            Number = pageSoup.find_all("div", {"class": "tm-shirt-number"})
            if len(Players)==0:
                Players=[]
                for i in pageSoup.find_all("span", {"class": "hide-for-small"},{"title":True}):
                    if len(i.text)>0:
                        Players.append(i)
                    
            Pos = pageSoup.find_all("table", {"class":"inline-table"})
            container = pageSoup.find_all("a", {"class":"spielprofil_tooltip"})
            nation = pageSoup.find("img", {"class": "flaggenrahmen"}, {"title":True})
            age = pageSoup.find_all("td", {"class":"zentriert"})
            
            PlayersList=[]
            ValuesList=[]
            PosList=[]
            cont=[]
            teamlist=[]
            complist=[]
            nationlist=[]
            agelist=[]
            idslist=[]
            numlist = []
            
            df = pd.DataFrame(columns={"Players","Values","POS","Team","Comp","Age","Nation","ID"})
            c = []
            for i in range(0,len(Players)):
                p = Players[i].text
                if len(p)!=0 and str(Players[i]) not in c:
                    c.append(str(Players[i]))
                    print(p)
                    PlayersList.append(Players[i].text)
                    ValuesList.append(Values[i].text)
                    numlist.append(Number[i].text)
                    PosList.append(Pos[i].text.replace(Players[i].text,''))
                    teamlist=team
                    for k in pageSoup.find_all("a"):
                        if p in k:
                            #print(p)
                            #print(k)
                            if k.get("href").split('/')[-1] not in idslist and 'spieler' in k.get("href"):
                                idslist.append(k.get("href").split('/')[-1])
                
                #nationlist.append(nation[i].text)
                #age[i]= age[i].text.split(' ')[-1]
                #agelist.append(age[i].text)
                try:
                    df = pd.DataFrame({"Players":PlayersList,"Values":ValuesList,"POS":PosList,"Team":teamlist,
                                   "ID":idslist,"Num":numlist})
                except:
                    pass
                
                
            df['Values'] = df['Values'].str.replace(r'[a-z]','').str.replace('€','')
            for i in ['Players','Team']:
                        df[i] = df[i].apply(unidecode)
            surnames = []
            for i in df['Players'].str.split(' '):
                surnames.append(i[-1])
            for i in surnames:
                df['POS'] = df['POS'].str.replace(i,'')
            
        
            print('\n')
             
            
            
        #dataf = pd.concat(list_df)
                
            df['Values'] = df.Values.str.split().str.join(' ')
            df['Values'] = df.Values.str.replace('T.','')
            df['Values'] = df.Values.astype(str)
            df['Values'] = df.Values.replace('nan','')
            df['Values'] = df.Values.replace('','0.10')
            df['Values'] = df.Values.replace('-','0.00')
            
            #df['Values'] = df.Values.astype(float)
            values = list(df.Values)
            a=[]
            for i in values:
                if '.' in i:
                    a.append(float(i))
                elif '-' in i:
                    i==0
                    a.append(i)
                else:
                    a.append(float(i)/1000)
            df['Values'] = a
            
            pos=poslist
            dfpos=df['POS'].unique().tolist()
            nlist=[]
            pos_transform={}
            for i in dfpos:
                p=process.extract(i, pos, scorer=fuzz.token_set_ratio)[0][0]
                pos_transform[i]=p
                nlist.append(p)
            df['POS']=df.POS.map(pos_transform)
        
    
            list_df.append(df) 
        
        
        dataf = pd.concat(list_df)
        dataf =dataf.sort_values(by='Values',ascending=False).drop_duplicates(subset='Players',keep='first').sort_values(by='Values',ascending=False)
        dataf.to_csv(DIR+'/leagues/tmarkt_{}-{}.csv'.format(league,sub),decimal=',',sep=';',index=False)
        time.sleep(5)
        
        
        lea = pd.DataFrame(columns=['team','teamid','league','league-id-tmarkt'])
        lea['team'] = x
        lea['teamid'] = xid
        lea['league'] = league
        lea['league-id-tmarkt'] = sub
        
        team_lg = pd.concat([team_lg,lea])
        
        
    
        
    team_lg.to_csv(DIR+'/leagues/exp_leagues_{}.csv'.format(season),decimal=',',sep=';',index=False)
        
        #time.sleep(5)


def transfermarkt_kaggle_scrap():
    download_tmarkt_files()

    players = pd.read_csv(DIR+'/players.csv')
    #players= players[players['last_season']] 
    players = players[['player_id','pretty_name','sub_position','foot','height_in_cm','market_value_in_gbp','url']]
    cambio = 1.17
    players.columns = ['ID','player','POS','foot','height','value','url']
    players['value'] = round(players['value']/1000000*cambio,1)
    players['value'].fillna(0,inplace=True)
    
    """
    comps = list(pd.read_csv(DIR+'/comps.csv')['competition_id'])
    apps = pd.read_csv(DIR+'/appearances.csv')
    apps = apps[apps['competition_id'].isin(comps)]
    apps = apps.drop_duplicates(subset='player_id')
    ids_apps = list(apps['player_id'])
    players = players[players['ID'].isin(ids_apps)]
    """
    players.to_csv(DIR+'/tmarkt_{}-{}.csv'.format(dt.datetime.today().strftime('%Y'),dt.datetime.today().strftime('%m')),index=False)
    
    
def tmarkt_join_files(ruta,season):
    tm = pd.DataFrame()
    for f in os.listdir(ruta):
        file = os.path.join(ruta,f)
        df = pd.read_csv(file,sep=';',decimal=',')
        if 'tmarkt_' in f:
            tm = pd.concat([tm,df])
    for i in tm.columns:
        if 'Unnamed: ' in i:
            tm.drop(i,inplace=True,axis=1)
    tm.to_csv(ruta+'/tmarkt_{}.csv'.format(season),decimal=',',sep=';')
    
    
def tmarkt_leagues_joiner(ruta,season,csv_name):
    df = pd.read_csv(ruta+'/{}_{}.csv'.format(csv_name,season),decimal=',',sep=';')
    exp = pd.read_csv(ruta+'/exp_{}_{}.csv'.format(csv_name,season),decimal=',',sep=';')
    
    df = pd.merge(df,exp[['teamid','team']],how='left',on='team')
    for i in df.columns:
        if 'Unnamed: ' in i:
            df.drop(i,inplace=True,axis=1)
    df.to_csv(ruta+'/{}_{}.csv'.format(csv_name,season),decimal=',',sep=';')




def tmarkt_add_urlsys(ruta,season,csvname):
    file = ruta+'/{}_{}.csv'.format(csvname,season)
    df = pd.read_csv(file,sep=';',decimal=',')
    url_sys = []
    for i,j in df.iterrows():
        teamid = j['teamid']
        team = j['team']
        url = 'https://www.transfermarkt.com/{}/spielplan/verein/{}/saison_id/{}/plus/1#ES1'.format(team,teamid,season[:4])
        url_sys.append(url)
     
    if df.shape[0]==len(url_sys):
        df['url_sys'] = url_sys
    for i in df.columns:
        if 'Unnamed: ' in i:
            df.drop(i,inplace=True,axis=1)
    df.to_csv(file,decimal=',',sep=';')
    
#tmarkt_add_urlsys(DIR,temporada,'leagues')

def tmarkt_system_scraper(ruta,season,csvname):
    df = pd.read_csv('{}/{}_{}.csv'.format(ruta,csvname,season),sep=';',decimal=',')
    #teams_url = list(df.url_sys.unique())
    rep = 0
    xid = []
    sistemas = pd.DataFrame()
    for i,j in df.iterrows():
        rep +=1
        url = j["url_sys"]     
        teamid = j['teamid']
        team = j['team']
        l = j['league'].lower()
        print('\n')
        print(team)
        data = pd.DataFrame()
        #print(url)
        xid.append(teamid)
    
        page = url
        #print(page)
        reqs = requests.get(url,headers=headers)
        soup = BeautifulSoup(reqs.content, 'html.parser')
        #elem = soup.find_all("a", {"name": "ES1"})
        table = soup.find("a", {"name": "{}".format(j['league-id-tmarkt'])}).find_next('table')
        system = table.find_all("td", {"class": "zentriert"})
        system_list = []
        
        for i in range(len(system)):
            sys = str(system[i].text).split(' ')[0].strip()
            if ('-' in sys or sys=='?') and len(sys)>0:
                system_list.append(sys)
        coach = table.find_all("a", id=lambda x: x is not None and x.isnumeric())
        jor = []
        it=0
        coach_list = []
        last_coach = ''
        for i in range(len(coach)):
            if ':' not in coach[i].text:
                it+=1   
                jor.append(it)
                coach_list.append(coach[i].text)
                last_coach = coach[i].text
        
        
        while len(system_list)>len(jor):
            jor.append(it+1)
            coach_list.append('0')
        else:
            
            data['jor'] = jor
            data['coach'] = coach_list
            data['system'] = system_list
            data['last_coach'] = last_coach
            
        data['teamid'] = teamid
        print(it)
        
        #sistemas = pd.concat([sistemas,data])
        data = data[(data.system!='?') & (data.system.isna()==False)]
        data['time'] = 90   
        gr = data.groupby(by='teamid',as_index=False)['jor'].agg('max')
        gr.columns=['teamid','total_time_played']
        data = pd.merge(data,gr,how='left',on='teamid')
        data['total_time_played'] = data['total_time_played'] * data['time']
        
        gr1 = data.groupby(by=['teamid','system'],as_index=False)['time'].agg('sum')
        gr1.columns = ['teamid','system','time_system']
        data = pd.merge(data,gr1,on=['system','teamid'],how='left')
        data['Use%'] = round(100*(data.time_system/data.total_time_played),2)
        
        gr = data.groupby(by='system',as_index=False)['Use%'].agg(['mean','std','count'])
        gr = gr.reset_index()
        gr.columns=['system','mean_Use%','std_Use%','num_games_played']
        data = pd.merge(data,gr,on='system',how='left')
        num = len(data.system.unique())
        data['num_formations_played'] =num
        
        
        data['stat_def'] = data.system.str[0]
        gr1 = data.groupby(by=['teamid','stat_def'],as_index=False)['time'].agg('sum')
        gr1.columns = ['teamid','stat_def','time_def']
        data = pd.merge(data,gr1,on=['teamid','stat_def'],how='left')
        data['Use%_def'] = round(100*(data.time_def/data.total_time_played),2)
        
        data['stat_med'] = data.system.str[2]
        gr1 = data.groupby(by=['teamid','stat_med'],as_index=False)['time'].agg('sum')
        gr1.columns = ['teamid','stat_med','time_med']
        data = pd.merge(data,gr1,on=['teamid','stat_med'],how='left')
        data['Use%_med'] = round(100*(data.time_med/data.total_time_played),2)
    
        for i,j in data.iterrows():
            if len(j['system'])!=5:
                data.loc[i,'stat_mp'] = data.loc[i,'system'][4]
            else:
                data.loc[i,'stat_mp'] = str(0)
        gr1 = data.groupby(by=['teamid','stat_mp'],as_index=False)['time'].agg('sum')
        gr1.columns = ['teamid','stat_mp','time_mp']
        data = pd.merge(data,gr1,on=['teamid','stat_mp'],how='left')
        data['Use%_mp'] = round(100*(data.time_mp/data.total_time_played),2)
        
        
        data['stat_ata'] = data.system.str[-1]
        gr1 = data.groupby(by=['system','stat_ata'],as_index=False)['time'].agg('sum')
        gr1.columns = ['system','stat_ata','time_ata']
        data = pd.merge(data,gr1,on=['system','stat_ata'],how='left')
        data['Use%_ata'] = round(100*(data.time_ata/data.total_time_played),2)
        
        
        for i,j in data.iterrows():
            for k in list(data.stat_def.unique()):
                if data.loc[i,'stat_def']==k:
                    data.loc[i,'time_def_k{}'.format(k)] = data.loc[i,'time_def']
                    data.loc[i,'Use%_def_k{}'.format(k)] = data.loc[i,'Use%_def']
                    
            for k in list(data.stat_med.unique()):
                if data.loc[i,'stat_med']==k:
                    data.loc[i,'time_med_k{}'.format(k)] = data.loc[i,'time_med']
                    data.loc[i,'Use%_med_k{}'.format(k)] = data.loc[i,'Use%_med']
                    
            for k in list(data.stat_mp.unique()):
                if data.loc[i,'stat_mp']==k:
                    data.loc[i,'time_mp_k{}'.format(k)] = data.loc[i,'time_mp']
                    data.loc[i,'Use%_mp_k{}'.format(k)] = data.loc[i,'Use%_mp']
                    
            for k in list(data.stat_ata.unique()):
                if data.loc[i,'stat_ata']==k:
                    data.loc[i,'time_ata_k{}'.format(k)] = data.loc[i,'time_ata']
                    data.loc[i,'Use%_ata_k{}'.format(k)] = data.loc[i,'Use%_ata']
                    
        data = data.fillna(0)            
        new_cols=[]
        for i in data.columns:
            if '_k' in i:
                new_cols.append(i)
        
        
        new_cols = sorted(new_cols)
                
        gr2 = data.groupby(by='teamid',as_index=False)[new_cols].agg('max')
        data.drop(new_cols,axis=1,inplace=True)
        data = pd.merge(data,gr2,how='left',on='teamid')
        
        for i in data.columns[3:]:
            if i!='teamid' and (data[i].astype=='float64' or data[i].astype=='float16' or data[i].astype=='float32'):
                data[i] = round(data[i],2)
        #data = data.sort_values(by=['team','Use%'],ascending=False)
        most_common_system = data.groupby(by='system',as_index=False).teamid.count()
        most_common_system = most_common_system[most_common_system.teamid.max()==most_common_system.teamid].system.values[0]
        data['most_common_system'] = most_common_system
        
        for r,i in zip(range(6),['def','med','mp','ata']):
            for k in ['time','Use%']:
                col = '{}_{}_k{}'.format(k,i,r)
                if col not in new_cols:
                    new_cols.append(col)
        
        for cols in new_cols:
            if cols not in data.columns:
                data[cols] = 0
        
        data = data[data.system==most_common_system]
        to_drop = ['jor','coach','system','time',
           'total_time_played']
        
        data.drop(to_drop,inplace=True,axis=1)
        data = data.drop_duplicates()
        
        sistemas = pd.concat([sistemas,data])
        
        time.sleep(5)
    
    if len(sistemas.isna().any().unique()) != 1:  
        for c in sistemas.columns:
            if sistemas[c].isna().any()==True:
                sistemas[c] = sistemas[c].fillna(0)
                
    sistemas=sistemas[['teamid','most_common_system','time_system','Use%','last_coach',  'mean_Use%', 'std_Use%',
           'num_games_played', 'num_formations_played', 'stat_def', 'time_def',
           'Use%_def', 'stat_med', 'time_med', 'Use%_med', 'stat_mp', 'time_mp',
           'Use%_mp', 'stat_ata', 'time_ata', 'Use%_ata', 'Use%_ata_k1',
           'Use%_ata_k2', 'Use%_ata_k3', 'Use%_def_k3', 'Use%_def_k4',
           'Use%_def_k5', 'Use%_med_k2', 'Use%_med_k3', 'Use%_med_k4',
           'Use%_med_k5', 'Use%_mp_k0', 'Use%_mp_k1', 'Use%_mp_k2', 'Use%_mp_k3',
           'time_ata_k1', 'time_ata_k2', 'time_ata_k3', 'time_def_k3',
           'time_def_k4', 'time_def_k5', 'time_med_k2', 'time_med_k3',
           'time_med_k4', 'time_med_k5', 'time_mp_k0', 'time_mp_k1', 'time_mp_k2',
           'time_mp_k3', 'time_def_k0', 'Use%_def_k0',
           'time_med_k1', 'Use%_med_k1', 'Use%_mp_k4', 'time_mp_k4']]
    sistemas.to_csv(ruta+'/equipos_sistemas_{}.csv'.format(season),decimal=',',sep=';',index=False)
    

    
