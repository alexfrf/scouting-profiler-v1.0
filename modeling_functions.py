# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 06:51:40 2022

@author: aleex
"""

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


def get_players_data(equipos,jugadores,cols):
    joiner = 'teamid'
    
    
    grteam = jugadores.groupby(by=joiner,as_index=False)[cols].sum()
    colsr = []
    for i in cols:
        grteam.rename({i:'{}_Players'.format(i)},inplace=True,axis=1)
        colsr.append('{}_Players'.format(i))
    if joiner not in list(grteam.columns):
        colsr.append(joiner)
    
    equipos = pd.merge(equipos,grteam,on=joiner,how='left')
    
    return equipos


def get_teams_data(equipos,jugadores,cols):
    joiner = 'teamid'
    
    if joiner not in cols:
        cols.append(joiner)
    
    jugadores = pd.merge(jugadores,equipos[cols],on=joiner,how='left')
    
    return jugadores



def position_mapping(df):
    df['Posición'] = np.where((df['POS'].str.contains('-Back')) & (df['Posición'].str.startswith('E')),df['Posición'].str.replace('E','L'),df['Posición'])
    df['Posición'] = np.where((df['POS']=='Right Midfield') & (df['Posición'].str.startswith('DC')) ,'LD',df['Posición'])
    df['Posición'] = np.where((df['POS']=='Left Midfield') & (df['Posición'].str.startswith('DC')) ,'LI',df['Posición'])
    df['Posición'] = np.where((df['POS']=='Right Midfield') & (df['Posición'].str.startswith('MC')) ,'MCO',df['Posición'])
    df['Posición'] = np.where((df['POS']=='Left Midfield') & (df['Posición'].str.startswith('MC')) ,'MCO',df['Posición'])
    df['Posición'] = np.where((df['POS']=='Right Midfield') & (df['Posición']=='D') ,'Right Winger',df['Posición'])
    df['Posición'] = np.where((df['POS']=='Left Midfield') & (df['Posición']=='D') ,'Left Winger',df['Posición'])
    df['Posición'] = np.where((df['Posición']=='MC') & ((df.POS.str.contains('Winger')) | (df.POS.str.contains('Forward')) | (df.POS.str.contains('Striker')) | df.POS.str.contains('Attacking')) ,'MCO',df['Posición'])
    df['PosE'] = ''
    df['PosE'] = np.where(df['Posición']=='DC','Centre-Back',df.PosE)
    df['PosE'] = np.where((df['Posición']=='D'),'Forward',df.PosE)
    df['PosE'] = np.where((df['Posición']=='LD')|(df['Posición']=='LI'),'Full-Back',df.PosE)
    df['PosE'] = np.where((df['Posición']=='MCD')|(df['Posición']=='MC'),'Midfielder',df.PosE)
    df['PosE'] = np.where((df['Posición']=='MCO') | (df['Posición'].str.startswith('E')),'Att. Midfield/Winger',df.PosE)
    
    return df

def metrics_players(df):
    df['xg+xa'] = df['Goles esperados'] + df['Expected assists']
    df['Acciones defensivas'] = df['Interceptaciones'] + df['Rechaces'] + df['Entradas'] + df['Faltas'] + df['Disputas defensivas']+ df['Balones recuperados']
    df['Acciones no defensivas'] = df['Acciones totales'] - df['Acciones defensivas']
    df['Acciones defensivas/100Acciones'] = 100*df['Acciones defensivas'] / df['Acciones totales']
    df['PF/100Pases'] = 100*df['Pases de finalización'] / df['Pases']
    df['KP/100Pases'] = 100*df['Pases de finalización efectivos'] / df['Pases']
    df['KP/100Acciones no defensivas'] = 100*df['Pases de finalización efectivos'] / df['Acciones no defensivas']
    df['Regates/100Acciones no defensivas'] = 100*df['Regates'] / df['Acciones no defensivas']
    df['Regates/100Centros'] = 100*df['Regates'] / df['Centros']
    df['Regates/100Acciones no defensivas'] = 100*df['Regates'] / df['Centros']
    df['Interceptaciones+Entradas'] = df['Interceptaciones'] + df['Entradas']
    df['CC/100Acciones no defensivas'] = 100*df['Ocasiones generadas'] / df['Acciones no defensivas']
    df['Tiros/100Acciones no defensivas'] = 100*df['Tiros'] / df['Acciones no defensivas']
    df['KP_Ocasiones%'] = 100 * df['Pases de finalización efectivos'] / df['Ocasiones generadas']
    df['CC/100Pases'] = 100*df['Ocasiones generadas'] / df['Pases']
    df['CC/100Regates'] = 100*df['Ocasiones generadas'] / df['Regates']
    df['CC/100Centros'] = 100*df['Ocasiones generadas'] / df['Centros']
    df['CC/100PF'] = 100*df['Ocasiones generadas'] / df['Pases de finalización']
    df['Centros/100Pases'] = 100*df['Centros'] / df['Pases']
    df['Centros/100Acciones no defensivas'] = 100*df['Centros'] / df['Acciones no defensivas']
    df['PF/100Acciones no defensivas'] = 100*df['Pases de finalización'] / df['Acciones no defensivas']
    df['Centros/100PF'] = 100*df['Centros'] / df['Pases de finalización']
    df['xA/PFe'] = df['Expected assists'] / df['Pases de finalización efectivos']
    df['xA/CC'] = df['Expected assists'] / df['Ocasiones generadas']
    df['Jugada_Gol/100CC'] = 100*df['Jugadas de gol'] / df['Ocasiones generadas']
    df['Jugada_Gol/100PF'] = 100*df['Jugadas de gol'] / df['Pases de finalización']
    df['Jugada_Gol/100Regates'] = 100*df['Jugadas de gol'] / df['Regates']
    df['Jugada_Gol/100Centros'] = 100*df['Jugadas de gol'] / df['Centros']
    df['xG/Jugada_Gol'] = df['Goles esperados'] / df['Jugadas de gol']
    df['Recuperaciones_crival%'] = 100*(df['Recuperaciones en campo rival'] / df['Balones recuperados'])
    df['Recuperaciones_crival/perdidas'] = 100*(df['Recuperaciones en campo rival'] / df['Balones perdidos'])
    df['Perdidas_crival'] = df['Balones perdidos'] - df['Pérdidas en campo propio']
    df['Perdidas_crival%'] = 100*df['Perdidas_crival']/df['Balones perdidos']
    df['Entradas, %'] = 100*df['Entradas efectivas'] / df['Entradas']
    df['Pases/100Acciones defensivas'] = 100*df.Pases / df['Acciones defensivas']
    df['Disputas defensivas/100Acciones defensivas'] = 100*df['Disputas defensivas'] / df['Acciones defensivas']
    df['Disputas/100Acciones defensivas'] = 100*df.Disputas / df['Acciones defensivas']
    for i in df.columns:
        if 'Disputas' in i and '%' not in i and 'PAdj' not in i and '/' not in i and i!='Disputas':
            new = '{}/100Disputas'.format(i)
            df[new] = 100*df[i] / df['Disputas']
            
    
    for i in ['Interceptaciones','Rechaces','Disputas defensivas','Balones recuperados','Entradas','Faltas']:
        new1 = '{}/100Acciones defensivas'.format(i)
        df[new1] = 100* df[i] / df['Acciones defensivas']
        if 'Balones recuperados' not in i:
            new2 = '{}/100Balones recuperados'.format(i)
            df[new2] = 100* df[i] / df['Balones recuperados']
    return df


def metrics_squads(df):
    df['CC'] = df['Asistencias_Players']+df['Pases de finalización efectivos']
    df['Acciones defensivas'] = df['Interceptaciones'] + df['Presión del equipo'] + df['Rechaces'] + df['Entradas'] + df['Faltas'] + df['Disputas defensivas']+ df['Balones recuperados']
    df['Acciones defensivas/100Acciones'] = 100*df['Acciones defensivas'] / df['Acciones totales']
    for i in ['Interceptaciones','Rechaces','Disputas defensivas','Presión del equipo','Balones recuperados','Entradas','Faltas']:
        new1 = '{}/100Acciones defensivas'.format(i)
        df[new1] = 100* df[i] / df['Acciones defensivas']
        if 'Balones recuperados' not in i:
            new2 = '{}/100Balones recuperados'.format(i)
            df[new2] = 100* df[i] / df['Balones recuperados']
    df['xg+xa'] = df['Goles esperados'] + df['Expected assists_Players']
    df['Presiones'] = df['High pressing'] + df['Low pressing']
    df['Presiones/AtaquesPos'] = 100*df['Presión del equipo'] / df['Ataques posicionales']
    df['Presiones/hPress'] = (df['High pressing']/(df['High pressing'] + df['Low pressing'])) *100
    df['Contraataques/100Bups'] = 100*df['Contraataques'] / df['Building-ups']
    df['Contraataques/100Posesiones'] = 100*df['Contraataques'] / df['Posesiones de balón, cantidad']
    df['Bups/100Posesiones'] = 100*df['Building-ups'] / df['Posesiones de balón, cantidad']
    df['Recuperaciones_crival%'] = 100*(df['Recuperaciones en campo rival'] / df['Balones recuperados'])
    df['Recuperaciones_crival/perdidas'] = 100*(df['Recuperaciones en campo rival'] / df['Balones perdidos'])
    df['Contraataques/100Recuperaciones'] = 100*(df['Contraataques'] / df['Balones recuperados'])
    df['HighPress/100RecuperacionesCRival'] = 100*(df['High pressing'] / df['Recuperaciones en campo rival'])
    df['Presiones/100Recuperaciones'] = 100*(df['Presión del equipo'] / df['Balones recuperados'])
    df['PerdidasCRival'] = df['Balones perdidos'] - df['Pérdidas en campo propio']
    df['Presiones/100PerdidasCRival'] = 100*(df['Presión del equipo'] / df['PerdidasCRival'])
    df['PF/100Pases'] = 100*df['Pases de finalización'] / df['Pases']
    df['KP/100Pases'] = 100*df['Pases de finalización efectivos'] / df['Pases']
    df['KP_Ocasiones%'] = 100 * df['Pases de finalización efectivos'] / df['Jugadas de gol']
    df['CC/100Pases'] = 100*df['CC'] / df['Pases']
    df['CC/100IncursionesUT'] = 100*df['CC'] / df['Entrada al último tercio de campo']
    df['KP/100IncARival'] = 100*df['Pases de finalización efectivos'] / df['Entrada al área rival']
    df['Centros/100IncARival'] = 100*df['Centros'] / df['Entrada al área rival']
    df['Centros/100Pases'] = 100*df['Centros'] / df['Pases']
    df['Centros/100PF'] = 100*df['Centros'] / df['Pases de finalización']
    df['IncursionesUT/Pases'] = 100*df['Entrada al último tercio de campo'] / df['Pases']
    df['IncursionesCRival/Pases'] = 100*df['Entrada a campo contrario'] / df['Pases']
    df['IncursionesUT/Regates'] = 100*df['Entrada al último tercio de campo']/ df['Regates']
    df['IncursionesARival/Pases'] = 100*df['Entrada al área rival'] / df['Pases']
    df['Juego por Banda, %'] = 100 * (df['Ataques por banda derecha'] + df['Ataques por banda izquierda']) / (df['Ataques por banda derecha'] + df['Ataques por banda izquierda'] + df['Ataques por la zona central'])
    df['Juego por Banda Derecha, %'] = 100 * (df['Ataques por banda derecha']) / (df['Ataques por banda derecha'] + df['Ataques por banda izquierda'] + df['Ataques por la zona central'])
    df['Juego por Banda Izquierda, %'] = 100 * (df['Ataques por banda izquierda']) / (df['Ataques por banda derecha'] + df['Ataques por banda izquierda'] + df['Ataques por la zona central'])
    df['Jugada_Gol/100Bups'] = 100*df['Jugadas de gol'] / df['Building-ups']
    df['Jugada_Gol/100Contraataques'] = 100*df['Jugadas de gol'] / df['Contraataques']
    df['Jugada_Gol/100ABP'] = 100*df['Jugadas de gol'] / df['Acciones a balón parado']
    df['Jugada_Gol/100PF'] = 100*df['Jugadas de gol'] / df['Pases de finalización']
    df['Jugada_Gol/100IncursionesUT'] = 100*df['Jugadas de gol'] / df['Entrada al último tercio de campo']
    df['Jugada_Gol/100IncursionesCRival'] = 100*df['Jugadas de gol'] / df['Entrada a campo contrario']
    df['Jugada_Gol/100Regates'] = 100*df['Jugadas de gol'] / df['Regates']
    df['Jugada_Gol/100Centros'] = 100*df['Jugadas de gol'] / df['Centros']
    df['xG/Jugada_Gol'] = df['Goles esperados'] / df['Jugadas de gol']
    df['xA/PFe'] = df['Expected assists_Players'] / df['Pases de finalización efectivos']
    df['xA/CC'] = df['Expected assists_Players'] / df['CC']
    df['pace'] = df['Posesión del balón, seg'] / (df['Entrada al último tercio de campo'])
    df['Tiros/JugadaGol'] = df['Tiros'] / df['Jugadas de gol']
    df['Tiros/100Acciones']= 100*df['Tiros'] / df['Acciones totales']
    df['xG/100Acciones']= 100*df['Goles esperados'] / df['Acciones totales']
    df['Centros/100AccionesBanda'] = 100*df['Centros'] / (df['Ataques por banda izquierda'] + df['Ataques por banda derecha'])
    return df

def possession_adj(df):
    cols = df.columns
    for j in cols:
        if is_numeric_dtype(df[j])==True and ' per ' not in j and '%' not in j and '/' not in j and 'InStat' not in j and 'id' not in j:
            new = 'PAdj_{}'.format(j)
            df[new] = df[j]* (df['Posesión del balón, %'] / 50)
    return df

def actions_adj(df):
    cols = df.columns
    for j in cols:
        if is_numeric_dtype(df[j])==True and 'PAdj' not in j and 'Acciones' not in j and ' per ' not in j and '%' not in j and '/' not in j and 'InStat' not in j and 'id' not in j:
            new = '{}/100Acciones'.format(j)
            df[new] = 100*df[j] / df['Acciones totales']
    return df
    

def gini(df,col):
    list_of_values = list(df[col])
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area

def scaler_idx(df,col):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X = df[col].values
    X = X.reshape(-1,1)
    X = scaler.fit_transform(X)
    return X