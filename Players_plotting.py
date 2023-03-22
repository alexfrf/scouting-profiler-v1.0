# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:47:25 2021

@author: aleex
"""
#from indexes import *
from mplsoccer import Pitch,add_image,FontManager
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm
from PIL import Image
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.patches import Circle, Rectangle, Arc
import matplotlib.patheffects as path_effects
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import io
import requests
from io import BytesIO

cbs_radar = list(set(['% disputas por arriba ganadas','PAdj_Interceptaciones+Entradas',
                'PAdj_Rechaces','PAdj_Entradas','Recuperaciones en campo rival',
                'Entradas/100Acciones defensivas','Entradas, %','Robos de balón con éxito, %',
        '% disputas defensivas ganadas','PAdj_Balones recuperados','PAdj_Acciones defensivas','Pases/100Acciones defensivas',
        '% de efectividad de pases','PAdj_Rechaces'
    ]))

mid_radar = list(set([
    'Pases','Regates efectivos','Ocasiones generadas','PAdj_Interceptaciones+Entradas',
    '% de efectividad de pases','PAdj_Disputas por arriba ganadas','Recuperaciones en campo rival',
    'PAdj_Balones recuperados','PAdj_Disputas ganadas','% de efectividad de pases',
    'Entradas/100Acciones defensivas','PAdj_Rechaces','Tiros/100Acciones no defensivas','Entradas, %',
    'Pases de finalización efectivos'
    ]))


flb_radar=list(set(['Centros efectivos','Pases de finalización efectivos',
     'Regates efectivos','% de efectividad de pases','% de efectividad de los centros',
     '% disputas por arriba ganadas','Recuperaciones en campo rival','PAdj_Interceptaciones+Entradas',
     'Ocasiones generadas','PAdj_Balones recuperados',
     'Expected assists','Disputas/100Acciones defensivas','Pases efectivos']
             ))

attm_radar=list(set(['Goles esperados','Pases efectivos','PF/100Acciones no defensivas',
     'Regates efectivos','% de efectividad de pases','Ocasiones generadas','Expected assists',
     'Pases de finalización efectivos','Recuperaciones en campo rival','Ocasiones generadas',
     'Centros efectivos','Jugadas de gol','Ocasiones de gol, % conversión']))

fwd_radar=list(set(['Regates efectivos','Pases de finalización efectivos',
              'Ocasiones generadas','Expected assists','Goles esperados',
     'Jugadas de gol','Tiros a portería','xG per shot','Ocasiones de gol, % conversión',
     '% de efectividad de pases','% de efectividad de pases','% disputas por arriba ganadas',
     'Pases efectivos']))

pos_dict= {'Centre-Back':cbs_radar,
           'Midfielder':mid_radar,
           'Att. Midfield/Winger':attm_radar,
           'Full-Back':flb_radar,
           'Forward':fwd_radar}

sns.set(style="whitegrid")
ruta_datos = os.path.join(os.getcwd(),"Datos")

players_df = pd.read_csv(ruta_datos+'/Modeled/jugadores.csv',sep=';',decimal=',')
squad_df = pd.read_csv(ruta_datos+'/Modeled/equipos.csv',sep=';',decimal=',')

pitch = Pitch(line_zorder=2, line_color='black')

bin_statistic = pitch.bin_statistic([0], [0], statistic='count', bins=(3, 1))
byteImgIO = io.BytesIO()
byteImg = Image.open(os.path.join(os.getcwd(),'Documentacion')+'/instat.png')
byteImg.save(byteImgIO, "PNG")
byteImgIO.seek(0)
byteImg = byteImgIO.read()


# Non test code
dataBytesIO = io.BytesIO(byteImg)
sb_logo = Image.open(dataBytesIO)
#fm = FontManager()


    
def subplotting(ax1,ax2):
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')
    ax[0]=ax1
    ax[1]=ax2
    fig.tight_layout()
    plt.show();
        

def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_df(df, ranges):
    """scales df[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(df[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = df[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdf = [d]
    for d, (y1, y2) in zip(df[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdf.append((d-y1) / (y2-y1) 
                     * (x2 - x1) + x1)
    return sdf

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.8,0.8],polar=True, alpha=0.6,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, 
                                         labels=variables,
                                         size=14,
                                         weight='bold',
                                         c='black')
        # [txt.set_rotation(angle-90) for txt, angle 
        #      in zip(text, angles)]
        for ax in axes[1:]:
            ax.tick_params(axis='both', which='major', pad=15,color='grey')
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
            ax.yaxis.grid(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], 
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) 
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            gridlabel[-1]=""
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i])
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.plot(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='darkmagenta')
    def mplot(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.plot(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='red',alpha=0.55)
    def maxplot(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.plot(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='grey',alpha=0.25)
    def fill(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.fill(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='darkmagenta')
    def mfill(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.fill(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='red')
    def maxfill(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.fill(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='grey')

class DashRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.001,0.22,0.45,0.7],polar=True, alpha=0.6,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, 
                                         labels=variables,
                                         size=12,
                                         weight='bold',
                                         c='black')
        # [txt.set_rotation(angle-90) for txt, angle 
        #      in zip(text, angles)]
        for ax in axes[1:]:
            ax.tick_params(axis='both', which='major', pad=15,color='grey')
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
            ax.yaxis.grid(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], 
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) 
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            gridlabel[-1]=""
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i])
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.plot(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='darkmagenta')
    def mplot(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.plot(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='red',alpha=0.7)
    def maxplot(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.plot(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='grey',alpha=0.25)
    def fill(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.fill(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='darkmagenta')
    def mfill(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.fill(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='red')
    def maxfill(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.fill(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='grey')



    
#Relación entre npxG/sh y CC+/90 minutos para analizar el valor añadido generado en cada oportunidad
#por cada jugador (npxG/s) y la calidad de las oportunidades propiciadas por cada uno (CC+)

col_radar_dict = { 
    'Centre-Back':cbs_radar[:13],
    'Full-Back':flb_radar[:13],
    'Midfielder':mid_radar[:13],
    'Att. Midfield/Winger':attm_radar[:13],
    'Forward':fwd_radar[:13]
    }  

changer_dict = { 
    'Centre-Back':[i[:50] for i in cbs_radar[:13]],
    'Full-Back':[i[:50] for i in flb_radar[:13]],
    'Midfielder':[i[:50] for i in mid_radar[:13]],
    'Att. Midfield/Winger':[i[:50] for i in attm_radar[:13]],
    'Forward':[i[:50] for i in fwd_radar[:13]]
    } 
        

# example data
#rdata = data[data['PosE']!='Goalkeeper']
def sradar(player):
    
    err=0
    rc = players_df[players_df['Nombre']==player]
    pos = rc.iloc[0].PosE
    comp = rc.iloc[0]['league-instat']
    try:
        response = requests.get("{}".format(rc.iloc[0].logo))
        img = Image.open(BytesIO(response.content))
        
    except:
        err+=1
        print('Could not load player photo')
        pass
    
    title = ['Nombre','Team','Edad','league-instat','Nacionalidad','Posición','PosE','Values']
    
    for i in col_radar_dict.keys():
        if i==pos:
            col_radar=col_radar_dict[i]
            changer=changer_dict[i]
    
    rc= rc.set_index(title)[col_radar]
    att = list(rc)
    rc.columns = changer
    
    values = rc.iloc[0].tolist()
    values = tuple(values)
    dmean = players_df[(players_df['PosE']==pos)] 
    if player in list(dmean.Nombre):
        pass
    else:
        dmean=pd.concat([dmean,players_df[players_df['Nombre']==player][title+col_radar]])
    ranges=[]
    for i in att:
        ranges.append((dmean[i].min(),dmean[i].max()))
        
    dmean = dmean[dmean['league-instat']==comp]
    dmean = dmean.set_index(title)[col_radar]
    dmean.columns = changer
    mean = dmean.mean().tolist()
    mean = tuple(mean)
    dmax = dmean.max().tolist()
    dmax= tuple(dmax)
              
    
    fig1 = plt.figure(figsize=(18, 15))

    # RADAR
    radar = ComplexRadar(fig1,changer, ranges)
    radar.plot(values)
    radar.mplot(mean)
    radar.maxplot(dmax)
    plt.style.use('seaborn-white')
    radar.fill(values, alpha=0.5)
    radar.mfill(mean, alpha=0.2)
    radar.maxfill(dmax, alpha=0.05)
    plt.figtext(0.07,0.98,'{}'.format("Radar/  "),size=17,weight='bold',color='dimgrey')
    plt.figtext(0.12,0.98,'{}'.format(rc.index[0][0]),size=24,weight='bold',color='darkmagenta')
    plt.figtext(0.07,0.945,'{} | {} | TMarkt Value: €{}M.\n{} | {} years old | {}'.format(rc.index[0][-3],
                                                    rc.index[0][1],int(rc.index[0][-1]),rc.index[0][-4],
                                                    rc.index[0][2], rc.index[0][3]),
                size=17)
    
    
    plt.figtext(0.72,0.10,'·',color="red",weight='bold',ha='center',size=90,alpha=0.8)
    plt.figtext(0.72,0.07,'·',color="grey",weight='bold',ha='center',size=90,alpha=0.25)
    plt.figtext(0.87,0.125,'Average for {} {}s'.format(rc.index[0][-5],
                                                      rc.index[0][-2]),
                color="black",ha='center',size=14)
    plt.figtext(0.87,0.095,'Max. for {} {}s'.format(rc.index[0][-5],rc.index[0][-2]),
                color="black",ha='center',size=14)
    plt.figtext(0.5,0.04,'Data in per 90 minutes. Defensive metrics adjusted to team possession',ha='center',size=16)
    plt.figtext(0.5,0.02,'{} Season'.format('2021-22'),ha='center',size=16)
    if err==0:
        add_image(img, fig1, left=0, bottom=0.94, width=0.07,height=0.07)
    else:
        pass
    add_image(sb_logo, fig1, left=0.9, bottom=0.02, width=0.07)
    #fig1.savefig(os.path.join(ruta,'Output','Players')+'/'+'Radar_{}'.format(player.replace(' ','_')),dpi=90)
    plt.show()
    
    
def radar_comp(player1,player2):
        
    rc = players_df[players_df['Nombre']==player1]
    rc2 = players_df[players_df['Nombre']==player2]
    pos = rc.iloc[0].PosE
    comp = rc.iloc[0]['league-instat']
    comp = comp.split('.')[-1]
    err=0
    rcc = pd.concat([rc,rc2])
    imgs=[]
    for i in range(2):
        try:
            response = requests.get("{}".format(rcc.iloc[i].logo))
            img = Image.open(BytesIO(response.content))
            imgs.append(img)
        except:
            err+=1
            print('Could not load player photo {}'.format(i+1))
            pass
    title = ['Nombre','Team','Edad','league-instat','Nacionalidad','Posición','PosE','Values']
    
    for i in col_radar_dict.keys():
        if i==pos:
            col_radar=col_radar_dict[i]
            changer=changer_dict[i]
    
    
    
    rc= rc.set_index(title)[col_radar]
    rc2= rc2.set_index(title)[col_radar]
    att = list(rc)
    rc.columns = changer
    rc2.columns = changer
    
    values = rc.iloc[0].tolist()
    
    
    values = tuple(values)
    values2 = rc2.iloc[0].tolist()
    values2 = tuple(values2)
    dmean = players_df[(players_df['PosE']==pos)]
    if player1 in list(dmean.Nombre):
        pass
    else:
        dmean=pd.concat([dmean,players_df[players_df['Nombre']==player1][title+col_radar]])
        
    if player2 in list(dmean.Nombre):
        pass
    else:
        dmean=pd.concat([dmean,players_df[players_df['Nombre']==player2][title+col_radar]])
    ranges=[]
    for i in att:
        ranges.append((dmean[i].min(),dmean[i].max()))
              
    # plotting
    fig1 = plt.figure(figsize=(18, 15))
    
    radar = ComplexRadar(fig1, changer, ranges)
    dmean = dmean.set_index(title)[col_radar]
    dmean.columns = changer
    dmean = dmean.mean().tolist()
    dmax= tuple(dmean)
    radar.plot(values)
    radar.mplot(values2)
    radar.maxplot(dmax)
    plt.style.use('seaborn-white')
    radar.fill(values, alpha=0.4)
    radar.mfill(values2, alpha=0.4)
    radar.maxfill(dmax, alpha=0.25)
    #plt.figtext(0.06,0.98,'{}'.format("Radar/  "),size=14,weight='bold',color='dimgrey')
    plt.figtext(0.25,0.98,'{}'.format(rc.index[0][0]),size=18,weight='bold',color='darkmagenta')
    plt.figtext(0.57,0.98,'{}'.format(rc2.index[0][0]),size=18,weight='bold',color='red')
    plt.figtext(0.25,0.945,'{} | {} | TMarkt Value: €{}M.\n{} | {} years old | {}'.format(rc.index[0][-3],
                                                    rc.index[0][1],int(rc.index[0][-1]),rc.index[0][-4],
                                                    rc.index[0][2],rc.index[0][3]),
                size=11)
    
    plt.figtext(0.57,0.945,'{} | {} | TMarkt Value: €{}M.\n{} | {} years old | {}'.format(rc2.index[0][-3],
                                                    rc2.index[0][1],int(rc2.index[0][-1]),rc2.index[0][-4],
                                                    rc2.index[0][2],rc2.index[0][3]),
                size=11)
    
    
    
    plt.figtext(0.4,0.04,'·',color="grey",weight='bold',ha='center',size=80,alpha=0.5)
    plt.figtext(0.5,0.06,'Average for Selected Leagues {}s'.format(rc.index[0][-2]),
                color="black",ha='center',size=11)
    plt.figtext(0.5,0.04,'Data in per 90 minutes. Defensive metrics adjusted to team possession',ha='center',size=12)
    plt.figtext(0.5,0.02,'{} Season'.format('2021-22'),ha='center',size=12)
    if err==0:
        add_image(imgs[0], fig1, left=0.18, bottom=0.94, width=0.07,height=0.07)
        add_image(imgs[1], fig1, left=0.78, bottom=0.94, width=0.07,height=0.07)
    else:
        pass
    add_image(sb_logo, fig1, left=0.8, bottom=0.04, width=0.07)
    #fig1.savefig(os.path.join(ruta,'Output','Players')+'/'+'Comp_Radar_{}_vs_{}'.format(player1.replace(' ','_'),player2.replace(' ','_')),dpi=90)
    plt.show()

def plot_percentiles(player,team_list):
    err=0
    rc = players_df[players_df['Nombre']==player]
    pos = rc.iloc[0].PosE
    comp = rc.iloc[0]['league-instat']
    try:
        response = requests.get("{}".format(rc.iloc[0].logo))
        img = Image.open(BytesIO(response.content))
        
    except:
        err+=1
        print('Could not load player photo')
        pass
    
    title = ['Nombre','Team','Edad','league-instat','Nacionalidad','Posición','PosE','Values']
    
    for i in col_radar_dict.keys():
        if i==pos:
            col_radar=col_radar_dict[i]
            changer=changer_dict[i]
    
    rc= rc.set_index(title)[col_radar]
    # PERCENTILES
    k= players_df[(players_df['PosE']==pos) & (players_df['Equipo'].isin(team_list))][title+col_radar]
    if player in list(k.Nombre):
        pass
    else:
        k=pd.concat([k,players_df[players_df['Nombre']==player][title+col_radar]])
    k.columns = title+changer
    col_radar_r=[]
    for i in changer:
        #lp[lp['Pos']==i]
        x=i+'pc'
        y=i+'rk'
        k[x] = k[i].rank(pct = True)*100
        k[i+'pc'] = round(k[i+'pc'],0)
        k = k.sort_values(by=i,ascending=False)
        index = range(1,k.shape[0]+1)
        k[i+'rk'] = index
        col_radar_r.append(x)
    kshape=k.shape[0]  
    k=k[k['Nombre']==player]
    k = k.set_index('Nombre').iloc[:,7:]
    k_transposed = k.T
    
    
    my_range=list(range(1,len(k_transposed.loc[col_radar_r].index)+1))
    x=k_transposed.loc[col_radar_r]
    cmap='BuGn'
    values = list(x[player])
    colors=[]
    for i in values:
        if i<10:
            colors.append("#ad0000")
        elif 20>i>=10:
            colors.append("#d73027")
        elif 30>i>=20:
            colors.append("#f46d43")
        elif 40>i>=30:
            colors.append("#fdae61")
        elif 50>i>=40:
            colors.append("#fee08b")
        elif 65>i>=50:
            colors.append("#f5f10a")
        elif 80>i>=65:
            colors.append("#a6d96a")
        elif 95>i>=80:
            colors.append("#10cc1d")
        elif 100>=i>=95:
            colors.append("#1a9850")
    x['colors']=colors
    
    fig2,ax = plt.subplots(figsize=(10,10))
    ax.tick_params(left = False, right = False, top=False)

    sns.barplot(x=player,y=col_radar_r,data=x, palette=colors,alpha=1,edgecolor='white',linewidth=1.5,
                    )
    
    plt.xticks(np.arange(0,100,20))
    
    plt.xlabel('')
    ax.set_title('\n{}, Percentiles vs. {}s ({} players) in selected Leagues\n'.format(player,pos,kshape),size=15,loc='right',weight='bold')
    
    #fig2.set_frame_on(False)
    ax.set_yticklabels(changer)
    for item in plt.gca().yaxis.get_ticklabels():
        item.set_fontsize(14)
        #item.set_fontname('Helvetica')
        #item.set_weight('bold')
    
    
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    #fig2.spines['left'].set_color('none')
    
    #fig2.spines['left'].set_smart_bounds(True)
    #fig2.spines['top'].set_smart_bounds(True)
    rects = ax.patches
    y=[]
    
    for rect,i in zip(rects,col_radar_r):
        j = i[:-2]
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2
        y.append(y_value)
    
        # Number of points between bar and label. Change to your liking.
        space = 3
        # Vertical alignment for positive values
        ha = 'left'
    
        # If value of bar is negative: Place label left of bar
        if x_value < 0:
            # Invert space to place label to the left
            space *= -1
            # Horizontally align label at right
            ha = 'right'
    
        # Use X value as label and format number with one decimal place
        label = "{:.0f} |".format(x_value)
        label2 = "{:.2f} ({:.0f}º)".format(float(k[j]),float(k[j+'rk']))
    
        # Create annotation
        plt.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(space, 0),          # Horizontally shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            va='center',                # Vertically center label
            ha=ha,
            size=14,
            weight='bold')                      # Horizontally align label differently for
                                        # positive and negative values.
                                        
        plt.annotate(
        label2,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space*11, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha,
        size=12,
        )                      # Horizontally align label differently for
                                    # positive and negative values.
    #plt.tight_layout()
    plt.show()
    


