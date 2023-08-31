
import os
import netCDF4 as nc
import numpy as np
from statistics import mean
from decimal import Decimal
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib.patches as patches
import bamt.Networks as Nets
import bamt.Preprocessors as pp
import bamt
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from pgmpy.estimators import K2Score
from bamt.Builders import StructureBuilder
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score
import math

def sektory(data, land):
    
    def kolichestvo_ed_v_sektore(n, array):
        kol_ed = np.count_nonzero(array)
        kol_ed_v_sektore = kol_ed // n
        return (kol_ed_v_sektore)

    koled50 = kolichestvo_ed_v_sektore(50, land)

    N=50
    i=0
    list_granic50 = list(range(1, N))
    sektory= np.split(land, list_granic50 ,axis=0)
    while i<N-1:
        while np.count_nonzero(sektory[i]) +600 < koled50:
            dop = list(range(0, N-i-1))
            for element in dop:
                list_granic50[i+element] += 1

                sektory= np.split(land,list_granic50 ,axis=0)

        i=i+1       

    zima_sektory= np.split(season_zima, list_granic50 ,axis=1)

    sektory_srednee=[]
    w=0
    while w<N:
        sektor_zima=zima_sektory[w]
        i=0
        while (i<data.shape[2]):
            q = sektor_zima[i, :, :]
            sektory_srednee.append(np.mean(q))
            i=i+1
        w=w+1

    df50 = pd.DataFrame(np.array(sektory_srednee).reshape(N,data.shape[2])), columns = list(range(0, data.shape[2]))))

    df50 = df50.T 

    df50 = df50.add_prefix('s')

    return df50


def mark (bn,uzel):
    
    mark_list_parents=[]
    mark_list_children=[]
    mark_list_chp=[]
    mark_list=[]
    for edge in bn.edges:
        if (edge[1]==uzel):
            mark_list_parents.append(edge) 
            mark_list.append(edge) 
        if (edge[0]==uzel):
            mark_list_children.append(edge)  
    
    for child_edge in mark_list_children:
        for edge1 in bn.edges:
            if (edge1[1]==child_edge[1]): 
                mark_list_chp.append(edge1)  
                mark_list.append(edge1) 
    
    #mark_list_parents, mark_list_children,mark_list_chp
    
    return mark_list


def comparasion (original,tested):
    sovpalo=[]
    for edge in tested:
        for or_edge in original:
            if (edge[0]==or_edge[0]) and (edge[1]==or_edge[1]):
                sovpalo.append(edge)
    if len(original) ==0:
        value_sorig=1
    else:
        value_sorig=len(sovpalo)/len(original)  
    if len(tested) ==0:
        value_stest=1   
    else:
        value_stest=len(sovpalo)/len(tested)  
    
    
    return(value_sorig,value_stest)
                
    
def child_dict(net: list):
    res_dict = dict()
    for e0, e1 in net:
        if e1 in res_dict:
            res_dict[e1].append(e0)
        else:
            res_dict[e1] = [e0]
    return res_dict

def precision_recall(pred_net: list, true_net: list, decimal = 4):
    pred_dict = child_dict(pred_net)
    true_dict = child_dict(true_net)
    corr_undir = 0
    corr_dir = 0
    for e0, e1 in pred_net:
        flag = True
        if e1 in true_dict:
            if e0 in true_dict[e1]:
                corr_undir += 1
                corr_dir += 1
                flag = False
        if (e0 in true_dict) and flag:
            if e1 in true_dict[e0]:
                corr_undir += 1
    pred_len = len(pred_net)
    true_len = len(true_net)
    shd = pred_len + true_len - corr_undir - corr_dir
    
    #return {'AP': round(corr_undir/pred_len, decimal),
            #'AR': round(corr_undir/true_len, decimal),
           # 'F1_undir':round(2*(corr_undir/pred_len)*(corr_undir/true_len)/(corr_undir/pred_len+corr_undir/true_len), decimal),
           # 'AHP': round(corr_dir/pred_len, decimal),
          #  'AHR': round(corr_dir/true_len, decimal),
#            'F1_directed': round(2*(corr_dir/pred_len)*(corr_dir/true_len)/(corr_dir/pred_len+corr_dir/true_len), decimal),
          #  'SHD': shd}

    return shd


def experement(df, procent, znachenie, sektor):
    
    df1=df.copy()
    
    kol_cells=round(procent*df50.shape[0])
    
    print(kol_cells)
    
    #for i in range(0,kol_cells):
        #df1.at[i, sektor] = znachenie
    df1.loc[:kol_cells, sektor] *=znachenie

    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    p = pp.Preprocessor([('discretizer', discretizer)])
    discretized_data , est = p.apply(df1)

    info = p.info
    
    bn_d = Nets.ContinuousBN(use_mixture=True)
    
    bn_d.add_nodes(descriptor=info)
    bn_d.add_edges(data=discretized_data, scoring_function=('K2',K2Score))
    
    

    mark_orig= mark (bn_d_orig,sektor)
    mark_test= mark (bn_d,sektor)
    
    value_1, value_2= comparasion (mark_orig,mark_test)
    
    shd=precision_recall(mark_test,mark_orig)
    
    
    
    return value_1, shd


def experement2(df, procent, znachenie, sektor):
    
    df1=df.copy()
    
    kol_cells=round(procent*df50.shape[0])
    
    
    print(kol_cells)
    
    #for i in range(0,kol_cells):
        #df1.at[i, sektor] = znachenie
    df1.loc[:kol_cells, sektor] *=znachenie
    
    df1=df1.round(6)

    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    p = pp.Preprocessor([('discretizer', discretizer)])
    discretized_data , est = p.apply(df1)

    info = p.info
    
    bn_d = Nets.ContinuousBN(use_mixture=True)
    
    bn_d.add_nodes(descriptor=info)
    bn_d.add_edges(data=discretized_data, scoring_function=('K2',K2Score))
    
    

    mark_orig= mark (bn_d_orig,sektor)
    mark_test= mark (bn_d,sektor)
    
    value_1, value_2= comparasion (mark_orig,mark_test)
    
    shd=precision_recall(mark_test,mark_orig)
    
    
    list_vse_uzly=[]
    list_vse_uzly_shd=[]
    
    for i in range(0,len(bn_d_orig.nodes)):
        
        mark_orig111= mark (bn_d_orig,f's{i}')
        mark_test111= mark (bn_d,f's{i}')
        
        value_111, value_222= comparasion (mark_orig111,mark_test111)
        list_vse_uzly.append(value_111)
        shd111=precision_recall(mark_test111,mark_orig111)
        list_vse_uzly_shd.append(shd111)
    
    return value_1, shd,list_vse_uzly,list_vse_uzly_shd

