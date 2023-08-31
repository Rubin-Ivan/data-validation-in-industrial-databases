import os
import netCDF4 as nc
import numpy as np
from statistics import mean
from decimal import Decimal
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib.patches as patches

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score
import math
import random
from scipy.spatial.distance import cdist
from numpy import array
from matplotlib import path

def generator (data, k, sred_anom_dlina,sigma_dlina, const_min, const_max, sigma):
    n_total= data.shape[0]*data.shape[1]*data.shape[2]
    n_anom=round(n_total*k)
    N=round(n_anom/sred_anom_dlina)
    vse_dliny = np.random.normal(sred_anom_dlina,sigma_dlina, N)
    const = np.random.uniform(const_min,const_max,N)
    print(vse_dliny.shape)
    print(N)
    print(const.shape)
    
    anomalii_list=[]
    i=0
    for z in vse_dliny:
        chislo_schet=random.randint(1,3)
        if (chislo_schet == 1):
            chislo=random.randint(0, data.shape[0]-round(z)-1)
            constitution=np.full(shape=round(z),fill_value=const[i])
            list_xy=[]
            x=random.randint(0, data.shape[1]-1)
            y=random.randint(0, data.shape[2]-1)
            list_xy.append(x)
            list_xy.append(y)
            anomalii_list.append(list_xy)
            for const1 in constitution:
                data[chislo,x,y]=const1
                chislo=chislo+1
            i=i+1
            
        if (chislo_schet == 2):
            chislo=random.randint(0, data.shape[0]-round(z)-1)
            constitution=np.full(shape=round(z),fill_value=const[i])
            noise = np.random.normal(0,sigma, round(z))
            noise=noise+constitution
            x=random.randint(0, data.shape[1]-1)
            y=random.randint(0, data.shape[2]-1)
            list_xy=[]
            list_xy.append(x)
            list_xy.append(y)
            anomalii_list.append(list_xy)
            for noi in noise:
                data[chislo,x,y]=noi
                chislo=chislo+1
            i=i+1
        
        if (chislo_schet == 3):
            chislo=random.randint(0, data.shape[0]-round(z)-1)
            constitution=np.full(shape=round(z),fill_value=const[i])
            noise = np.random.normal(0,sigma, round(z))
            noise=noise+constitution
            d1=noise
            for i in range(0,chislo):
                d1=np.insert(d1, 0,0)
            for y in range(0,data.shape[0]-chislo-noise.shape[0]):
                d1=np.append(d1, 0)
            
            
            d1 = d1.reshape(data.shape[0],1,1)
            
            x=random.randint(0, data.shape[1]-1)
            y=random.randint(0, data.shape[2]-1)
            list_xy=[]
            list_xy.append(x)
            list_xy.append(y)
            anomalii_list.append(list_xy)
            
            data[:,x,y]=data[:,x,y]+d1[:,0,0]
            
            i=i+1
            
    return data, anomalii_list

