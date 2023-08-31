#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

def anom_koord(files,conf_name):
    for file in files:
        DATA = np.load(file)
        DATA_AN = np.load(f'{file}_anom_{conf_name}.npy')
        MATRIX = np.full((DATA.shape[0],3584 , 2432), True)
        for z in range(0,DATA.shape[0]):
            print(z)
            for y in range(0,DATA.shape[1]):
                for x in range(0,DATA.shape[2]):
                    if (DATA[z,y,x] != DATA_AN[z,y,x]):
                        MATRIX[z,y,x]=False
        np.save(f'{file}_anom_koord_{conf_name}.npy', MATRIX)
        
def dataset_creating(files, seasons,conf_name,sea_cluster_file, sea_name):
    i=0
    for file in files:
        DATA_AN = np.load(f'{file}_anom_{conf_name}.npy')
        KOORD = np.load(f'{file}_anom_koord_{conf_name}.npy')
        MORE_sektor = np.load(sea_cluster)
        list_znach=[]
        list_anom=[]
        for yx in MORE_sektor:
            y=yx[0]
            x=yx[1]
            for z in range(0,DATA_AN.shape[0]):
                list_znach.append(DATA_AN[z,y,x])
                list_anom.append(KOORD[z,y,x])
        listofzeros=[seasons[i]] * len(list_anom)
        df = pd.DataFrame({
        'VALUE':list_znach ,
        'NORM': list_anom,
        'SEASON': listofzeros})


        df.to_csv(f'{file}_{sea_name}_{conf_name}.csv')  
        i=i+1

