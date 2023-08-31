#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import numpy as np
from statistics import mean
from decimal import Decimal
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib.patches as patches
import bamt.networks as Nets
import bamt.preprocessors as pp
import bamt
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from bamt.builders import StructureBuilder
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score
import math
import random
from scipy.spatial.distance import cdist
from numpy import array

from sklearn import preprocessing


from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed
from pyvis.network import Network
from pyitlib import discrete_random_variable as drv

import bamt.builders as builders
import itertools 
from itertools import islice

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from bamt.networks.continuous_bn import ContinuousBN
from bamt.networks.hybrid_bn import HybridBN

def BN_learning (ends):
    f1=[]
    for end in ends:
        ZIMA = pd.read_csv(f'Зима2019.npy{end}')
        VESNA = pd.read_csv(f'Весна2019.npy{end}')
        LETO = pd.read_csv(f'Лето2019.npy{end}')
        OSEN = pd.read_csv(f'Осень2019.npy{end}')

        ZIMA =ZIMA.drop('Unnamed: 0', axis=1)
        VESNA =VESNA.drop('Unnamed: 0', axis=1)
        LETO =LETO.drop('Unnamed: 0', axis=1)
        OSEN =OSEN.drop('Unnamed: 0', axis=1)

        df = pd.concat([ZIMA, VESNA, LETO, OSEN], ignore_index= True)

        df=df.astype({'NORM': 'float64'})

        bn = HybridBN(use_mixture=True, has_logit=True)

        encoder = preprocessing.LabelEncoder()
        discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

        p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
        discretized_data, est = p.apply(df)

        info = p.info

        bn.add_nodes(info)

        structure = [("VALUE", "NORM"),
                ("SEASON", "NORM")]

        bn.set_structure(edges=structure)

        data_train, data_test = train_test_split(df, test_size=0.3, random_state=42)

        target = data_test['NORM'].values

        data_test = data_test.drop(columns=['NORM'])

        bn.fit_parameters(data_train)

        predictions = bn.predict(data_test, 4)

        pred= covert_to_boolean(predictions['NORM'])

        f1.append(f1_score(target,pred, average='macro'))
    return f1

