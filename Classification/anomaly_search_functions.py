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
from pgmpy.estimators import K2Score
from bamt.builders import StructureBuilder
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score
import math
import random
from scipy.spatial.distance import cdist
from numpy import array
from pgmpy.utils import get_example_model
from pgmpy.estimators import PC
from pgmpy.estimators import K2Score
from sklearn import preprocessing
from pgmpy.estimators import K2Score


from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed
from pyvis.network import Network
from pyitlib import discrete_random_variable as drv
from pgmpy.estimators import K2Score
import bamt.builders as builders
import itertools 
from itertools import islice

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


import numpy as np
from bamt.networks.continuous_bn import ContinuousBN
from bamt.networks.hybrid_bn import HybridBN

def bn_learning (data_train):
    
    bn = HybridBN(use_mixture=True, has_logit=True)

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data_train)

    info = p.info

    bn.add_nodes(info)

    structure = [("VALUE", "NORM"),
            ("SEASON", "NORM")]

    bn.set_structure(edges=structure)

    bn.fit_parameters(data_train)
    
    return (bn)

def anom_search_func(data,bn,a):
    
    anom_label = bn.predict(data, 4)
    anom_label= [bool(x) for x in anom_label['NORM']]
    return anom_label

