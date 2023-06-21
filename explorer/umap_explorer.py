import os
import re

import sqlite3

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import umap

import matplotlib.pyplot as plt
import seaborn as sns

class UE():
    def __init__(self):
        self.data = None
        self.clusters_names = []
    
    def load_data(self, fileanme, filetype='csv', data_cols = "*", tablen_name='Per_Image', sheet_name='Sheet1'):
        filetypes = ['csv', 'db', 'excel', 'DRUG TREATMENT JOIN']
        if filetype not in filetypes:
            raise ValueError("Invalid file type. Expected one of: %s" % filetypes)
        if filetype=='csv':
            self.data = pd.read_csv(fileanme)
        elif fileanme =='db':
            query = f"SELECT {data_cols} FROM {tablen_name}"
            con = sqlite3.connect(fileanme)
            self.data = pd.read_sql_query(query, con)
            con.close()
        elif fileanme == 'excel':
            self.data = pd.read_excel(fileanme, sheet_name=sheet_name)