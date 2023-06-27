import os
import re

import sqlite3

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import umap

import hdbscan

import igraph as ig
import leidenalg as la

import matplotlib.pyplot as plt
import seaborn as sns

class UE():
    def __init__(self):
        self.df = pd.DataFrame
        self.data_cols = "*"
        self.embedder = None
    
    def load_data(self, fileanme, filetype='csv', data_cols = "*", tablen_name='Per_Image', sheet_name='Sheet1'):
        filetypes = ['csv', 'db', 'excel', 'DRUG TREATMENT JOIN']
        if filetype not in filetypes:
            raise ValueError("Invalid file type. Expected one of: %s" % filetypes)
        if filetype=='csv':
            self.df = pd.read_csv(fileanme)
        elif fileanme =='db':
            query = f"SELECT {data_cols} FROM {tablen_name}"
            self.data_cols = data_cols
            con = sqlite3.connect(fileanme)
            self.df = pd.read_sql_query(query, con)
            con.close()
        elif fileanme == 'excel':
            self.df = pd.read_excel(fileanme, sheet_name=sheet_name)
            
    def get_data_columns(self, print_cols=False, dtype:str="float64", extra=list):
        pattern = "ImageNumber|Location|Center|Execution_Time|Parent|Child|Metadata"
        if len(extra) > 0:
            pattern+= "|".join(extra)
        meta_cols = self.df.columns[self.df.columns.str.contains(pat=pattern, flags=re.IGNORECASE)].tolist()
        self.data_cols = self.df.drop(columns=meta_cols).select_dtypes(include=dtype).columns.tolist()
        if print_cols:
            print(self.data_cols)
    
    def export(self, filename:str):
        if filename.endswith('csv'):
            filename = filename+".csv"
        self.df.to_csv(filename, index=False)
    
    def export_db(self, db, tablename, if_exist='fail'):
        con = sqlite3.connect(db)
        self.df.to_sql(tablename, self.df, con=con, if_exists=if_exist,)
        con.close()
    
    def embed(self, a=None, b=None, n_neighbors=15, min_dist=0.1, metric='eucliean'):
        self.embedder = umap.UMAP(
            n_neighbors=n_neighbors,
            metric=metric,
            min_dist=min_dist,
            a=a, b=b,
            random_state=69
            )
        scaled = StandardScaler().fit_transform(self.df[self.data_cols])
        self.df[['x','y']] = self.embedder.fit_transform(scaled)
        
    def plot(self, x='x', y='y', color_on='cond', save=None, fname='my_plot'):
        ftypes = [None, 'svg', 'png']
        if save not in ftypes:
            raise ValueError("Invalid file type. Expected one of: %s" % ftypes)
        ax = sns.scatterplot(data=self.df, x=x, y=y, hue=color_on)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        if save:
            if save == 'both':
                plt.savefig(".".join((fname,'svg')), format='svg')
                plt.savefig(".".join((fname, 'png')), format='png')
            else:
                plt.savefig(".".join((fname, save)), format=save)
    
    def head(self):
        return self.df.head()
    
    def shape(self):
        return self.df.shape
    
    def cluster(self, type='hdbscan', min_clusters=5):
        types = ['hdbscan', 'leiden']
        if type not in types:
            raise ValueError("Invaild cluster type, Expected one of: %s" % types)
        if type == 'hdbscan':
            cluster_algo = hdbscan.HDBSCAN(min_cluster_size=min_clusters, gen_min_span_tree=True)
            cluster_algo.fit(self.df[['x','y']])
            self.df['cluster'] = cluster_algo.labels_
        elif type == 'leiden':
            data = self.df[['x','y']]
            dist_matrix = np.sqrt((data[:, 0, None] - data[:, 0])**2 + (data[:, 1, None] - data[:, 1])**2)
            graph = ig.Graph.Adjacency((dist_matrix < 1).tolist())
            partition = la.find_partition(graph, la.ModularityVertexPartition)
            self.df['cluster'] = partition.membership
