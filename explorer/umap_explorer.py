import os
import re

import sqlite3

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score
from sklearn.cluster import KMeans

import umap
import xgboost as xgb
import shap

import hdbscan

import igraph as ig
import leidenalg as la

import matplotlib.pyplot as plt
import seaborn as sns

from .find_clusters import get_crops

class UE():
    def __init__(self,df=None, data_cols="*", embedder=None, cluster_labels=None):
        self.df = df
        self.data_cols = data_cols
        self.embedder = embedder
        self.cluster_labes = cluster_labels
        self.model = xgb.XGBRegressor()
    
    def view_tables(self, dbfile):
        con = sqlite3.connect(dbfile)
        cursor = con.cursor()
        cursor.execute('SELECT name FROM sqlite_master WHERE type="table";')
        print(cursor.fetchall())
        con.close()

    def load_data(self, filename, filetype='csv', data_cols = "*", table_name='Per_Image', sheet_name='Sheet1'):
        filetypes = ['csv', 'db', 'excel', 'DRUG TREATMENT JOIN']
        if filetype not in filetypes:
            raise ValueError("Invalid file type. Expected one of: %s" % filetypes)
        if filetype=='csv':
            self.df = pd.read_csv(filename)
        elif filetype =='db':
            query = f"SELECT {data_cols} FROM {table_name}"
            print(query)
            self.data_cols = data_cols
            con = sqlite3.connect(filename)
            self.df = pd.read_sql_query(query, con)
            con.close()
        elif filename == 'excel':
            self.df = pd.read_excel(filename, sheet_name=sheet_name)
            
    def get_data_columns(self, print_cols=False, dtype:str="float64", extra=[]):
        pattern = "ImageNumber|Location|Center|ExecutionTime|Parent|Child|Metadata|Scaling"
        if len(extra) > 0:
            pattern+= "|".join(extra)
        meta_cols = self.df.columns[self.df.columns.str.contains(pat=pattern, flags=re.IGNORECASE)].tolist()
        self.data_cols = self.df.drop(columns=meta_cols).select_dtypes(include=dtype).columns.tolist()
        if print_cols:
            for c in self.data_cols:
                print(c)
    
    def join_meta(self, meta, left_on=[], right_on=[]):
        print('joining metadata')
        self.df = pd.merge(self.df, meta, left_on=left_on, right_on=right_on, how='left')

    def export(self, filename:str):
        if filename.endswith('csv'):
            filename = filename+".csv"
        self.df.to_csv(filename, index=False)
    
    def export_db(self, db, tablename, if_exist='fail'):
        con = sqlite3.connect(db)
        self.df.to_sql(tablename,  con=con, if_exists=if_exist,)
        con.close()
    
    def embed(self, a=None, b=None, n_neighbors=15, min_dist=0.1, metric='euclidean'):
        self.embedder = umap.UMAP(
            n_neighbors=n_neighbors,
            metric=metric,
            min_dist=min_dist,
            a=a, b=b,
            random_state=69
            )
        self.df[self.data_cols].fillna(value=self.df[self.data_cols].mean(), inplace=True)
        scaled = StandardScaler().fit_transform(self.df[self.data_cols])
        self.df[['umap_1','umap_2']] = self.embedder.fit_transform(scaled)
        
    def plot(self, x='umap_1', y='umap_2', color_on='cond', save=None, fname='my_plot'):
        ftypes = [None, 'svg', 'png']
        if save not in ftypes:
            raise ValueError("Invalid file type. Expected one of: %s" % ftypes)
        ax = sns.scatterplot(data=self.df, x=x, y=y, hue=color_on)
        if color_on is not None:
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
    
    def cluster(self, cluster_type='leiden', min_clusters=5, resolution_parameter=None):
        types = ['hdbscan', 'leiden', 'kmeans']
        if cluster_type not in types:
            raise ValueError("Invaild cluster type, Expected one of: %s" % types)
        if cluster_type == 'hdbscan':
            cluster_algo = hdbscan.HDBSCAN(min_cluster_size=min_clusters, gen_min_span_tree=True)
            cluster_algo.fit(self.df[['umap_1','umap_2']])
            self.df['cluster'] = cluster_algo.labels_
            self.cluster_labels = self.df['cluster'].unique().tolist()
            self.cluster_labels.sort()
        elif cluster_type == 'leiden':
            data = self.df[['umap_1','umap_2']].values
            dist_matrix = np.sqrt((data[:, 0, None] - data[:, 0])**2 + (data[:, 1, None] - data[:, 1])**2)
            graph = ig.Graph.Adjacency((dist_matrix < 1).tolist())
            if resolution_parameter is None:
                 partition = la.find_partition(graph, la.ModularityVertexPartition, )
            else:
                partition = la.find_partition(graph, la.CPMVertexPartition, resolution_parameter=resolution_parameter)
            self.df['cluster'] = partition.membership
            self.cluster_labels = self.df['cluster'].unique().tolist()
            self.cluster_labels.sort()
        elif cluster_type == 'kmeans':
            data = self.df[['umap_1', 'umap_2']].values
            kmeans = KMeans(n_clusters=min_clusters, random_state=42)
            self.df['cluster'] = kmeans.fit_transform(data)
            
        silhoute = silhouette_score(self.df[['umap_1', 'umap_2']], self.df['cluster'])
        print(f"Cluster Silhoutte Score: {silhoute}")
        print("Best value is 1, 0 indicates overlapping clusters, Negative values indicates poorly assigned clusters")
        print()
        
    
    def gen_model(self, cluster_1, cluster_2):
        # convert to list if single cluster ids
        if type(cluster_1) is not list: cluster_1 = [cluster_1]
        if type(cluster_2) is not list: cluster_2 = [cluster_2]
        
        # if cluster_2 value is rest, use all labels not in cluster_1
        if 'rest' in cluster_2:
            cluster_2 = list(set(self.cluster_labels)-set(cluster_1))
        
        clus_1 = [clust in self.cluster_labels for clust in cluster_1]
        clus_2 = [clust in self.cluster_labels for clust in cluster_2]
        
        if not all(clus_1) or not all(clus_2):
            raise ValueError("Invalid clusters, Expected two clusters of: %s" % self.cluster_labels)
        scaler = StandardScaler()
        dt = self.df.loc[self.df['cluster'].isin(cluster_1+cluster_2)]
        dt['label'] = 0.0
        dt.loc[dt['cluster'].isin(cluster_1), 'label'] = 1.0
        X = scaler.fit_transform(dt[self.data_cols])
        y = dt.label.values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
            )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        print(f"Model R2 Score: {r2_score(y_test, preds):.2f}")
        print(f"Model MSE: {mean_squared_error(y_test, preds):.2f}")
        # maybe put something here to accept or change cluster
        scaled = scaler.fit_transform(self.df[self.data_cols])
        self.df['score'] = self.model.predict(scaled)
    
    def get_shaps(self, cluster_1, cluster_2, max_display=10):
        # get shap top features
        # need teh clusters to get the same data as before
        explainer = shap.TreeExplainer(self.model)
        dt = self.df.loc[self.df['cluster'].isin(cluster_1+cluster_2)]
        shap_values = explainer.shap_values(dt[self.data_cols])
        shap.summary_plot(shap_values, dt[self.data_cols], max_display=max_display)
    
    def cluster_violins(self, column, savefig=False, fname=None, hue=None):
        plt.clf()
        ax = sns.violinplot(data=self.df, x='cluster', y=column, hue=hue, split=True)
        if hue != None:
            sns.move_legend(ax, 'upper left',  bbox_to_anchor=(1, 1))
        if savefig:
            if fname == None:
                fname = f"clustre_{column}.png"'cluster'
            plt.savefig(fname, format='png', bbox_inches='tight')
            
    def get_cluster_imgs(self, img_dir, mask_dir, output_dir, diam_col, diam_val=100, obj_num_col='ObjectNumber',
              well_col='Image_Metadata_WellID', field_col='Image_Metadata_Field',
              chan1='C01', chan2='C02', chan3='C03', mask_chan='C01'):
        get_crops(self.df, img_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir, 
                  dima_col=diam_col, diam_val=diam_val, obj_num_col=obj_num_col,
              well_col=well_col, field_col=field_col,
              chan1=chan1, chan2=chan2, chan3=chan3, mask_chan=mask_chan)