# Find representative images for UMAP clusters

import os
import numpy as np
import pandas as pd

from skimage import measure, io
from skimage.morphology import label

import glob
import tqdm

def crop_image(chan1, chan2, chan3, xmin, xmax, ymin, ymax):
    crop_dna = chan1[ymin:ymax, xmin:xmax]
    crop_cmo = chan2[ymin:ymax, xmin:xmax]
    crop_target = chan3[ymin:ymax, xmin:xmax]
    return np.stack((crop_dna, crop_cmo, crop_target), axis=0)

def sample_df(df, cluster_col, n=5):
    return df.groupby(cluster_col).apply(lambda x: x.sample(n=n, random_state=1)).reset_index(drop=True)

def get_crops(df, img_dir, output_dir, bbox_col,
              chan1_col, chan2_col, chan3_col, n_sampels=5,
              cluster_col='leiden_cluster',  bbox_area=100,
              bbox_X_min='Cell_AreaShape_BoundingBoxMinimum_X', bbox_X_max='Cell_AreaShape_BoundingBoxMaximum_X', 
              bbox_Y_min='Cell_AreaShape_BoundingBoxMinimum_Y', bbox_Y_max='Cell_AreaShape_BoundingBoxMaximum_Y'):
    dt= df.loc[df[bbox_col]>bbox_area] # filter df to cells with diameter > diam_val
    dt = sample_df(dt, cluster_col, n=n_sampels) # randomly pull n samples from each cluster group
    for ind, row in tqdm.tqdm(dt.iterrows()):
        cluster_id = row[cluster_col]
        
        chan1 = io.imread(os.path.join(img_dir, row[chan1_col][0]))
        chan2 = io.imread(os.path.join(img_dir, row[chan2_col][0]))
        chan3 = io.imread(os.path.join(img_dir, row[chan3_col][0]))
        
        xmin = int(row[bbox_X_min])
        xmax = int(row[bbox_X_max])
        ymin = int(row[bbox_Y_min])
        ymax = int(row[bbox_Y_max])
        
        crop_img = crop_image(chan1, chan2, chan3, xmin, xmax, ymin, ymax)
        fname = os.path.join(output_dir, f"cluster_{cluster_id}_object_{ind}_.tif")
        io.imsave(fname, crop_img)
