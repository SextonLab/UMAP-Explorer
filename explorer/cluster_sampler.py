# Find representative images for UMAP clusters

import os
import sqlite3
import numpy as np
import pandas as pd

from skimage import measure, io
from skimage.morphology import label

import glob
import tqdm

def get_img_path(mask_path, well_id, field_id, channel, plate_type='*'):
    field_id = f"F{field_id}"
    fname = f"{plate_type}_{well_id}_*{field_id}*{channel}.tif"
    return glob.glob(os.path.join(mask_path, fname))[0]

def crop_image(dna_image, cmo_image, target_image, mask_image, object_number, padding=5):
    # crops single image
    # given well_id field_id object_number find 
    # load composite parts of image i.e dna, cmo, some third channel to make an RGB tif image
    # load in mask image
    # use label func to get 
    
    mask_label = label(mask_image)
    props = measure.regionprops(mask_label)
    for prop in props:
        if prop.label == object_number:
            minr, minc, maxr, maxc = prop.bbox
            break
    
    minr = max(minr - padding, 0)
    minc = max(minc - padding, 0)
    maxr = min(maxr + padding, mask_image.shape[0])
    maxc = min(maxc + padding, mask_image.shape[1])
    
    
    crop_dna = dna_image[minr:maxr, minc:maxc]
    crop_cmo = cmo_image[minr:maxr, minc:maxc]
    crop_target = target_image[minr:maxr, minc:maxc]
    
    return np.stack([crop_dna, crop_cmo, crop_target], axis=0)

def sample_df(df, cluster_col, n=5):
    return df.groupby(cluster_col).apply(lambda x: x.sample(n=n, random_state=1)).reset_index(drop=True)

def get_crops(df, img_dir, mask_dir, output_dir, diam_col, 
              diam_val=100, cluster_col='leiden_cluster', obj_num_col='ObjectNumber',
              well_col='Image_Metadata_WellID', field_col='Image_Metadata_Field',
              chan1='C01', chan2='C02', chan3='C03', mask_chan='C01'):
    dt= df.loc[df[diam_col]>diam_val] # filter df to cells with diameter > diam_val
    dt = sample_df(dt, cluster_col) # randomly pull n samples from each cluster group
    for ind, row in tqdm.tqdm(dt.iterrows()):
        wid = row[well_col]
        fid = row[field_col]
        obj_num = row[obj_num_col]
        
        img_c1 = io.imread(get_img_path(mask_path=img_dir, well_id=wid, field_id=fid, channel=chan1))
        img_c2 = io.imread(get_img_path(mask_path=img_dir, well_id=wid, field_id=fid, channel=chan2))
        img_c3 = io.imread(get_img_path(mask_path=img_dir, well_id=wid, field_id=fid, channel=chan3))
        mask = io.imread(get_img_path(mask_path=mask_dir, well_id=wid, field_id=fid, channel=mask_chan))
        crop_img = crop_image(img_c1, img_c2, img_c3, mask,obj_num)
        fname = os.path.join(output_dir, f"cluster_{row[cluster_col]}_object_{ind}.tif")
        io.imsave(fname, crop_img)