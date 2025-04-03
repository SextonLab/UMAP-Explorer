#derive random single-cell crop composites from cluster labels using CV8000 images

import os
import sqlite3
import pandas as pd
from skimage import measure, io
from skimage.morphology import label
import numpy as np

# Connect to your database
conn = sqlite3.connect('BIS_009.db') #change to your data directory

# Fetch data from table with cluster labels
query_objects = """
SELECT a.leiden_cluster, a.ObjectNumber, b.Image_Metadata_WellID, b.Image_Metadata_Field, b.ImageNumber
FROM MyExpt_Per_Object_UMAP a
JOIN MyExpt_Per_Image b ON a.ImageNumber = b.ImageNumber
""" #change UMAP table name and column names accodingly
df = pd.read_sql_query(query_objects, conn)

# Close the connection
conn.close()

# Function to find the diameter of the mask for a given ObjectNumber
def find_diameter(mask_path, object_number):
    mask_image = io.imread(mask_path, as_gray=True)
    labeled_mask = label(mask_image)
    props = measure.regionprops(labeled_mask)
    
    for prop in props:
        if prop.label == object_number:
            return prop.equivalent_diameter
    return None

# Filter cells based on their diameter (remove false masks)
def filter_cells(df, min_diameter=100):
    filtered_rows = []
    
    for index, row in df.iterrows():
        well_id = row['Image_Metadata_WellID']
        field_id = row['Image_Metadata_Field']
        object_number = row['ObjectNumber']
        
        # Format the FieldID with 'F' prefix
        field_id_formatted = f"F{field_id}"
        
        # Get the mask path, adjusting for A03 in the path
        mask_path = f"Y:/CV8000/Sophia/BIS009_20240701_102850/masks/PECCU_{well_id}_T0001{field_id_formatted}L01A03Z01C03.tif" #change to your mask path
        
        diameter = find_diameter(mask_path, object_number)
        if diameter and diameter > min_diameter:
            filtered_rows.append(row)
    
    return pd.DataFrame(filtered_rows)

# Filter the DataFrame to only include cells with diameter > 100 pixels
filtered_df = filter_cells(df, min_diameter=100) #change minimum diameter accordingly

# Randomly sample 5 objects from each cluster#derive random single-cell crop composites from cluster labels using CV8000 images

import os
import sqlite3
import pandas as pd
from skimage import measure, io
from skimage.morphology import label
import numpy as np

# Connect to your database
conn = sqlite3.connect('BIS_009.db') #change to your data directory

# Fetch data from table with cluster labels
query_objects = """
SELECT a.leiden_cluster, a.ObjectNumber, b.Image_Metadata_WellID, b.Image_Metadata_Field, b.ImageNumber
FROM MyExpt_Per_Object_UMAP a
JOIN MyExpt_Per_Image b ON a.ImageNumber = b.ImageNumber
""" #change UMAP table name and column names accodingly
df = pd.read_sql_query(query_objects, conn)

# Close the connection
conn.close()

# Function to find the diameter of the mask for a given ObjectNumber
def find_diameter(mask_path, object_number):
    mask_image = io.imread(mask_path, as_gray=True)
    labeled_mask = label(mask_image)
    props = measure.regionprops(labeled_mask)
    
    for prop in props:
        if prop.label == object_number:
            return prop.equivalent_diameter
    return None

# Filter cells based on their diameter (remove false masks)
def filter_cells(df, min_diameter=100):
    filtered_rows = []
    
    for index, row in df.iterrows():
        well_id = row['Image_Metadata_WellID']
        field_id = row['Image_Metadata_Field']
        object_number = row['ObjectNumber']
        
        # Format the FieldID with 'F' prefix
        field_id_formatted = f"F{field_id}"
        
        # Get the mask path, adjusting for A03 in the path
        mask_path = f"Y:/CV8000/Sophia/BIS009_20240701_102850/masks/PECCU_{well_id}_T0001{field_id_formatted}L01A03Z01C03.tif" #change to your mask path
        
        diameter = find_diameter(mask_path, object_number)
        if diameter and diameter > min_diameter:
            filtered_rows.append(row)
    
    return pd.DataFrame(filtered_rows)

# Filter the DataFrame to only include cells with diameter > 100 pixels
filtered_df = filter_cells(df, min_diameter=100) #change minimum diameter accordingly

# Randomly sample 5 objects from each cluster
def sample_objects(df, cluster_col, n_samples=5):
    sampled_df = df.groupby(cluster_col).apply(lambda x: x.sample(n=n_samples, random_state=1)).reset_index(drop=True)
    return sampled_df

# Sample 5 objects from each cluster
sampled_df = sample_objects(filtered_df, 'leiden_cluster')

# Function to find the centroid of the mask for a given ObjectNumber
def find_centroid(mask_path, object_number):
    mask_image = io.imread(mask_path, as_gray=True)
    labeled_mask = label(mask_image)
    props = measure.regionprops(labeled_mask)
    
    for prop in props:
        if prop.label == object_number:
            centroid_y, centroid_x = prop.centroid
            return int(centroid_x), int(centroid_y)
    return None, None

# Function to create a cropped image with the cell
def create_cropped_image(base_path, well_id, field_id_formatted, object_number, padding=5):
    # Load the images
    dna_path = f"{base_path}{well_id}_T0001{field_id_formatted}L01A01Z01C01.tif"
    cmo_path = f"{base_path}{well_id}_T0001{field_id_formatted}L01A02Z01C02.tif"
    claudin2_path = f"{base_path}{well_id}_T0001{field_id_formatted}L01A03Z01C04.tif"
    
    try:
        dna_image = io.imread(dna_path)
        cmo_image = io.imread(cmo_path)
        claudin2_image = io.imread(claudin2_path)
        
        # Load the mask image to find the bounding box of the object
        mask_path = f"Y:/CV8000/Sophia/BIS009_20240701_102850/masks/PECCU_{well_id}_T0001{field_id_formatted}L01A03Z01C03.tif" #change to your mask path
        mask_image = io.imread(mask_path, as_gray=True)
        labeled_mask = label(mask_image)
        props = measure.regionprops(labeled_mask)
        
        for prop in props:
            if prop.label == object_number:
                minr, minc, maxr, maxc = prop.bbox
                break
        
        # Calculate the crop boundaries with padding
        minr = max(minr - padding, 0)
        minc = max(minc - padding, 0)
        maxr = min(maxr + padding, mask_image.shape[0])
        maxc = min(maxc + padding, mask_image.shape[1])
        
        # Crop the images
        cropped_dna = dna_image[minr:maxr, minc:maxc]
        cropped_cmo = cmo_image[minr:maxr, minc:maxc]
        cropped_claudin2 = claudin2_image[minr:maxr, minc:maxc]
        
        return cropped_dna, cropped_cmo, cropped_claudin2
    except Exception as e:
        print(f"Error creating cropped image: {e}")
        return None, None, None

# Directory to save the cropped images
output_dir = 'output_single_cell_crops'
os.makedirs(output_dir, exist_ok=True)

# Base path for the images
base_path = "Y:/CV8000/Sophia/BIS009_20240701_102850/PECCU/PECCU_"

# Create and save single-cell crops for sampled objects
for index, row in sampled_df.iterrows():
    well_id = row['Image_Metadata_WellID']
    field_id = row['Image_Metadata_Field']
    object_number = row['ObjectNumber']
    
    # Format the FieldID with 'F' prefix
    field_id_formatted = f"F{field_id}"
    
    # Get the mask path, adjusting for A03 in the path
    mask_path = f"Y:/CV8000/Sophia/BIS009_20240701_102850/masks/PECCU_{well_id}_T0001{field_id_formatted}L01A03Z01C03.tif"
    
    # Find the centroid of the cell in the mask
    centroid_x, centroid_y = find_centroid(mask_path, object_number)
    if centroid_x is None or centroid_y is None:
        continue  # Skip this object if centroid calculation failed
    
    # Create the cropped images
    cropped_dna, cropped_cmo, cropped_claudin2 = create_cropped_image(base_path, well_id, field_id_formatted, object_number)
    if cropped_dna is not None and cropped_cmo is not None and cropped_claudin2 is not None:
        # Save the cropped images as a multi-channel TIFF stack
        output_path = os.path.join(output_dir, f"cluster_{row['leiden_cluster']}_object_{index}.tif") #change cluster label accordingly
        io.imsave(output_path, np.stack([cropped_dna, cropped_cmo, cropped_claudin2], axis=0))
        print(f'Saved: {output_path}')
def sample_objects(df, cluster_col, n_samples=5):
    sampled_df = df.groupby(cluster_col).apply(lambda x: x.sample(n=n_samples, random_state=1)).reset_index(drop=True)
    return sampled_df

# Sample 5 objects from each cluster
sampled_df = sample_objects(filtered_df, 'leiden_cluster')

# Function to find the centroid of the mask for a given ObjectNumber
def find_centroid(mask_path, object_number):
    mask_image = io.imread(mask_path, as_gray=True)
    labeled_mask = label(mask_image)
    props = measure.regionprops(labeled_mask)
    
    for prop in props:
        if prop.label == object_number:
            centroid_y, centroid_x = prop.centroid
            return int(centroid_x), int(centroid_y)
    return None, None

# Function to create a cropped image with the cell
def create_cropped_image(base_path, well_id, field_id_formatted, object_number, padding=5):
    # Load the images
    dna_path = f"{base_path}{well_id}_T0001{field_id_formatted}L01A01Z01C01.tif"
    cmo_path = f"{base_path}{well_id}_T0001{field_id_formatted}L01A02Z01C02.tif"
    claudin2_path = f"{base_path}{well_id}_T0001{field_id_formatted}L01A03Z01C04.tif"
    
    try:
        dna_image = io.imread(dna_path)
        cmo_image = io.imread(cmo_path)
        claudin2_image = io.imread(claudin2_path)
        
        # Load the mask image to find the bounding box of the object
        mask_path = f"Y:/CV8000/Sophia/BIS009_20240701_102850/masks/PECCU_{well_id}_T0001{field_id_formatted}L01A03Z01C03.tif" #change to your mask path
        mask_image = io.imread(mask_path, as_gray=True)
        labeled_mask = label(mask_image)
        props = measure.regionprops(labeled_mask)
        
        for prop in props:
            if prop.label == object_number:
                minr, minc, maxr, maxc = prop.bbox
                break
        
        # Calculate the crop boundaries with padding
        minr = max(minr - padding, 0)
        minc = max(minc - padding, 0)
        maxr = min(maxr + padding, mask_image.shape[0])
        maxc = min(maxc + padding, mask_image.shape[1])
        
        # Crop the images
        cropped_dna = dna_image[minr:maxr, minc:maxc]
        cropped_cmo = cmo_image[minr:maxr, minc:maxc]
        cropped_claudin2 = claudin2_image[minr:maxr, minc:maxc]
        
        return cropped_dna, cropped_cmo, cropped_claudin2
    except Exception as e:
        print(f"Error creating cropped image: {e}")
        return None, None, None

# Directory to save the cropped images
output_dir = 'output_single_cell_crops'
os.makedirs(output_dir, exist_ok=True)

# Base path for the images
base_path = "Y:/CV8000/Sophia/BIS009_20240701_102850/PECCU/PECCU_"

# Create and save single-cell crops for sampled objects
for index, row in sampled_df.iterrows():
    well_id = row['Image_Metadata_WellID']
    field_id = row['Image_Metadata_Field']
    object_number = row['ObjectNumber']
    
    # Format the FieldID with 'F' prefix
    field_id_formatted = f"F{field_id}"
    
    # Get the mask path, adjusting for A03 in the path
    mask_path = f"Y:/CV8000/Sophia/BIS009_20240701_102850/masks/PECCU_{well_id}_T0001{field_id_formatted}L01A03Z01C03.tif"
    
    # Find the centroid of the cell in the mask
    centroid_x, centroid_y = find_centroid(mask_path, object_number)
    if centroid_x is None or centroid_y is None:
        continue  # Skip this object if centroid calculation failed
    
    # Create the cropped images
    cropped_dna, cropped_cmo, cropped_claudin2 = create_cropped_image(base_path, well_id, field_id_formatted, object_number)
    if cropped_dna is not None and cropped_cmo is not None and cropped_claudin2 is not None:
        # Save the cropped images as a multi-channel TIFF stack
        output_path = os.path.join(output_dir, f"cluster_{row['leiden_cluster']}_object_{index}.tif") #change cluster label accordingly
        io.imsave(output_path, np.stack([cropped_dna, cropped_cmo, cropped_claudin2], axis=0))
        print(f'Saved: {output_path}')