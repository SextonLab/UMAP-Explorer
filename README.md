# UMAP-Explorer

UMAP-Explorer is a comprehensive tool designed for high-dimensional cell morphology data analysis. Leveraging the power of UMAP (Uniform Manifold Approximation and Projection) for dimension reduction, this package provides a streamlined workflow for transforming complex morphological data into intuitive, two-dimensional representations. Further functionalities, including advanced clustering and detailed analytical tools, facilitate the extraction of meaningful insights from Cell Profiler data.

![UMAP-Explorer Logo](./images/umap2.jpg)
## Features

* **UMAP Embedding:** Reduce the dimensionality of your high-dimensional cell morphology data, making it easier to visualize and interpret.

* **Clustering:** Apply clustering algorithms - HDBScan and Leiden clustering - to identify distinct groups or patterns in your data.

* **Analysis Tools:** Use our analytical tools to delve deeper into the characteristics and structure of your data including cluster comparison by XGBoost/SHAP analysis, cluster extraction and re-embedding.

## Installation

```
git clone git@github.com:SextonLab/UMAP-Explorer.git

cd UMAP-Explorer

pip install -e .
```

### Optional install directly from GitHub
`pip install git+https://github.com/yourusername/UMAP-Explorer.git`

## Running

Example Workflow, notebook can be found in `notebooks`

```
import explorer as ue

explore = ue.UE()

explore.load_data('My_Expt.db', type='db', table_name='Per_Image')
explore.get_data_columns()
explore.embed(n_neighbors=25, min_dist=0.3)

```

## Function Guide

### `load_data`

Loads data from source file into UMAP Explorer

**Parameters**

- filename - name of measurement data file
- *Optional*: filetype - default: `csv`
    - csv, db, execl, DRUG TREATMENT JOIN
- *Optional*: data_cols - default: "*"
    - List of columns used during db or sqlite file reads to reduce RAM load
- *Optional*: table_name - default: "Per_Image"
    - Table name of database to load
- *Optional*: sheet_name - default: "Sheet1"
    - For reading in excel files, reads specific excel sheet

### `get_data_columns`

Generates list of data columns seperated from all columns of dataset

**Parameters**

- *Optional*: print_cols - default: False
    - Prints list of data columns for checking
- *Optional*: dtype - default: 'float64'
    - Data columns data type, 'float64' is the default from cellprofiler
- *Optional*: extra - default: []
    - List of columns to include in removal from data columns

### `export`

Exports UMAP dataframe to csv

**Parameters**

- filename - name of csv file without ".csv" extension

### `export_db`

Exports to db or sqlite file as a new table

**Parameters**

- db - Database filename
- tablename - New Table name
- *Optional*: if_exist - default: 'fail'
    - What to do if the new table already exists

### `embed`

Embed the UMAP, passing standard parameters from umap-learn

**Parameters**

- *Optional*: a - Alpha value - default: None
- *Optional*: b - Beta value - default: None
- *Optional*: n_neighbors - default: 15
- *Optional*: min_dist - default: 0.1
- *Optional*: metric - default: 'euclidean'

### `plot`

Plot the UMAP and color and allow for saving

**Parameters**

- *Optional*: x - UMAP x value - default: 'x'
- *Optional*: y - UMAP y value - default : 'y'
- *Optional*: color_on - Column to color points on - default: 'cond'
- *Optional*: save - Save file - default: None
    - Options: None, both, png, svg
- *Optional*: fname - name of plot - default: 'my_plot'


### `head`

Returns the dataframe's head call

### `shape`

Returns the shape of the dataframe

### `cluster`

Applies either hdbscan or leiden clustering to the UMAP's xy-coordinates

**Parameters**

- type - default: "leiden"
- *Optional*: min_clusters - default:5
    - min_cluster_size for hdbscan

### `gen_model`

Generates an XGBoost Regressor model between 2 cluster groups

**Parameters**

- cluster_1
    - Single or list of cluster ids to use as "1.0" label
- cluster_2
    - Single or list of cluster ids to use as "0.0" label
    - *Optional*: 'rest' this value uses all clusters **not** in cluster_1