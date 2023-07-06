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
explore.embed()
```