# scViewer: Scanpy-based Single Cell Analysis Made Easy

**scViewer** is an app that makes Scanpy-based analysis of single cell data easy and accessible. It supports the standard pipeline for single cell data analysis from pre-processing to differential gene expression. It also supports advanced analysis that allows users to analyze their data catered to their questions.

It is available via docker deployment, and also as a .pkg file under Releases for those who don't want to code! If download the .pkg file, make sure to click "Open Anyway" under macOS Privary & Security settings.

---

## Features

- Input & output: single cell data in .h5ad format
- Standard pipeline:
  - Basic filtering, outlier detection, doublet detection, pre-processing, feature selection
  - Linear and non-linear dimensionality reduction
  - Leiden and Louvain clustering
  - Gene expression, clusters, and quality metrics visualization
  - Export Anndata results
- Advanced pipeline:
  - Cell-level manipulation
    - Label clusters
    - Subset and filter cells
    - Subset and filter cells based on gene expressions
    - Annotate cell groups
  - Gene-level manipulation
    - Differential gene expression analysis
    - Co-expression analysis
  - Export AnnData objects and subsets for downstream use

---

## Requirements

- Python 3.10+  
- Docker (recommended for easy deployment)  

---

## Quickstart with Docker

### 1. Build the Docker image

Clone this repo and run:

```bash
docker build -t scviewer .
```

### 2. Run the Docker image
```bash
docker run -p 8501:8501 scviewer
```
### 3. Open the web tool
Open your browser at http://localhost:8501 to use scViewer.

---

## No Docker
If you prefer running locally without Docker:
``` bash
pip install -r requirements.txt
streamlit run app.py --server.maxUploadSize=10000
```






















