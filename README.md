# scViewer: Interactive Single-Cell Data Visualization App

**scViewer** is a Streamlit-based web app for interactive analysis and visualization of single-cell RNA-seq data using Scanpy and AnnData.  
It supports cell annotation, cluster visualization, and flexible export of processed data.

---

## Features

- Visualize clusters (UMAP, tSNE) with dynamic selection  
- Annotate cells based on gene expression thresholds  
- Visualize annotated boolean cell subsets  
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























