# scViewer: Single cell data analysis made easy

**scViewer** is a Streamlit-based web tool for Scanpy-based analysis for single cell data. It supports standard pipeline for single cell data analysis from pre-processing to differential gene expression. It also supports advanced analysis that allows users to analyze their data catered to their questions.

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























