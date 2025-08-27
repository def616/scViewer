import streamlit as st
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import seaborn as sns
import scrublet as scr
from io import StringIO
import io
import os
import tempfile
import zipfile
import anndata as ad
import warnings
import igraph
import leidenalg
import altair as alt
import re
import random
import gzip
import psutil
import json
import time
import gc
import igraph
import leidenalg
import requests
import operator as op
from scipy.stats import median_abs_deviation
import matplotlib as mpl

warnings.filterwarnings('ignore')
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['figure.facecolor'] = 'white'

random.seed(0)

# Configure scanpy
sc.settings.verbosity = 3  # verbosity level

# Page config
st.set_page_config(
    page_title="scViewer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'adata' not in st.session_state:
    st.session_state.adata = None
if 'analysis_step' not in st.session_state:
    st.session_state.analysis_step = 0

# Title
st.markdown('<h1 class="main-header">scViewer: Scanpy-based Single-cell Analysis Made Easy</h1>', unsafe_allow_html=True)

# Sidebar for navigation
mode = st.sidebar.radio("Choose mode:", ["Standard Pipeline", "Advanced Analysis"], key="mode_selector")
# st.sidebar.title("Standard Analysis Pipeline")
pipeline_steps = [
    "Data Loading",
    'Basic Filtering',
    "Basic Quality Control",
    'Outlier Detection',
    'Doublet Detection',
    'Post-Quality Control Visualization',
    "Normalization & Logarithmization",
    "Feature Selection",
    'Adjustment & Scaling',
    "Dimensionality Reduction",
    "Clustering",
    "Differential Gene Expression",
    "Visualization",
    "Export Results"
]

# selected_step = st.sidebar.radio("Select Analysis Step:", pipeline_steps)

advanced_steps = [
    "Data Loading", 
    "Cell-level Manipulation",
    "Gene-level Manipulation",
    'Visualization',
    "Export Results"
]

# Step selection (conditionally rendered)
if mode == "Standard Pipeline":
    selected_step = st.sidebar.radio("Standard Analysis Step:", options=pipeline_steps, key="standard_step")
elif mode == "Advanced Analysis":
    selected_step = st.sidebar.radio("Advanced Analysis Step:", options=advanced_steps, key="advanced_step")

# Shared AnnData object across modes
adata = st.session_state.get("adata", None)

# helper functions
def display_data_summary(adata):
    """Display basic information about the dataset"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cells", adata.n_obs)
    with col2:
        st.metric("Genes", adata.n_vars)
    with col3:
        if 'sample' in adata.obs.columns:
            counts = adata.obs["sample"].value_counts()
            for sample, count in counts.items():
                st.metric(f"Cells ({sample})", int(count))

def display_data_summary_advanced(adata):
    """Display basic information about the dataset -- advanced analysis"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cells", adata.n_obs)
    with col2:
        st.metric("Genes", adata.n_vars)
    with col3:
        if 'total_counts' in adata.obs.columns:
            st.metric('Total Counts', adata.obs['total_counts'].sum())


def plot_qc_metrics(adata):
    """Plot quality control metrics"""
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # Total counts per cell
    sc.pl.violin(adata, ['total_counts'], ax=axes[0,0], jitter=0.4)
    axes[0,0].set_title('Total Counts per Cell')
    
    # Number of genes per cell
    sc.pl.violin(adata, ['n_genes_by_counts'], ax=axes[0,1], jitter=0.4)
    axes[0,1].set_title('Number of Genes per Cell')
    
    # Mitochondrial gene percentage
    if 'pct_counts_mt' in adata.obs.columns:
        sc.pl.violin(adata, ['pct_counts_mt'], ax=axes[1,0], jitter=0.4)
        axes[1,0].set_title('Mitochondrial Gene %')

    # hemoglobin gene percentage
    if 'pct_counts_hb' in adata.obs.columns:
        sc.pl.violin(adata, ['pct_counts_hb'], ax=axes[1,1], jitter=0.4)
        axes[1,1].set_title('Hemoglobin Gene %')
    
    # hemoglobin gene percentage
    if 'pct_counts_ribo' in adata.obs.columns:
        sc.pl.violin(adata, ['pct_counts_ribo'], ax=axes[2,0], jitter=0.4)
        axes[2,0].set_title('Ribosomal Gene %')

    if all(col in adata.obs.columns for col in ["total_counts", "n_genes_by_counts", "pct_counts_mt"]):
        ax = axes[2,1]
        x = adata.obs["total_counts"]
        y = adata.obs["n_genes_by_counts"]
        c = adata.obs["pct_counts_mt"]

        sc_plot = ax.scatter(x, y, c=c, cmap='viridis', s=10)
        cbar = fig.colorbar(sc_plot, ax=ax)
        cbar.set_label('pct_counts_mt')
        ax.set_title('Total Counts vs Genes (Colored by Mito %)')

    
    plt.tight_layout()
    return fig

def plot_other_qc_metrics(adata, qc_source="obs", metric=None, plot_type="Violin Plot"):
    """Plot a selected QC metric with the specified plot type."""

    if metric is None:
        return None

    # Extract metric data
    data = adata.obs[metric] if qc_source == "obs" else adata.var[metric]
    df = pd.DataFrame({metric: data})

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))

    if plot_type == "Violin Plot":
        sc.pl.violin(adata, keys=metric)
    elif plot_type == "Scatter Plot":
        sc.pl.scatter(adata, key=metric)
    else:
        ax.text(0.5, 0.5, "Invalid plot type selected.", ha='center')

    ax.set_title(f"{plot_type} of `{metric}`")
    return fig

def findOutliers(adata, metric, nmads):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
            np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier

def check_system_resources():
    """Check available system resources before processing"""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    if memory.percent > 85:
        return False, f"High memory usage ({memory.percent:.1f}%). Please close other applications."
    
    if available_gb < 2:
        return False, f"Low available memory ({available_gb:.1f}GB). Need at least 2GB free."
    
    return True, f"Memory OK ({memory.percent:.1f}% used, {available_gb:.1f}GB available)"

def get_file_size_info(uploaded_file):
    """Get file size and provide memory estimation"""
    file_size_mb = uploaded_file.size / (1024 * 1024)
    # H5AD files typically expand 2-4x in memory
    estimated_memory_mb = file_size_mb * 3
    
    return file_size_mb, estimated_memory_mb

def load_h5ad_safely(uploaded_file, show_progress=True):
    """Safely load H5AD file with progress tracking and memory management"""
    
    tmp_file_path = None
    
    try:
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Preparing file for loading...")
        
        # Create temporary file
        fd, tmp_file_path = tempfile.mkstemp(suffix='.h5ad')
        
        # Write uploaded file content to temporary file
        with os.fdopen(fd, 'wb') as tmp_file:
            file_content = uploaded_file.getvalue()
            tmp_file.write(file_content)
        
        if show_progress:
            progress_bar.progress(0.5)
            status_text.text("Loading H5AD file...")
        
        # Load the H5AD file using scanpy
        adata = sc.read_h5ad(tmp_file_path)
        
        if show_progress:
            progress_bar.progress(1.0)
            status_text.text("Successfully loaded!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
        
        return adata, None
        
    except Exception as e:
        return None, str(e)
    finally:
        # Clean up temp file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

# content based on selected step
if mode == "Standard Pipeline" and selected_step == "Data Loading":
    st.markdown('<h2 class="step-header">Step 1: Data Loading</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>What we're doing:</strong> Loading your single-cell RNA-seq data. Supported format: H5AD
    </div>
    """, unsafe_allow_html=True)
    
    # System resource check
    resource_ok, resource_msg = check_system_resources()
    
    if not resource_ok:
        st.error(f"{resource_msg}")
        st.info("**Tips to free up memory:**\n- Close other applications\n")
    else:
        st.success(f"{resource_msg}")
    
    # File upload with size checking
    uploaded_file = st.file_uploader(
        "Upload your single-cell data file:",
        type=['h5ad'],
        help="H5AD files larger than 1GB may take several minutes to load"
    )
    
    if uploaded_file is not None:
        # Get file size info
        file_size_mb, est_memory_mb = get_file_size_info(uploaded_file)
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**File Size:** {file_size_mb:.1f} MB")
        with col2:
            st.info(f"**Est. Memory:** {est_memory_mb:.1f} MB")
        with col3:
            load_time_est = max(10, file_size_mb / 50)  # Rough estimate
            st.info(f"⏱**Est. Load Time:** {load_time_est:.0f}s")
        
        # Warning for very large files
        if file_size_mb > 1000:  # 1GB
            st.warning("""
            **Large File Detected**
            - This file is quite large and may take several minutes to load
            """)
            
            proceed = st.button("Proceed with Loading (Large File)", type="primary")
        else:
            proceed = st.button("Load File", type="primary")
        
        if proceed:
            # Final resource check before loading
            resource_ok, resource_msg = check_system_resources()
            
            if not resource_ok:
                st.error(f"Cannot proceed: {resource_msg}")
                st.stop()
            
            # Load file with error handling
            with st.spinner(f"Loading {uploaded_file.name}... This may take a few minutes for large files."):
                try:
                    # Clear any existing data to free memory
                    if 'adata' in st.session_state:
                        del st.session_state.adata
                        gc.collect()
                    
                    # Load the file
                    adata, error = load_h5ad_safely(uploaded_file, show_progress=True)
                    
                    if error:
                        st.error(f"Error loading file: {error}")
                        st.info("""
                        **Troubleshooting tips:**
                        - Ensure the file is a valid H5AD format
                        - Try closing other applications to free memory
                        - For very large files, consider using a machine with more RAM
                        """)
                    else:
                        # Successfully loaded
                        st.session_state.adata = adata
                        st.session_state.dataset_overview = True
                        st.success(f"Successfully loaded {uploaded_file.name}")
                        
                        # Show current memory usage
                        current_memory = psutil.virtual_memory().percent
                        st.info(f"Current memory usage: {current_memory:.1f}%")
                        
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
                    st.info("If this error persists, try restarting the application.")
    
    # Display summary only once - after upload or when returning to tab
    if st.session_state.get('dataset_overview', False) and 'adata' in st.session_state:

        adata = st.session_state.adata.copy()
        adata.var_names = [gene.lower() for gene in adata.var_names]

        st.subheader("Dataset Overview")
        display_data_summary(adata)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("First 10 Cells")
            df = adata.obs.head(10).copy()
            for col in df.select_dtypes(include=bool).columns:
                df[col] = df[col].astype(str)
            st.dataframe(df)
        with col2:
            st.subheader("First 10 Genes")
            df = adata.var.head(10).copy()
            for col in df.select_dtypes(include=bool).columns:
                df[col] = df[col].astype(str)
            st.dataframe(df)


elif mode == 'Standard Pipeline' and selected_step == 'Basic Filtering':
    if st.session_state.adata is None:
        st.warning("Please load data first.")
    else:
        st.markdown('<h2 class="step-header">Step 2: Basic Filtering </h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <strong>What we’re doing:</strong> This step filters the desired minimum cell and genes.
        </div>
        """, unsafe_allow_html=True)

        adata = st.session_state.adata.copy()

        # filtering
        st.subheader("Filtering Based on Counts")
        col1, col2 = st.columns(2)
        with col1:
            min_genes = st.number_input("Min genes per cell", min_value=0, value=200)
        with col2:
            min_cells = st.number_input("Min cells per gene", min_value=0, value=3)

        filtering_done = False

        if st.button("Apply Filtering"):
            with st.spinner("Filtering cells and genes..."):
                n_cells_before = adata.n_obs
                n_genes_before = adata.n_vars

                sc.pp.filter_cells(adata, min_genes=min_genes)
                sc.pp.filter_genes(adata, min_cells=min_cells)

                st.success("Filtering complete.")
                st.info(f"Cells: {n_cells_before} → {adata.n_obs}")
                st.info(f"Genes: {n_genes_before} → {adata.n_vars}")

                st.session_state.adata = adata
                st.session_state.basic_filtering = True

                filtering_done = True
        
        if filtering_done:
            st.subheader("Filtered Dataset Summary")
            display_data_summary(st.session_state.adata)

        elif st.session_state.get('basic_filtering', False) and 'adata' in st.session_state:
            st.subheader('Filtered Dataset Summary')
            display_data_summary(st.session_state.adata)


elif mode == "Standard Pipeline" and selected_step == "Basic Quality Control":
    if st.session_state.adata is None:
        st.warning("Please load data first.")
    else:
        st.markdown('<h2 class="step-header">Step 3: Basic Quality Control </h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <strong>What we’re doing:</strong> This step calculates standard QC metrics, optionally annotates mitochondrial, hemoglobin, 
                    and ribosomal genes, and allows you to specify the top N genes to include in the QC metrics.
        """, unsafe_allow_html=True)

        adata = st.session_state.adata.copy()

        species = st.selectbox('Species', ['Mouse', 'Human'])
        mito_gene = st.checkbox('Annotate mitochondrial genes', value=True)
        hemo_gene = st.checkbox('Annotate hemoglobin genes', value=False)
        ribo_gene = st.checkbox('Annotate ribosomal genes', value=False)

        percent_top = st.text_input('Top N genes for QC metric (leave blank for none):', value='20')

        if percent_top.strip() == "":
            percent_top = None
        else:
            try:
                percent_top = int(percent_top)
            except ValueError:
                st.error("Please enter an integer or leave blank.")
                percent_top = None
        
        # plot helper functions
        def highest_expr_genes_plot(adata, n_top):
            sc.pl.highest_expr_genes(adata, n_top=n_top, show=False)
            fig = plt.gcf()  
            st.pyplot(fig)

        # Define and show QC violin plots
        def preQCplots(adata):
            groupby = "sample" if "sample" in adata.obs.columns else None
            
            # Base metrics that are always present
            base_metrics = ["total_counts", "n_genes_by_counts"]
            
            # Check which percentage metrics exist
            pct_metrics = []
            if 'pct_counts_mt' in adata.obs.columns:
                pct_metrics.append('pct_counts_mt')
            if 'pct_counts_hb' in adata.obs.columns:
                pct_metrics.append('pct_counts_hb')
            if 'pct_counts_ribo' in adata.obs.columns:
                pct_metrics.append('pct_counts_ribo')
            
            # Combine all available metrics
            all_metrics = base_metrics + pct_metrics
            num_metrics = len(all_metrics)
            
            if num_metrics < 3:
                # Only basic metrics (total_counts, n_genes_by_counts)
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                # Total counts per cell
                sc.pl.violin(adata, ['total_counts'], groupby=groupby, ax=axes[0], jitter=0.4)
                axes[0].set_title('Total Counts per Cell')
                
                # Number of genes per cell
                sc.pl.violin(adata, ['n_genes_by_counts'], groupby=groupby, ax=axes[1], jitter=0.4)
                axes[1].set_title('Number of Genes per Cell')
                
            elif num_metrics == 3:
                # Basic metrics + one percentage metric
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Total counts per cell
                sc.pl.violin(adata, ['total_counts'], groupby=groupby, ax=axes[0], jitter=0.4)
                axes[0].set_title('Total Counts per Cell')
                
                # Number of genes per cell
                sc.pl.violin(adata, ['n_genes_by_counts'], groupby=groupby, ax=axes[1], jitter=0.4)
                axes[1].set_title('Number of Genes per Cell')
                
                # The one available percentage metric
                pct_metric = pct_metrics[0]
                sc.pl.violin(adata, [pct_metric], groupby=groupby, ax=axes[2], jitter=0.4)
                
                # Set appropriate title based on which metric it is
                if pct_metric == 'pct_counts_mt':
                    axes[2].set_title('Mitochondrial Gene %')
                elif pct_metric == 'pct_counts_hb':
                    axes[2].set_title('Hemoglobin Gene %')
                elif pct_metric == 'pct_counts_ribo':
                    axes[2].set_title('Ribosomal Gene %')
                
            else:  # num_metrics > 3
                # Basic metrics + multiple percentage metrics
                # Create a grid that can accommodate all metrics
                if num_metrics <= 4:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    axes = axes.flatten()
                else:  # num_metrics == 5
                    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                    axes = axes.flatten()
                
                # Plot basic metrics
                sc.pl.violin(adata, ['total_counts'], groupby=groupby, ax=axes[0], jitter=0.4)
                axes[0].set_title('Total Counts per Cell')
                
                sc.pl.violin(adata, ['n_genes_by_counts'], groupby=groupby, ax=axes[1], jitter=0.4)
                axes[1].set_title('Number of Genes per Cell')
                
                # Plot percentage metrics
                for i, pct_metric in enumerate(pct_metrics, start=2):
                    sc.pl.violin(adata, [pct_metric], groupby=groupby, ax=axes[i], jitter=0.4)
                    
                    if pct_metric == 'pct_counts_mt':
                        axes[i].set_title('Mitochondrial Gene %')
                    elif pct_metric == 'pct_counts_hb':
                        axes[i].set_title('Hemoglobin Gene %')
                    elif pct_metric == 'pct_counts_ribo':
                        axes[i].set_title('Ribosomal Gene %')
                
                # Hide unused subplots if any
                for j in range(num_metrics, len(axes)):
                    axes[j].set_visible(False)
            
            plt.tight_layout()
            return fig
        
        qc_calculated = False

        if st.button("Calculate QC Metrics"):
            with st.spinner("Running QC..."):
                adata.var_names_make_unique()
                mito_prefix = 'mt-' if species == 'Mouse' else 'MT-'
                adata.var['mt'] = adata.var_names.str.startswith(mito_prefix)

                if ribo_gene:
                    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
                if hemo_gene:
                    if species == 'Mouse':
                        adata.var["hb"] = adata.var_names.str.contains(r"^Hb[ab]")
                    else:
                        adata.var["hb"] = adata.var_names.str.contains(r"^HB[AB]")

                qc_vars = ['mt']
                if ribo_gene: qc_vars.append('ribo')
                if hemo_gene: qc_vars.append('hb')

                sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, 
                                           percent_top=[percent_top] if percent_top is not None else None,
                                           log1p=True, 
                                           inplace=True)

                st.session_state.adata = adata
                st.session_state.qc_metrics_calculated = True
                qc_calculated = True
                st.success("QC metrics calculated.")

        # Show plots if QC was just calculated OR if previously calculated (persist)
        if qc_calculated or st.session_state.get("qc_metrics_calculated", False):
            if "n_genes_by_counts" in st.session_state.adata.obs.columns:
                st.subheader(f"Top {percent_top} Highest Expressed Genes")
                highest_expr_genes_plot(st.session_state.adata, percent_top)

                st.subheader("Violin Plots of QC Metrics")
                # st.pyplot(preQCplots(st.session_state.adata))

                try:
                    col1, col2, col3 = st.columns(3)
                    groupby = "sample" if "sample" in adata.obs.columns else None
                    
                    with col1:
                        # total count per cell
                        fig1, ax1 = plt.subplots(figsize=(6, 4))
                        sc.pl.violin(adata, ['total_counts'], groupby=groupby, ax=ax1, show=False)
                        ax1.set_title('Total Counts per Cell')
                        st.pyplot(fig1)
                        plt.close(fig1)


                    with col2:
                        # number of genes per cell
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        sc.pl.violin(adata, ['n_genes_by_counts'], groupby=groupby, ax=ax2, show=False)
                        ax2.set_title('Number of Genes per Cell')
                        st.pyplot(fig2)
                        plt.close(fig2)

                    with col3:
                        # MT % 
                        fig3, ax3 = plt.subplots(figsize=(6, 4))
                        sc.pl.violin(adata, ['pct_counts_mt'], groupby=groupby, ax=ax3, show=False)
                        ax3.set_title('Mitochondrial Gene % per Sample')
                        st.pyplot(fig3)
                        plt.close(fig3)
                    
                    col4, col5, col6= st.columns(3)

                    with col4:
                        # HB %
                        if 'pct_counts_hb' in adata.obs.columns:
                            fig4, ax4 = plt.subplots(figsize=(6, 4))
                            sc.pl.violin(adata, ['pct_counts_hb'], groupby=groupby, ax=ax4, show=False)
                            ax4.set_title('Hemoglobin Gene % per Sample')
                            st.pyplot(fig4)
                            plt.close(fig4)
                        else:
                            None
                    
                    with col5:
                        # RB %
                        rb = [col for col in adata.obs.columns if col.startswith('pct_counts_ribo') or col.startswith('pct_counts_rb')]
                        if any(col in adata.obs.columns for col in ['pct_counts_ribo', 'pct_counts_rb']):
                            fig5, ax5 = plt.subplots(figsize=(6, 4))
                            sc.pl.violin(adata, rb, groupby=groupby, ax=ax5, show=False)
                            ax5.set_title('Ribosomal Gene % per Sample')
                            st.pyplot(fig5)
                            plt.close(fig5)
                        else:
                            None
                    
                    with col6:
                        st.empty()

                except Exception as e:
                    st.error(f"Error creating plots: {str(e)}")
                    st.write('Please calculate QC metrics first to view the plots.')


elif mode == 'Standard Pipeline' and selected_step == 'Outlier Detection':
    if st.session_state.adata is None:
        st.warning("Please load data first.")
    else:
        st.markdown('<h2 class="step-header">Step 4: Outlier Detection </h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Identifies and removes poor-quality cells from your single-cell data based on unusual patterns like high mitochondrial 
                    gene expression or extreme cell counts. Users can customize the filtering criteria and see before/after comparisons to ensure only healthy 
                    cells are kept for analysis. 
        </div>
        """, unsafe_allow_html=True)

        adata = st.session_state.adata.copy()
        
        def findOutliers(adata, metric, nmads):
            """Find outliers based on median absolute deviation (MAD).
            Handles both single metrics (str) and lists of metrics.
            """
            if isinstance(metric, list):
                outlier = np.zeros(adata.n_obs, dtype=bool)
                for m in metric:
                    if m in adata.obs.columns:
                        M = adata.obs[m]
                        outlier_m = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
                            M > np.median(M) + nmads * median_abs_deviation(M)
                        )
                        outlier |= outlier_m
                return outlier
            else:
                if metric not in adata.obs.columns:
                    return np.zeros(adata.n_obs, dtype=bool)
                M = adata.obs[metric]
                outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
                    M > np.median(M) + nmads * median_abs_deviation(M)
                )
                return outlier
        
        def find_pct_top_col(adata):
            pattern = r"^pct_counts_in_top_\d+_genes$"
            cols = [col for col in adata.obs.columns if re.match(pattern, col)]
            return cols

        def filterOutliers(adata, mt_filter, hb_filter, rb_filter,
                           mt_threshold_dict=None, hb_threshold_dict=None, rb_threshold_dict=None,
                           mt_nmads=3, hb_nmads=3, rb_nmads=3, min_cells=None):
            
            # find the pct_counts_in_top_20_genes column, '20' can change
            percent_top = find_pct_top_col(adata)

            # Basic outlier detection
            basic_outliers = np.zeros(adata.n_obs, dtype=bool)
            
            # Check if required columns exist before using them
            if "log1p_total_counts" in adata.obs.columns:
                basic_outliers = basic_outliers | findOutliers(adata, "log1p_total_counts", 5)
            if "log1p_n_genes_by_counts" in adata.obs.columns:
                basic_outliers = basic_outliers | findOutliers(adata, "log1p_n_genes_by_counts", 5)
            if percent_top:  # Only if we have percent_top columns
                basic_outliers = basic_outliers | findOutliers(adata, percent_top, 5)
            
            adata.obs["outlier"] = basic_outliers
            
            # Mitochondrial filtering
            if mt_filter and "pct_counts_mt" in adata.obs.columns:
                if mt_threshold_dict is not None:
                    mt_outliers = np.zeros(adata.n_obs, dtype=bool)
                    if 'sample' in adata.obs.columns:
                        samples = adata.obs['sample'].unique().tolist()
                        for sample in samples:
                            thresh = mt_threshold_dict.get(sample, None)
                            if thresh is not None:
                                mt_outliers_sample = (adata.obs['pct_counts_mt'] > thresh) & (adata.obs['sample'] == sample)
                                mt_outliers = mt_outliers | mt_outliers_sample
                    adata.obs["mt_outlier"] = mt_outliers
                else:
                    mt_outliers = findOutliers(adata, "pct_counts_mt", mt_nmads) | (adata.obs["pct_counts_mt"] > 8)
                    adata.obs["mt_outlier"] = mt_outliers
            else:
                adata.obs["mt_outlier"] = np.zeros(adata.n_obs, dtype=bool)

            # Hemoglobin filtering
            if hb_filter and "pct_counts_hb" in adata.obs.columns:
                if hb_threshold_dict is not None:
                    hb_outliers = np.zeros(adata.n_obs, dtype=bool)
                    if 'sample' in adata.obs.columns:
                        samples = adata.obs['sample'].unique().tolist()
                        for sample in samples:
                            thresh = hb_threshold_dict.get(sample, None)
                            if thresh is not None:
                                hb_outliers_sample = (adata.obs['pct_counts_hb'] > thresh) & (adata.obs['sample'] == sample)
                                hb_outliers = hb_outliers | hb_outliers_sample
                    adata.obs["hb_outlier"] = hb_outliers
                else:
                    adata.obs["hb_outlier"] = findOutliers(adata, "pct_counts_hb", hb_nmads)
            else:
                adata.obs["hb_outlier"] = np.zeros(adata.n_obs, dtype=bool)

            # Ribosomal filtering
            if rb_filter and "pct_counts_ribo" in adata.obs.columns:
                if rb_threshold_dict is not None:
                    rb_outliers = np.zeros(adata.n_obs, dtype=bool)
                    if 'sample' in adata.obs.columns:
                        samples = adata.obs['sample'].unique().tolist()
                        for sample in samples:
                            thresh = rb_threshold_dict.get(sample, None)
                            if thresh is not None:
                                rb_outliers_sample = (adata.obs['pct_counts_ribo'] > thresh) & (adata.obs['sample'] == sample)
                                rb_outliers = rb_outliers | rb_outliers_sample
                    adata.obs["rb_outlier"] = rb_outliers
                else:
                    adata.obs["rb_outlier"] = findOutliers(adata, "pct_counts_ribo", rb_nmads)
            else:
                adata.obs["rb_outlier"] = np.zeros(adata.n_obs, dtype=bool)

            # Combine all outliers
            combined_outliers = (
                adata.obs["outlier"]
                | adata.obs["mt_outlier"]
                | adata.obs["hb_outlier"]
                | adata.obs["rb_outlier"]
            )

            # Filter the data
            adata_filtered = adata[~combined_outliers].copy()

            # filter out genes:
            sc.pp.filter_genes(adata_filtered, min_cells=min_cells)

            return adata_filtered

        # UI controls
        st.subheader("Filtering Options")
        mt_filter = st.checkbox("Mitochondrial gene filtering", value=True)
        hb_filter = st.checkbox("Hemoglobin gene filtering", value=False)
        rb_filter = st.checkbox("Ribosomal gene filtering", value=False)

        # Initialize variables to avoid UnboundLocalError
        mt_nmads = 3
        hb_nmads = 3
        rb_nmads = 3

        # Mitochondrial thresholds
        use_mt_thresholds = False
        mt_threshold_dict = None
        if mt_filter:
            use_mt_thresholds = st.checkbox("Use sample-specific mitochondrial % thresholds", value=False)
            if use_mt_thresholds and 'sample' in adata.obs.columns:
                mt_threshold_dict = {}
                st.write("**Sample-specific MT thresholds:**")
                for sample in adata.obs['sample'].unique():
                    mt_threshold_dict[sample] = st.number_input(
                        f"MT threshold for sample {sample} (%)",
                        min_value=0.0, max_value=100.0, value=8.0,
                        key=f"mt_thresh_{sample}"
                    )
            elif use_mt_thresholds and 'sample' not in adata.obs.columns:
                st.warning("Sample column not found. Using global threshold instead.")
                use_mt_thresholds = False
            
            if not use_mt_thresholds:
                mt_nmads = st.number_input("Mitochondrial MAD threshold (nmads):", min_value=1, max_value=10, value=3)

        # Hemoglobin thresholds
        use_hb_thresholds = False
        hb_threshold_dict = None
        if hb_filter:
            use_hb_thresholds = st.checkbox("Use sample-specific hemoglobin % thresholds", value=False)
            if use_hb_thresholds and 'sample' in adata.obs.columns:
                hb_threshold_dict = {}
                st.write("**Sample-specific HB thresholds:**")
                for sample in adata.obs['sample'].unique():
                    hb_threshold_dict[sample] = st.number_input(
                        f"HB threshold for sample {sample} (%)",
                        min_value=0.0, max_value=100.0, value=5.0,
                        key=f"hb_thresh_{sample}"
                    )
            elif use_hb_thresholds and 'sample' not in adata.obs.columns:
                st.warning("Sample column not found. Using global threshold instead.")
                use_hb_thresholds = False
            
            if not use_hb_thresholds:
                hb_nmads = st.number_input("Hemoglobin MAD threshold (nmads):", min_value=1, max_value=10, value=3)

        # Ribosomal thresholds
        use_rb_thresholds = False
        rb_threshold_dict = None
        if rb_filter:
            use_rb_thresholds = st.checkbox("Use sample-specific ribosomal % thresholds", value=False)
            if use_rb_thresholds and 'sample' in adata.obs.columns:
                rb_threshold_dict = {}
                st.write("**Sample-specific RB thresholds:**")
                for sample in adata.obs['sample'].unique():
                    rb_threshold_dict[sample] = st.number_input(
                        f"RB threshold for sample {sample} (%)",
                        min_value=0.0, max_value=100.0, value=20.0,
                        key=f"rb_thresh_{sample}"
                    )
            elif use_rb_thresholds and 'sample' not in adata.obs.columns:
                st.warning("Sample column not found. Using global threshold instead.")
                use_rb_thresholds = False
            
            if not use_rb_thresholds:
                rb_nmads = st.number_input("Ribosomal MAD threshold (nmads):", min_value=1, max_value=10, value=3)

        filter_gene = st.checkbox('Filter genes:', value=True)

        if filter_gene: 
            min_cells = st.number_input('Mininum cells per gene:', min_value=1, max_value=adata.n_vars)

        # Run outlier detection
        if st.button("Run Outlier Detection"):
            with st.spinner('Detecting outliers...'):
                try:
                    filtered_adata = filterOutliers(
                        adata,
                        mt_filter=mt_filter,
                        hb_filter=hb_filter,
                        rb_filter=rb_filter,
                        mt_threshold_dict=mt_threshold_dict if use_mt_thresholds else None,
                        hb_threshold_dict=hb_threshold_dict if use_hb_thresholds else None,
                        rb_threshold_dict=rb_threshold_dict if use_rb_thresholds else None,
                        mt_nmads=mt_nmads,
                        hb_nmads=hb_nmads,
                        rb_nmads=rb_nmads,
                        min_cells=min_cells
                    )

                    st.session_state.filtered_adata = filtered_adata
                    st.session_state.outlier_detection_done = True

                    st.success(f"Outlier filtering complete! Removed {adata.n_obs - filtered_adata.n_obs} cells.")
                
                except Exception as e:
                    st.error(f"Error during outlier detection: {str(e)}")
                    st.write("Please check your data and parameters.")

        # Show results and plots if already done
        if st.session_state.get("outlier_detection_done", False):
            filtered_adata = st.session_state.filtered_adata

            st.subheader("Filtering Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cells", f"{filtered_adata.n_obs:,}", f"{filtered_adata.n_obs - adata.n_obs:,}")
                st.metric("Genes", f"{filtered_adata.n_vars:,}", f"{filtered_adata.n_vars - adata.n_vars:,}")
            with col2:
                cells_removed_pct = ((adata.n_obs - filtered_adata.n_obs) / adata.n_obs) * 100
                st.write(f"**Cells removed:** {adata.n_obs - filtered_adata.n_obs:,} ({cells_removed_pct:.1f}%)")
                st.write(f"**Cells retained:** {filtered_adata.n_obs:,} ({100-cells_removed_pct:.1f}%)")

            st.subheader("QC Plots After Filtering")

            # Only create plots if required columns exist
            try:
                col1, col2, col3 = st.columns(3)
                groupby = "sample" if "sample" in adata.obs.columns else None
                
                with col1:
                    # total count per cell
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    sc.pl.violin(filtered_adata, ['total_counts'], groupby=groupby, ax=ax1, show=False)
                    ax1.set_title('Total Counts per Cell')
                    st.pyplot(fig1)
                    plt.close(fig1)


                with col2:
                    # number of genes per cell
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    sc.pl.violin(filtered_adata, ['n_genes_by_counts'], groupby=groupby, ax=ax2, show=False)
                    ax2.set_title('Number of Genes per Cell')
                    st.pyplot(fig2)
                    plt.close(fig2)

                with col3:
                    # MT % 
                    fig3, ax3 = plt.subplots(figsize=(6, 4))
                    sc.pl.violin(filtered_adata, ['pct_counts_mt'], groupby=groupby, ax=ax3, show=False)
                    ax3.set_title('Mitochondrial Gene % per Sample')
                    st.pyplot(fig3)
                    plt.close(fig3)
                
                col4, col5, col6= st.columns(3)

                with col4:
                    # HB %
                    if 'pct_counts_hb' in filtered_adata.obs.columns:
                        fig4, ax4 = plt.subplots(figsize=(6, 4))
                        sc.pl.violin(filtered_adata, ['pct_counts_hb'], groupby=groupby, ax=ax4, show=False)
                        ax4.set_title('Hemoglobin Gene % per Sample')
                        st.pyplot(fig4)
                        plt.close(fig4)
                    else:
                        None
                
                with col5:
                    # RB %
                    rb = [col for col in filtered_adata.obs.columns if col.startswith('pct_counts_ribo') or col.startswith('pct_counts_rb')]
                    if any(col in filtered_adata.obs.columns for col in ['pct_counts_ribo', 'pct_counts_rb']):
                        fig5, ax5 = plt.subplots(figsize=(6, 4))
                        sc.pl.violin(filtered_adata, rb, groupby=groupby, ax=ax5, show=False)
                        ax5.set_title('Ribosomal Gene % per Sample')
                        st.pyplot(fig5)
                        plt.close(fig5)
                    else:
                        None
                
                with col6:
                    st.empty()
                    
            except Exception as e:
                st.error(f"Error creating plots: {str(e)}")
                st.write("Plots could not be generated. Please check your data.")


elif mode == "Standard Pipeline" and selected_step == 'Doublet Detection':
    if st.session_state.adata is None:
        st.warning("Please load data first.")
    else:
        st.markdown('<h2 class="step-header">Step 5: Doublet Detection </h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Detect and remove doublet cells per sample.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.filtered_adata.copy()
        
        # Initialize session state variables
        if 'doublet_detected' not in st.session_state:
            st.session_state.doublet_detected = False
        if 'scrublet_inst' not in st.session_state:
            st.session_state.scrublet_inst = None
        if 'doublet_summary_stats' not in st.session_state:
            st.session_state.doublet_summary_stats = None
        if 'doublet_histograms' not in st.session_state:
            st.session_state.doublet_histograms = None
            
        # users input target cells recovered based on technology
        st.subheader('Target Cells Recovered:')
        min_cells = st.number_input("Minimum recovered cells:", min_value=1, value=500, step=1)
        max_cells = st.number_input("Maximum recovered cells:", min_value=min_cells, value=10000, step=1)

        # starts at 500, 1000 then increments by 1000
        counts_cells_recovered = [min_cells]
        current = min_cells if min_cells > 1000 else 1000
        while current <= max_cells:
            counts_cells_recovered.append(current)
            current += 1000
        if counts_cells_recovered[-1] != max_cells:
            counts_cells_recovered.append(max_cells)
        
        # users input multiplet percentage per target cells recovered
        st.markdown("### Enter expected multiplet rate (%) for each recovered cell count:")
        multiplet_rates = {}
        for count in counts_cells_recovered:
            multiplet_rates[count] = st.number_input(
                f"{count} cells", min_value=0.0, max_value=100.0, value=0.0, step=0.1
            )
        st.markdown("### Your inputs:")
        st.write(multiplet_rates)

        def get_expected_multiplets_rate(cell_count, min_cells, max_cells, multiplet_rates):
            m = (multiplet_rates[max_cells] - multiplet_rates[min_cells]) / (max_cells - min_cells)
            b = multiplet_rates[min_cells] - m * min_cells
            expected_rate = m * cell_count + b
            return expected_rate / 100.0

        # detect doublets
        if st.button('Detect doublet cells'):
            with st.spinner('Detecting doublets...'):
                # Get per-sample cell counts
                sample_multiplets_dict = {}
                for sample in adata.obs['sample'].unique():
                    n_cells = adata.obs[adata.obs['sample'] == sample].shape[0]
                    # interpolate expected multiplet rate for this sample
                    expected_rate = get_expected_multiplets_rate(
                        n_cells,
                        min_cells,
                        max_cells,
                        multiplet_rates
                    )
                    sample_multiplets_dict[sample] = expected_rate

                # run scrublet per sample
                samples = adata.obs['sample'].unique().tolist()
                scrublet_dict = dict()
                adata_dict = dict()
                histogram_dict = dict()  # Store histograms

                for sample in samples:
                    # initialize scrublet object per sample
                    scrub = scr.Scrublet(
                        adata[adata.obs['sample'] == sample].X,
                        expected_doublet_rate=sample_multiplets_dict[sample],
                        random_state=0
                    )
                    scrublet_dict[sample] = scrub

                    # subset anndata for the sample
                    sample_adata = adata[adata.obs['sample'] == sample].copy()

                     # run doublet detection
                    sample_adata.obs['doublet_scores'], sample_adata.obs['predicted_doublets'] = scrub.scrub_doublets(
                        min_counts=2,
                        min_cells=3,
                        min_gene_variability_pctl=85,
                        n_prin_comps=30
                    )

                    # Generate and store histogram
                    fig, ax = scrub.plot_histogram()
                    histogram_dict[sample] = fig

                    # save per sample adata
                    adata_dict[sample] = sample_adata

                # concatenate adata objects
                samples = list(adata_dict.keys())
                adata_scrubbed = adata_dict[samples[0]].concatenate(
                                    *[adata_dict[s] for s in samples[1:]], join="outer"
                                )
                
                # save counts in .X as a layer
                adata_scrubbed.layers["counts_postqc"] = adata_scrubbed.X
                
                # Generate summary statistics
                summary_stats = {}
                for sample in adata_scrubbed.obs['sample'].unique():
                    doublet_sum = int(
                        adata_scrubbed.obs.loc[
                            adata_scrubbed.obs['sample'] == sample, 'predicted_doublets'
                        ].sum()
                    )
                    total = (adata_scrubbed.obs['sample'] == sample).sum()
                    summary_stats[sample] = {
                        'doublets': doublet_sum,
                        'total': total,
                        'percentage': 100 * doublet_sum / total
                    }
                
                # Save state
                st.session_state.adata = adata_scrubbed
                st.session_state.scrublet_inst = scrublet_dict
                st.session_state.doublet_detected = True
                st.session_state.doublet_summary_stats = summary_stats
                st.session_state.doublet_histograms = histogram_dict
                st.success("Doublet detection completed.")
        
        # display results if already detected
        if st.session_state.doublet_detected:
            adata_scrubbed = st.session_state.adata

            # Display histograms (now persistent)
            if st.session_state.doublet_histograms is not None:
                for sample, fig in st.session_state.doublet_histograms.items():
                    st.subheader(f"Histogram for {sample}")
                    st.pyplot(fig)

            # Display summary statistics (now persistent)
            if st.session_state.doublet_summary_stats is not None:
                for sample, stats in st.session_state.doublet_summary_stats.items():
                    st.write(
                        f"Sample {sample}: {stats['doublets']}/{stats['total']} predicted doublets "
                        f"({stats['percentage']:.2f}%)"
                    )

            # Removal option
            if st.checkbox("Remove predicted doublets from dataset"):
                adata_filtered = adata_scrubbed[adata_scrubbed.obs['predicted_doublets'] == False, :]

                st.session_state.adata = adata_filtered
                st.success(
                    f"Doublets removed. Dataset now has "
                    f"{adata_filtered.n_obs} cells and {adata_filtered.n_vars} genes."
                )

                # Keep detection flag and results, just clear scrublet instances
                st.session_state.scrublet_inst = None

                st.subheader('Cell-level info (.obs)')
                df = adata_filtered.obs.head(20).copy()
                for col in df.select_dtypes(include=bool).columns:
                    df[col] = df[col].astype(str)
                st.dataframe(df)


elif mode == "Standard Pipeline" and selected_step == "Post-Quality Control Visualization":
    if st.session_state.adata is None:
        st.warning("Please load data first.")
    else:
        st.markdown('<h2 class="step-header">Step 6: Post-Quality Control Visualization </h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <strong>What we’re doing:</strong> Visualize QC metrics post-filtering.
        </div>
        """, unsafe_allow_html=True)

        adata = st.session_state.adata.copy()

        if 'postqc_plot' not in st.session_state:
            st.session_state.postqc_plot = False
        if 'postqc_plot_fig' not in st.session_state:
            st.session_state.postqc_plot_fig = None

        if st.button('Visualize post-QC metrics'):
            def postQCplots(adata):
                groupby = "sample" if "sample" in adata.obs.columns else None
                metrics = ["total_counts", "n_genes_by_counts"]

                # add percentage metrics if available
                for m in ["pct_counts_mt", "pct_counts_hb", "pct_counts_ribo"]:
                    if m in adata.obs.columns:
                        metrics.append(m)

                has_doublets = {"doublet_score", "predicted_doublet"} <= set(adata.obs.columns)

                # how many subplots we need
                nplots = len(metrics) + (1 if has_doublets else 0)
                ncols = min(3, nplots)
                nrows = -(-nplots // ncols)  # ceil division

                fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
                axes = np.array(axes).reshape(-1)

                # violin plots for metrics
                for i, metric in enumerate(metrics):
                    sc.pl.violin(adata, [metric], groupby=groupby, ax=axes[i], show=False, jitter=0.4)
                    titles = {
                        "total_counts": "Total Counts per Cell",
                        "n_genes_by_counts": "Number of Genes per Cell",
                        "pct_counts_mt": "Mitochondrial Gene %",
                        "pct_counts_hb": "Hemoglobin Gene %",
                        "pct_counts_ribo": "Ribosomal Gene %"
                    }
                    axes[i].set_title(titles.get(metric, metric))

                # doublet histogram
                if has_doublets:
                    ax = axes[len(metrics)]
                    ax.hist(adata.obs["doublet_score"], bins=50, alpha=0.7,
                            color='lightblue', label='All cells')
                    doublet_scores = adata.obs.loc[adata.obs["predicted_doublet"], "doublet_score"]
                    ax.hist(doublet_scores, bins=50, alpha=0.8,
                            color='red', label='Predicted doublets')
                    ax.set_xlabel('Doublet Score')
                    ax.set_ylabel('Cell Count')
                    ax.set_title('Doublet Score Distribution')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                # hide unused axes
                for j in range(nplots, len(axes)):
                    axes[j].set_visible(False)

                plt.tight_layout()
                return fig

            fig = postQCplots(adata)
            st.session_state.postqc_plot_fig = fig
            st.session_state.postqc_plot = True

        if st.session_state.postqc_plot and st.session_state.postqc_plot_fig is not None:
            st.subheader("Violin Plots of QC Metrics")
            st.pyplot(st.session_state.postqc_plot_fig)


elif mode == "Standard Pipeline" and selected_step == "Normalization & Logarithmization":
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Step 7: Normalization & Logarithmization</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong>  Gene expression is normalized to account for sequencing depth differences by scaling counts and applying a log transformation.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.adata.copy()

        if 'norm_and_log' not in st.session_state:
            st.session_state.norm_and_log = False
        
        # Normalization options
        st.subheader("Normalization Method")

        # input for normalization target sum
        target_sum = st.number_input("Target sum", min_value=0.0, value=1e4, step=1e3, format="%.0f")
        
        # checkbox for log1p
        log_transform = st.checkbox("Apply log(x+1) transformation", value=True)
        
        if st.button("Apply Normalization"):
            with st.spinner("Normalizing data..."):
                # save counts_postqc as .raw
                adata.raw = ad.AnnData(
                                X=adata.layers['counts_postqc'],
                                var=adata.var.copy(),
                                obs=adata.obs.copy()
                                )

                # Normalize
                if target_sum == 0:
                    computed_target = np.median(adata.X.sum(axis=1))
                    sc.pp.normalize_total(adata, target_sum=computed_target)
                else:
                    sc.pp.normalize_total(adata, target_sum=target_sum)
                
                # Log transform
                if log_transform:
                    sc.pp.log1p(adata)
                
                # store normalized counts as a layer
                adata.layers['norm_log'] = adata.X

                # Store updated data
                st.session_state.adata = adata
                st.session_state.norm_and_log = True
                st.success("Normalization complete!")

        if st.session_state.norm_and_log:
            # Show before/after comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Raw data distribution
            raw_counts = adata.raw.X.sum(axis=1).A1 if hasattr(adata.raw.X, 'A1') else adata.raw.X.sum(axis=1)
            axes[0].hist(raw_counts, bins=50, alpha=0.7)
            axes[0].set_title('Before Normalization')
            axes[0].set_xlabel('Total Counts')
            axes[0].set_ylabel('Frequency')
            
            # Normalized data distribution
            norm_counts = adata.X.sum(axis=1).A1 if hasattr(adata.X, 'A1') else adata.X.sum(axis=1)
            axes[1].hist(norm_counts, bins=50, alpha=0.7)
            axes[1].set_title('After Normalization')
            axes[1].set_xlabel('Total Counts')
            axes[1].set_ylabel('Frequency')
            
            plt.tight_layout()
            st.pyplot(fig)

elif mode == "Standard Pipeline" and selected_step == "Feature Selection":
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Step 8: Feature Selection</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Identifying highly variable genes (HVGs) that drive cell-to-cell variation.
        These genes are most informative for downstream analysis.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.adata.copy()

        if 'hvg' not in st.session_state:
            st.session_state.hvg = False
        if 'num_hvg' not in st.session_state:
            st.session_state.num_hvg = None

        # Feature selection parameters
        st.subheader("Feature Selection Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            min_mean = st.number_input("Min mean expression", 0.0, 1.0, 0.0125)
            max_mean = st.number_input("Max mean expression", 1.0, 10.0, 3.0)
        
        with col2:
            min_disp = st.number_input("Min dispersion", 0.0, 2.0, 0.5)
            n_top_genes = st.number_input("Number of top genes", 1000, 10000, 2000)
        
        if st.button("Find HVGs"):
            with st.spinner("Finding highly variable genes..."):
                sc.pp.highly_variable_genes(adata, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp)

                # Store results
                st.session_state.adata = adata
                st.session_state.hvg = True
                n_hvg = adata.var['highly_variable'].sum()
                st.session_state.num_hvg = n_hvg
        
        if st.session_state.hvg and st.session_state.num_hvg is not None:
            n_hvg = st.session_state.num_hvg
            st.success(f"Found {n_hvg} highly variable genes!")
            if n_hvg > 0:
                # Plot highly variable genes
                plt.clf()
                sc.pl.highly_variable_genes(adata, show=False)
                fig = plt.gcf()
                st.pyplot(fig)

                # Show top HVGs
                st.subheader("Top 20 Highly Variable Genes")
                hvg_df = adata.var[adata.var['highly_variable']].sort_values('dispersions_norm', ascending=False).head(20)
                st.dataframe(hvg_df[['means', 'dispersions', 'dispersions_norm']])
            else:
                st.warning("No highly variable genes were found. Try adjusting the parameters.")

elif mode == "Standard Pipeline" and selected_step == 'Adjustment & Scaling':
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Step 9: Adjustment & Scaling </h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Unwanted variation is regressed out, 
                    and gene expression values are standardized to ensure all features contribute equally to downstream analyses.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.adata.copy()

        if 'adjust_and_scale' not in st.session_state:
            st.session_state.adjust_and_scale = False

        # checkbox for regress_out
        regress = st.checkbox('Regress out', value=True)

        # checkbox for scaling
        scaling = st.checkbox('Scale data', value=True)

        if st.button('Apply adjustments'):
            with st.spinner('Adjust and/or scale data...'):
                # regress out
                if regress:
                    sc.pp.regress_out(adata, ['total_counts'])
                
                # scale
                if scaling:
                    sc.pp.scale(adata, max_value=10)
                
                # save adjustments/scaling as a layer
                adata.layers['scaled'] = adata.X

                st.session_state.adata = adata
                st.session_state.adjust_and_scale = True
        
        if st.session_state.adjust_and_scale:
            st.success("Adjustments complete!")

            # Show before/after comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # normalized data distribution
            if 'norm_log' in adata.layers:
                normalized_counts = adata.layers['norm_log'].sum(axis=1).A1 \
                    if hasattr(adata.layers['norm_log'], 'A1') else adata.layers['norm_log'].sum(axis=1)
            else:
                normalized_counts = adata.X.sum(axis=1).A1 \
                    if hasattr(adata.X, 'A1') else adata.X.sum(axis=1)
            axes[0].hist(normalized_counts, bins=50, alpha=0.7)
            axes[0].set_title('Before Adjustment & Scaling')
            axes[0].set_xlabel('Total Counts')
            axes[0].set_ylabel('Frequency')
            
            # scaled data distribution
            plt.hist(adata.X.flatten(), bins=100, alpha=0.7)
            plt.title('Distribution of all scaled values')
            plt.xlabel('Scaled expression (z-score)')
            plt.ylabel('Frequency')
            plt.show()
            
            plt.tight_layout()
            st.pyplot(fig)
        

elif mode == "Standard Pipeline" and selected_step == "Dimensionality Reduction":
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Step 10: Dimensionality Reduction</h2>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Reducing the dimensionality of the data using linear (PCA) and non-linear (UMAP/t-SNE) methods.
        This helps visualize the data and identify cell populations.
        </div>
        """, unsafe_allow_html=True)

        adata = st.session_state.adata.copy()

        if 'pca' not in st.session_state:
            st.session_state.pca = False
        if 'umap' not in st.session_state:
            st.session_state.umap = False
        
        random.seed(0)

        n_pcs_input = st.text_input("Number of principal components (default: None)", "")
        if n_pcs_input.strip() == "":
            n_pcs = None
        else:
            try:
                n_pcs = int(n_pcs_input)
                if n_pcs < 1:
                    st.error("Number of PCs must be positive.")
                    n_pcs = None
            except ValueError:
                st.error("Please enter a valid integer.")
                n_pcs = None

        # options to run PCA on scaled or norm_log
        mat_options = st.selectbox('Choose which data to run PCA on:', ['Adjusted & Scaled', 'Normalized & Logarithmized'])

        # pca
        if st.button("Run PCA"):
            with st.spinner("Running PCA..."):

                if mat_options == 'Adjusted & Scaled':
                    if "scaled" in adata.layers:
                        # heuristic: check if current X looks unscaled
                        if adata.X.min() >= 0:   # norm+log has only positive values
                            st.info("Switching to scaled data for PCA...")
                            adata.X = adata.layers["scaled"].copy()
                    else:
                        st.warning('Adjusted and scaled data not found. Please make sure you run Adjustments & Scaling for your dataset.')
                else:
                    if 'norm_log' in adata.layers:
                        st.info("Switching to norm_log data for PCA...")
                        adata.X = adata.layers['norm_log'].copy()
                    else:
                        st.warning('Normalized and logarithmized data not found. Please make sure you run Normalization & Transformation for your dataset.')

                sc.tl.pca(
                    adata,
                    n_comps=n_pcs,
                    svd_solver='arpack'
                )

                # st.session_state.adata = adata
                st.session_state.pca = True
                st.success("PCA complete!")

        # pca plots
        if st.session_state.pca and "X_pca" in adata.obsm:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("PCA Scatter Plot")
                sc.pl.pca(adata, show=False, title="")
                fig = plt.gcf()
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                st.subheader("PCA Scatter Plot Colored by Sample")
                sc.pl.pca(adata, color="sample", legend_loc="upper right", title="", show=False)
                fig2 = plt.gcf()
                st.pyplot(fig2)
                plt.close(fig2)

        # umap/t-sne
        if st.session_state.pca and "X_pca" in adata.obsm:
            st.subheader("Non-linear Dimensionality Reduction (UMAP/t-SNE)")
            embedding_method = st.selectbox("Choose method:", ["UMAP", "t-SNE"])
            if embedding_method == "UMAP":
                n_pcs_input_umap = st.text_input("Number of PCs (default: None)", "")
                n_pcs_umap = None  # Initialize the variable
                
                if n_pcs_input_umap.strip() != "":  # Check if not empty after stripping
                    try:
                        n_pcs_umap = int(n_pcs_input_umap.strip())  # Strip here too
                        if n_pcs_umap < 1:
                            st.error("Number of PCs must be positive.")
                            n_pcs_umap = None
                    except ValueError:
                        st.error("Please enter a valid integer.")
                        n_pcs_umap = None
                
                if st.button("Run UMAP"):
                    with st.spinner("Running UMAP..."):

                        sc.pp.neighbors(adata, n_pcs=n_pcs_umap)  # Use n_pcs_umap if provided, else defaults to all PCs
                        st.write("Neighbors complete, starting UMAP")

                        sc.tl.umap(adata)

                        st.session_state.adata = adata
                        st.session_state.umap = True
                        st.success("UMAP complete!")

                if st.session_state.umap and "X_umap" in adata.obsm:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("UMAP Embedding")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sc.pl.umap(adata, ax=ax, title="")
                        st.pyplot(fig)
                        plt.close(fig)

                    with col2:
                        st.subheader("UMAP Embedding Colored by Sample")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sc.pl.umap(adata, color="sample", legend_loc="upper right", title="", ax=ax)
                        st.pyplot(fig)
                        plt.close(fig)

            elif embedding_method == "t-SNE":
                col1, col2 = st.columns(2)
                with col1:
                    perplexity = st.slider("Perplexity", 5, 100, 30)
                    early_exaggeration = st.slider("Early exaggeration", 5, 50, 12)
                with col2:
                    learning_rate = st.slider("Learning rate", 10, 1000, 200)

                if st.button("Run t-SNE"):
                    with st.spinner("Running t-SNE..."):
                        sc.tl.tsne(
                            adata,
                            perplexity=perplexity,
                            early_exaggeration=early_exaggeration,
                            learning_rate=learning_rate,
                            random_state=0  # reproducible
                        )
                        st.success("t-SNE complete!")

                if "X_tsne" in adata.obsm:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sc.pl.tsne(adata, ax=ax, title="")
                    st.pyplot(fig)
                    plt.close(fig)

elif mode == "Standard Pipeline" and selected_step == "Clustering":
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Step 11: Clustering</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Grouping cells into clusters based on their gene expression similarity.
        This helps identify distinct cell types or states.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.adata.copy()
        if 'clustering' not in st.session_state:
            st.session_state.clustering = False
        
        # Check for clustering prerequisites
        has_neighbors = 'neighbors' in adata.uns
        has_embedding = any(key in adata.obsm for key in ['X_umap', 'X_tsne'])

        if not has_neighbors:
            st.warning("Missing neighborhood graph. Please run PCA and compute neighbors (required for Leiden/Louvain).")
        elif not has_embedding:
            st.warning("No embedding (UMAP or t-SNE) found. You may still cluster, but plots won't be generated.")
        else:
            # Clustering parameters
            st.subheader("Clustering Parameters")
            
            clustering_method = st.selectbox(
                "Clustering method:",
                ["Leiden", "Louvain"]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                resolution = st.slider("Resolution", 0.1, 2.0, 0.5, 0.1)
            with col2:
                n_iterations = st.number_input("Number of iterations", 1, 10, 2)
            
            if st.button(f"Run {clustering_method} Clustering"):
                with st.spinner(f"Running {clustering_method} clustering..."):
                    if clustering_method == "Leiden":
                        sc.tl.leiden(adata, resolution=resolution, n_iterations=n_iterations, key_added=f'leiden_{resolution}')
                        cluster_key = f'leiden_{resolution}'
                    else:
                        sc.tl.louvain(adata, resolution=resolution, key_added=f'louvain_{resolution}')
                        cluster_key = f'louvain_{resolution}'
                    
                    # Store results
                    st.session_state.adata = adata
                    st.session_state.clustering = True
                    
                    n_clusters = len(adata.obs[cluster_key].unique())
                    st.success(f"{clustering_method} clustering complete! Identified {n_clusters} clusters.")
                    
        # Plot clustering results - moved outside the else block and added proper checks
        clustering_keys = [col for col in adata.obs.columns if col.startswith("leiden_") or col.startswith("louvain_")]
        
        if clustering_keys:
            # Get the most recent clustering key (or let user select)
            cluster_key = clustering_keys[-1]  # Use the last (most recent) clustering result
            
            # Plot UMAP or t-SNE with clusters
            if 'X_umap' in adata.obsm:
                col1, col2 = st.columns(2)
                
                # UMAP colored by clusters
                with col1:
                    st.subheader(f'Clusters ({cluster_key})')
                    fig, ax = plt.subplots(figsize=(8,6))
                    sc.pl.umap(adata, 
                               color=cluster_key, 
                               ax=ax, 
                               legend_loc='right', 
                               frameon=False,
                               title='')
                    st.pyplot(fig)
                    plt.close(fig)
                
                # UMAP colored by sample
                with col2:
                    st.subheader('UMAP Colored by Sample')
                    fig, ax = plt.subplots(figsize=(8,6))
                    sc.pl.umap(adata, 
                               color='sample', 
                               ax=ax, 
                               legend_loc='right', 
                               frameon=False,
                               title='')
                    st.pyplot(fig)
                    plt.close(fig)
                    
            elif 'X_tsne' in adata.obsm:
                st.subheader(f't-SNE Colored by Clusters ({cluster_key})')
                fig, ax = plt.subplots(figsize=(8, 6))
                sc.pl.tsne(adata, color=cluster_key, ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            
            # Cluster composition analysis
            col1, col2 = st.columns(2)

            with col1:
                # Cluster composition
                st.subheader("Cell counts per cluster")
                cluster_counts = adata.obs[cluster_key].value_counts().sort_index()

                fig, ax = plt.subplots(figsize=(8, 4))
                # Use matplotlib bar plot instead of seaborn to avoid import issues
                ax.bar(cluster_counts.index.astype(str), cluster_counts.values, color="steelblue")
                ax.set_xlabel("Cluster")
                ax.set_ylabel("Cell count")
                ax.tick_params(left=False, bottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                plt.xticks(rotation=0)
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                # Cluster composition colored by sample
                st.subheader("Cell counts per cluster by samples")
                
                grouped_count = (
                    adata.obs[[cluster_key, 'sample']]
                    .value_counts()
                    .reset_index(name='count')
                )
                pivot_df = grouped_count.pivot(index=cluster_key, columns='sample', values='count').fillna(0)

                fig, ax = plt.subplots(figsize=(8, 4))
                # Track bottom of the stacked bars
                bottom = None
                for sample in pivot_df.columns:
                    ax.bar(pivot_df.index.astype(str), pivot_df[sample], label=sample, bottom=bottom)
                    bottom = pivot_df[sample] if bottom is None else bottom + pivot_df[sample]
                ax.set_xlabel("Cluster")
                ax.set_ylabel("Cell count")
                ax.tick_params(left=False, bottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                plt.xticks(rotation=0)
                ax.legend(title="Sample")
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No clustering results found. Please run clustering first.")
                
        # Show updated .obs and .var
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Cell-level info (.obs)')
            df = adata.obs.head(20).copy()
            for col in df.select_dtypes(include=bool).columns:
                df[col] = df[col].astype(str)
            st.dataframe(df)
    
        with col2:
            st.subheader('Gene-level info (.var)')
            df = adata.var.head(20).copy()
            for col in df.select_dtypes(include=bool).columns:
                df[col] = df[col].astype(str)
            st.dataframe(df)

elif mode == "Standard Pipeline" and selected_step == "Differential Gene Expression":
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Step 12: Differential Gene Expression </h2>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Identify differentially expressed genes (DEGs) in each cluster.
        </div>
        """, unsafe_allow_html=True)

        adata = st.session_state.adata.copy()

        # Get clustering keys
        clustering_keys = [col for col in adata.obs.columns if col.startswith("leiden_") or col.startswith("louvain_")]

        if not clustering_keys:
            st.warning("Please run Leiden or Louvain clustering first!")
        else:
            st.subheader("Clustering Parameters")

            # Let user pick which clustering to use
            selected_cluster_key = st.selectbox("Select clustering key to group by:", clustering_keys)

            # DEG method selection
            rank_deg_method = st.selectbox("Choose test method:", ['t-test', 't-test_overestim_var', 'wilcoxon', 'logreg'])

            if st.button('Identify DEGs'):
                with st.spinner('Identifying DEGs...'):
                    key_added = f'deg_{selected_cluster_key}_{rank_deg_method}'
                    sc.tl.rank_genes_groups(
                        adata,
                        groupby=selected_cluster_key,
                        use_raw=False,
                        method=rank_deg_method,
                        key_added=key_added
                    )

                    st.session_state.adata = adata  # store results back

                    # plot top DEGs
                    st.subheader('Top 25 DEGs per cluster')
                    axes_list = sc.pl.rank_genes_groups(
                                adata,
                                n_genes=25,
                                sharey=False,
                                key=key_added,
                                show=False,
                                return_fig=True
                            )

                    # Convert to a Figure
                    if isinstance(axes_list, list):
                        fig = axes_list[0].get_figure()  # use the first Axes
                    else:
                        fig = axes_list.get_figure()

                    st.pyplot(fig)

                    # show degs as table
                    # Extract DEG results
                    rank_results = adata.uns[key_added]
                    groups = rank_results['names'].dtype.names  # cluster names

                    # Collect top 25 genes per cluster into a tidy dataframe
                    top_genes = []
                    for group in groups:
                        names = rank_results['names'][group][:25]
                        scores = rank_results['scores'][group][:25]
                        pvals = rank_results['pvals_adj'][group][:25] if 'pvals_adj' in rank_results else rank_results['pvals'][group][:25]

                        for rank, (gene, score, pval) in enumerate(zip(names, scores, pvals), start=1):
                            top_genes.append({
                                'Cluster': group,
                                'Rank': rank,
                                'Gene': gene,
                                'Score': score,
                                'Adjusted p-value': pval
                            })

                    # Convert to DataFrame
                    deg_df = pd.DataFrame(top_genes)
                    # convert to scientific notation
                    deg_df["Adjusted p-value"] = deg_df["Adjusted p-value"].apply(lambda x: f"{x:.2e}")

                    # Show in Streamlit
                    st.subheader("Top 25 DEGs per cluster (table)")
                    st.dataframe(deg_df)


elif mode == "Standard Pipeline" and selected_step == "Visualization":
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Step 13: Visualization</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Creating various visualizations to explore your data.
        This includes gene expression plots, cluster comparisons, and quality metrics.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.adata.copy()
        adata.var_names = adata.var_names.str.lower()

        # Visualization options
        viz_type = st.selectbox(
            "Choose visualization type:",
            ["Gene Expression", "Cluster Analysis", "Quality Metrics"]
        )
        
        if viz_type == "Gene Expression":
            st.subheader("Gene Expression Visualization")

            # Gene selection
            genes = adata.var_names.tolist()
            gene_input = st.multiselect('Enter gene names:', genes)

            if gene_input:
                available_genes = [gene for gene in gene_input if gene in adata.var_names]
                unavailable_genes = [gene for gene in gene_input if gene not in adata.var_names]

                if unavailable_genes:
                    st.warning(f"The following genes were not found and will be ignored: {', '.join(unavailable_genes)}")

                if available_genes:
                    plot_type = st.selectbox(
                        'Plot type:',
                        ['Scatter plot', 'Violin plot', 'Dot plot', 'Heatmap', 'Matrix plot', 'Stacked violin plot']
                    )

                    if plot_type == 'Scatter plot':
                        projection_type = st.selectbox("Projection type:", ["UMAP", "t-SNE", "PCA"])

                        if st.button("Plot Gene Expression"):
                            layer = 'norm_log' if 'norm_log' in adata.layers else None
                            if layer is None:
                                st.warning('Normalized and logarithmized data not found. Please run Normalization & Transformation.')
                            else:
                                n_genes = len(available_genes)
                                n_cols = min(3, n_genes)
                                n_rows = (n_genes + n_cols - 1) // n_cols

                                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                                axes = np.atleast_1d(axes).flatten()

                                for i, gene in enumerate(available_genes):
                                    ax = axes[i]
                                    if projection_type == "UMAP" and "X_umap" in adata.obsm:
                                        sc.pl.umap(adata, color=gene, ax=ax, ncols=3, frameon=False, show=False, layer=layer, color_map='gnuplot')
                                    elif projection_type == "t-SNE" and "X_tsne" in adata.obsm:
                                        sc.pl.tsne(adata, color=gene, ax=ax, ncols=3, frameon=False, show=False, layer=layer, color_map='gnuplot')
                                    elif projection_type == "PCA" and "X_pca" in adata.obsm:
                                        sc.pl.pca(adata, color=gene, ax=ax, ncols=3, frameon=False, show=False, layer=layer, color_map='gnuplot')

                                    ax.set_title(gene)

                                # remove unused subplots
                                for j in range(i + 1, len(axes)):
                                    fig.delaxes(axes[j])

                                plt.tight_layout()
                                st.pyplot(fig)


                    elif plot_type == 'Violin plot':
                        if st.button("Plot Gene Expression"):
                            n_genes = len(available_genes)
                            n_cols = min(3, n_genes)
                            n_rows = (n_genes + n_cols - 1) // n_cols
                            
                            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                            axes = axes.flatten() if n_genes > 1 else [axes]
                            
                            for i, gene in enumerate(available_genes):
                                ax = axes[i]
                                sc.pl.violin(adata, keys=gene, ax=ax, show=False)
                                ax.set_title(gene)
                            
                            for j in range(i + 1, len(axes)):
                                fig.delaxes(axes[j])  # remove empty subplots

                            plt.tight_layout()
                            st.pyplot(fig)
                    elif plot_type == 'Dot plot':

                        clustering_keys = [col for col in adata.obs.columns if col.startswith("leiden") or col.startswith("louvain")]
                        if clustering_keys:
                            group = st.selectbox("Group by:", clustering_keys)
                        else:
                            st.warning("No clustering keys found in adata.obs.")
                            group = None

                        if st.button("Plot Gene Expression") and group is not None:
                            n_genes = len(available_genes)
                            n_cols = min(3, n_genes)
                            n_rows = (n_genes + n_cols - 1) // n_cols
                            
                            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                            axes = axes.flatten() if n_genes > 1 else [axes]
                            
                            for i, gene in enumerate(available_genes):
                                ax = axes[i]
                                sc.pl.dotplot(adata, var_names=gene, groupby=group, ax=ax, show=False, dendrogram=True)
                                ax.set_title(gene)
                            
                            for j in range(i + 1, len(axes)):
                                fig.delaxes(axes[j])  # remove empty subplots

                            plt.tight_layout()
                            st.pyplot(fig)
                    elif plot_type == 'Heatmap':

                        clustering_keys = [col for col in adata.obs.columns if col.startswith("leiden") or col.startswith("louvain")]
                        if clustering_keys:
                            group = st.selectbox("Group by:", clustering_keys)
                        else:
                            st.warning("No clustering keys found in adata.obs.")
                            group = None

                        if st.button("Plot Gene Expression") and group is not None:
                            n_genes = len(available_genes)
                            width = 8  # Fixed width
                            height = max(4, n_genes * 0.5 + 2)

                            sc.pl.heatmap(adata, var_names=available_genes, groupby=group, dendrogram=True, show=False, figsize=(width, height), swap_axes=True)
                            fig = plt.gcf()  # Get current figure
                            st.pyplot(fig)
                            plt.close(fig)
                    elif plot_type == 'Matrix plot':

                        clustering_keys = [col for col in adata.obs.columns if col.startswith("leiden") or col.startswith("louvain")]
                        if clustering_keys:
                            group = st.selectbox("Group by:", clustering_keys)
                        else:
                            st.warning("No clustering keys found in adata.obs.")
                            group = None

                        if st.button("Plot Gene Expression") and group is not None:
                            n_genes = len(available_genes)
                            width = 8  # Fixed width
                            height = max(4, n_genes * 0.5 + 2)

                            sc.pl.matrixplot(adata, var_names=available_genes, groupby=group, dendrogram=True, show=False, figsize=(width, height), swap_axes=True)
                            fig = plt.gcf()  # Get current figure
                            st.pyplot(fig)
                            plt.close(fig)
                    elif plot_type == 'Stacked violin plot':

                        clustering_keys = [col for col in adata.obs.columns if col.startswith("leiden") or col.startswith("louvain")]
                        if clustering_keys:
                            group = st.selectbox("Group by:", clustering_keys)
                        else:
                            st.warning("No clustering keys found in adata.obs.")
                            group = None

                        if st.button("Plot Gene Expression") and group is not None:
                            n_genes = len(available_genes)
                            width = 8  # Fixed width
                            height = max(4, n_genes * 0.5 + 2)

                            sc.pl.stacked_violin(adata, var_names=available_genes, groupby=group, dendrogram=True, show=False, figsize=(width, height), swap_axes=True)
                            fig = plt.gcf()  # Get current figure
                            st.pyplot(fig)
                            plt.close(fig)

                    else:
                        st.warning("No valid gene names entered.")
                    
        elif viz_type == "Cluster Analysis":
            st.subheader("Cluster Visualization")

            # Find clustering keys
            cluster_keys = [col for col in adata.obs.columns if col.startswith("leiden") or col.startswith("louvain")]

            # Ensure projection options are available
            projection_options = [proj.replace('X_', '') for proj in adata.obsm.keys() if proj.startswith('X_')]
            if not projection_options:
                st.warning("No projection found in `adata.obsm` starting with 'X_'.")
            else:
                projection_type = st.selectbox('Projection type:', projection_options)

                if not cluster_keys:
                    st.warning("No cluster annotations found.")
                else:
                    cluster_key = st.selectbox("Select clustering key:", cluster_keys)
                    if st.button("Plot Clusters"):
                        fig, ax = plt.subplots(figsize=(8, 6))
                        if projection_type.lower() == 'umap':
                            sc.pl.umap(adata, color=cluster_key, ax=ax, frameon=False, show=False)
                        elif projection_type.lower() == 'tsne':
                            sc.pl.tsne(adata, color=cluster_key, ax=ax, frameon=False, show=False)
                        elif projection_type.lower() == 'pca':
                            sc.pl.pca(adata, color=cluster_key, ax=ax, frameon=False, show=False)
                        else:
                            st.warning(f"No plotting function defined for projection: {projection_type}")
                        st.pyplot(fig)

        elif viz_type == "Quality Metrics":
            st.subheader("QC Metric Visualization")

            # Filter for common QC metrics patterns in observations
            qc_obs_patterns = ['n_genes', 'n_counts', 'total_counts', 'pct_counts', 'percent_', 'pct_', 'mt_', 'ribo_', 'hb_', 'doublet', 'scrublet']
            qc_metrics_col = [col for col in adata.obs.columns 
                            if any(pattern in col.lower() for pattern in qc_obs_patterns) 
                            and not col.startswith(('leiden_', 'louvain_',)) 
                            and 'highly_variable' not in col.lower()]
            
            # Filter for common QC metrics patterns in variables
            qc_var_patterns = ['mean', 'std', 'var', 'cv', 'dropout', 'highly_variable', 'dispersions', 'pct_dropout']
            qc_metrics_var = [col for col in adata.var.columns 
                            if any(pattern in col.lower() for pattern in qc_var_patterns)]

            all_qc_metrics = qc_metrics_col + qc_metrics_var

            if not all_qc_metrics:
                st.warning("No QC metrics found in the data.")

            qc_genes = st.multiselect("Select QC metrics to visualize:", all_qc_metrics)

            if qc_genes:
                # Separate obs and var metrics
                obs_metrics = [metric for metric in qc_genes if metric in adata.obs.columns]
                var_metrics = [metric for metric in qc_genes if metric in adata.var.columns]
                
                # Plot observation metrics (if any)
                if obs_metrics:
                    st.write("**Cell-level QC Metrics (Observation Data):**")
                    sc.pl.violin(
                        adata,
                        obs_metrics,
                        jitter=0.4,
                        multi_panel=True,
                        show=False
                    )
                    fig = plt.gcf()
                    st.pyplot(fig)
                    plt.close(fig)
        
                # Plot variable metrics (if any)
                if var_metrics:
                    st.write("**Gene-level QC Metrics (Variable Data):**")
                    for metric in var_metrics:
                        # Create histogram for each variable metric
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(adata.var[metric].dropna(), bins=50, alpha=0.7, edgecolor='black')
                        ax.set_xlabel(metric)
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Distribution of {metric}')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)

elif mode == "Standard Pipeline" and selected_step == "Export Results":
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Step 14: Export Results </h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Export results as a .h5ad file that can be reuploaded into scViewer for other analysis and visualizations.
        </div>
        """, unsafe_allow_html=True)

        adata = st.session_state.adata.copy()
        
        # Estimate file size
        estimated_size_mb = (adata.n_obs * adata.n_vars * 4) / (1024 * 1024)
        st.info(f"Estimated file size: ~{estimated_size_mb:.1f}MB")
        
        # Single, fast export method
        if st.button("Export Results", key="export_results"):
            timestamp = int(time.time())
            
            # Get user's Downloads folder
            downloads_path = os.path.expanduser("~/Downloads")
            filename = f"exported_results_{timestamp}.h5ad"
            full_path = os.path.join(downloads_path, filename)
            
            with st.spinner("Exporting to Downloads folder..."):
                try:
                    # Direct write to Downloads folder
                    adata.write_h5ad(full_path)
                    file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    
                    st.success(f"Export complete!")
                    st.info(f"**File:** `{filename}`")
                    st.info(f"**Size:** {file_size_mb:.1f} MB")
                    st.info(f"**Location:** `{full_path}`")
                    
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        
        st.markdown("---")

        # reset button
        if st.button("Reset Analysis"):
            # Store keys to preserve (if any)
            keys_to_preserve = {"selected_step"}  # add keys that users want to keep
            
            # Create a list of keys to delete (avoiding dict modification during iteration)
            keys_to_reset = [key for key in st.session_state.keys() if key not in keys_to_preserve]
            
            # Clear session state variables
            for key in keys_to_reset:
                if key in st.session_state:  # Extra safety check
                    del st.session_state[key]
            
            # Explicitly reset critical variables
            st.session_state.adata = None
            
            st.success("Session has been reset. Please upload a new dataset to begin again.")
            st.rerun()

elif mode == 'Advanced Analysis' and selected_step == 'Data Loading':
    st.markdown('<h2 class="step-header">Data Loading</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>What we're doing:</strong> Advanced analysis allows you to do cell and gene-level specific manipulations catered to your questions. It is suggested that you run the standard pipeline first 
                before doing advanced analysis. 
    </div>
    """, unsafe_allow_html=True)

    if 'adata' in st.session_state and st.session_state.adata is not None:
        # make all genes names lowercase
        adata = st.session_state.adata.copy()
        adata.var_names = [gene.lower() for gene in st.session_state.adata.var_names]

        # Display dataset overview when data exists
        st.success('Dataset detected in session.')

        st.subheader("Dataset Overview")
        display_data_summary(adata)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("First 10 Cells")
            df = adata.obs.head(10).copy()
            for col in df.select_dtypes(include=bool).columns:
                df[col] = df[col].astype(str)
            st.dataframe(df)
        with col2:
            st.subheader("First 10 Genes")
            df = adata.var.head(10).copy()
            for col in df.select_dtypes(include=bool).columns:
                df[col] = df[col].astype(str)
            st.dataframe(df)
            
        # Option to reload/replace data
        st.markdown("---")
        if st.button("Load Different Dataset", type="secondary"):
            # Clear existing data
            st.session_state.adata = None
            if 'dataset_overview' in st.session_state:
                del st.session_state.dataset_overview
            st.rerun()

    else:
        # File upload section when no data exists
        uploaded_file = st.file_uploader(
            "Upload your single-cell data file:",
            type=['h5ad'],
            help="H5AD files larger than 1GB may take several minutes to load"
        )
        
        if uploaded_file is not None:
            # Get file size info
            file_size_mb, est_memory_mb = get_file_size_info(uploaded_file)
            
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**File Size:** {file_size_mb:.1f} MB")
            with col2:
                st.info(f"**Est. Memory:** {est_memory_mb:.1f} MB")
            with col3:
                load_time_est = max(10, file_size_mb / 50)  # Rough estimate
                st.info(f"⏱**Est. Load Time:** {load_time_est:.0f}s")
            
            # Warning for very large files
            if file_size_mb > 1000:  # 1GB
                st.warning("""
                **Large File Detected**
                - This file is quite large and may take several minutes to load
                """)
                
                proceed = st.button("Proceed with Loading (Large File)", type="primary")
            else:
                proceed = st.button("Load File", type="primary")
            
            if proceed:
                # Final resource check before loading
                resource_ok, resource_msg = check_system_resources()
                
                if not resource_ok:
                    st.error(f"Cannot proceed: {resource_msg}")
                    st.stop()
                
                # Load file with error handling
                with st.spinner(f"Loading {uploaded_file.name}... This may take a few minutes for large files."):
                    try:
                        # Clear any existing data to free memory
                        if 'adata' in st.session_state and st.session_state.adata is not None:
                            del st.session_state.adata
                            gc.collect()
                        
                        # Load the file
                        adata, error = load_h5ad_safely(uploaded_file, show_progress=True)
                        
                        if error:
                            st.error(f"Error loading file: {error}")
                            st.info("""
                            **Troubleshooting tips:**
                            - Ensure the file is a valid H5AD format
                            - Try closing other applications to free memory
                            - For very large files, consider using a machine with more RAM
                            - Check that the file is not corrupted by trying to open it elsewhere
                            """)
                        else:
                            # Successfully loaded
                            st.session_state.adata = adata
                            st.session_state.dataset_overview = True
                            st.success(f"Successfully loaded {uploaded_file.name}")
                            
                            # Show basic dataset info
                            st.info(f"**Cells:** {adata.n_obs:,} | **Genes:** {adata.n_vars:,}")
                            
                            # Show current memory usage
                            current_memory = psutil.virtual_memory().percent
                            st.info(f"Current memory usage: {current_memory:.1f}%")
                            
                            # Automatically show overview after successful load
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")
                        st.info("If this error persists, try restarting the application.")
                        # Clear any partial data
                        if 'adata' in st.session_state:
                            st.session_state.adata = None

elif mode == 'Advanced Analysis' and selected_step == 'Cell-level Manipulation':
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Cell-level Manipulation</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Cell-level manipulation allows you to label, cluster, and so on on a cell-level.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.adata.copy()
        adata.var_names = adata.var_names.str.lower()

        task = st.selectbox('Choose a task:', ['Label clusters', 'Subset and filter', 'Subset and filter cells (gene expression)', 'Annotate cells (gene expression)', 'Re-analyze clusters'])

        # label clusters
        # check if clusters are in .obs
        if task == 'Label clusters':

            st.subheader('Label cluster names')

            # Look for clustering columns
            clustering_cols = [col for col in adata.obs.columns if col.startswith("leiden") or col.startswith("louvain")]
            if len(clustering_cols) == 0:
                st.warning('No clustering labels found in `.obs` (like "leiden" or "louvain").')
            else:
                selected_cluster_key = st.selectbox("Select clustering column:", clustering_cols)
                unique_clusters = sorted(adata.obs[selected_cluster_key].unique().tolist(), key=lambda x: str(x))
                
                # Plot the original clusters label first
                fig, ax = plt.subplots(figsize=(8,6))
                sc.pl.umap(adata, color=selected_cluster_key, ax=ax, frameon=False, show=False) 
                st.pyplot(fig, use_container_width=True)
                
                st.markdown("Provide custom labels for each cluster:")
                new_labels = {}
                for cluster in unique_clusters:
                    new_label = st.text_input(f"Cluster {cluster}", value=f"Cluster {cluster}", key=f"{selected_cluster_key}_{cluster}")
                    new_labels[str(cluster)] = new_label

                # user-defined custom label
                st.markdown("### Save custom labels to `.obs` with a name")
                new_column_name = st.text_input("New column name (e.g., 'custom_labels_1')", value=f"{selected_cluster_key}_custom")
                
                # Apply the new labels
                if st.button("Apply Labels"):
                    if new_column_name.strip() == "":
                        st.error("Please enter a valid column name.")
                    elif new_column_name in adata.obs.columns:
                        st.warning(f"A column named `{new_column_name}` already exists. Choose another name.")
                    else:
                        adata.obs[new_column_name] = adata.obs[selected_cluster_key].astype(str).map(new_labels).astype("category")
                        st.success(f"Saved to `adata.obs['{new_column_name}']`.")
                        st.session_state.adata = adata
                    
                    # Display results using the temporary custom_labels for visualization
                    label_counts = adata.obs[new_column_name].value_counts().reset_index()
                    label_counts.columns = [new_column_name, 'Cell Count']
                    st.dataframe(label_counts)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sc.pl.umap(adata, color=new_column_name, ax=ax, frameon=False, show=False)
                    st.pyplot(fig, use_container_width=True)
                    
                            
        elif task == 'Subset and filter':
            st.subheader('Subset and filter cells')

            obs_col = adata.obs.columns.tolist()
            subset = st.selectbox('Subset by:', obs_col)

            selected_values = None
            operator = None
            threshold = None

            subset_series = adata.obs[subset]
            col_dtype = subset_series.dtype

            if pd.api.types.is_bool_dtype(col_dtype) or pd.api.types.is_categorical_dtype(col_dtype) or not pd.api.types.is_numeric_dtype(col_dtype):
                unique_values = subset_series.unique().tolist()
                selected_values = st.multiselect(f"Select `{subset}` values to keep:", unique_values)

            elif pd.api.types.is_numeric_dtype(col_dtype):
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    operator = st.selectbox('Operator:', ['=', '>', '<', '>=', '<='])
                with col2:
                    try:
                        default_val = float(np.nanmedian(subset_series))
                    except Exception:
                        default_val = 0.0
                    threshold = st.number_input('Value:', value=default_val)
                with col3:
                    st.write(f"Range: {np.nanmin(subset_series):.2f} to {np.nanmax(subset_series):.2f}")

            # Prepare name input BEFORE button press, so it survives rerun
            st.markdown("### Save this subset")
            subset_name = st.text_input("Name this subset:", value=f"{subset}_filtered")

            if st.button('Subset'):
                subset_adata = None

                # Numeric filtering
                if pd.api.types.is_numeric_dtype(col_dtype) and operator is not None:
                    if operator == '=':
                        mask = subset_series == threshold
                    elif operator == '>':
                        mask = subset_series > threshold
                    elif operator == '<':
                        mask = subset_series < threshold
                    elif operator == '>=':
                        mask = subset_series >= threshold
                    elif operator == '<=':
                        mask = subset_series <= threshold
                    else:
                        st.warning("Invalid operator.")
                        mask = None

                    if mask is not None:
                        subset_adata = adata[mask].copy()

                # Categorical or boolean filtering
                elif selected_values:
                    mask = subset_series.isin(selected_values)
                    subset_adata = adata[mask].copy()

                if subset_adata is not None:
                    st.success(f"Subset applied. {subset_adata.n_obs} cells retained. {subset_adata.n_vars} genes retained.")

                    # Show preview
                    df = subset_adata.obs[[subset]].copy()
                    for col in df.select_dtypes(include=bool).columns:
                        df[col] = df[col].astype(str)
                    st.dataframe(df.head(20))

                    # Save subset to session state
                    if 'saved_subsets' not in st.session_state:
                        st.session_state['saved_subsets'] = {}

                    if subset_name.strip() != "":
                        st.session_state['saved_subsets'][subset_name] = subset_adata
                        st.success(f"Subset saved as `{subset_name}`.")
                    else:
                        st.warning("Please enter a valid subset name.")
                else:
                    st.warning("No subsetting applied. Please check your selections.")


        elif task == 'Subset and filter cells (gene expression)':
            st.subheader('Subset and filter cells based on gene expressions')

            adata.var_names = adata.var_names.str.lower()
            genes = adata.var_names.tolist()

            gene_input = st.multiselect('Select gene names:', genes)

            if gene_input:
                gene_filters = {}
                for gene in gene_input:
                    gene_expr = adata[:, gene].X
                    gene_expr = gene_expr.toarray().flatten() if hasattr(gene_expr, "toarray") else np.array(gene_expr).flatten()

                    min_val = float(np.min(gene_expr))
                    max_val = float(np.max(gene_expr))

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        operator = st.selectbox(
                            f"Operator for {gene}",
                            ['>', '>=', '<', '<=', '='],
                            key=f"{gene}_op"
                        )
                    with col2:
                        value = st.number_input(
                            f"Expression threshold for {gene}",
                            min_value=min_val,
                            max_value=max_val,
                            value=0.0,
                            key=f"{gene}_val"
                        )

                    gene_filters[gene] = (operator, value)

                # Persist this above the button to survive rerun
                subset_name = st.text_input("Name your subset:", value="_filtered_by_expression")

                if st.button('Subset'):
                    if not subset_name.strip():
                        st.warning("Please enter a subset name.")
                    else:
                        mask = np.ones(adata.n_obs, dtype=bool)

                        for gene, (op, threshold) in gene_filters.items():
                            gene_expr = adata[:, gene].X
                            gene_expr = gene_expr.toarray().flatten() if hasattr(gene_expr, "toarray") else np.array(gene_expr).flatten()

                            # Apply the operator
                            if op == '>':
                                gene_mask = gene_expr > threshold
                            elif op == '>=':
                                gene_mask = gene_expr >= threshold
                            elif op == '<':
                                gene_mask = gene_expr < threshold
                            elif op == '<=':
                                gene_mask = gene_expr <= threshold
                            elif op == '=':
                                gene_mask = gene_expr == threshold
                            else:
                                raise ValueError(f"Unsupported operator: {op}")

                            mask &= gene_mask

                        subset_adata = adata[mask].copy()

                        if subset_adata.n_obs == 0:
                            st.warning("No cells matched the given expression criteria.")
                        else:
                            # Preview
                            st.success(f"Subset applied. {subset_adata.n_obs} cells retained.")
                            st.dataframe(subset_adata.obs.head(20))

                            # Save subset
                            if 'saved_subsets' not in st.session_state:
                                st.session_state['saved_subsets'] = {}

                            st.session_state['saved_subsets'][subset_name] = subset_adata
                            st.success(f"Subset saved as `{subset_name}`.")

        elif task == 'Annotate cells (gene expression)':
            st.subheader('Annotate cells based on gene expressions')

            genes = adata.var_names.tolist()
            gene_input = st.multiselect('Select gene names:', genes)

            if gene_input:
                gene_filters = {}
                for gene in gene_input:
                    gene_expr = adata[:, gene].X
                    gene_expr = gene_expr.toarray().flatten() if hasattr(gene_expr, "toarray") else np.array(gene_expr).flatten()

                    min_val = float(np.min(gene_expr))
                    max_val = float(np.max(gene_expr))

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        operator = st.selectbox(
                            f"Operator for {gene}",
                            ['>', '>=', '<', '<=', '='],
                            key=f"{gene}_op"
                        )
                    with col2:
                        value = st.number_input(
                            f"Expression threshold for {gene}",
                            min_value=min_val,
                            max_value=max_val,
                            value=0.0,
                            key=f"{gene}_val"
                        )

                    gene_filters[gene] = (operator, value)

                annotation_name = st.text_input("Name your annotation:", value="")

                if st.button("Annotate"):

                    ops = {
                        '>': op.gt,
                        '>=': op.ge,
                        '<': op.lt,
                        '<=': op.le,
                        '==': op.eq,
                        '=': op.eq
                    }

                    # Start with all cells passing
                    mask = np.ones(adata.n_obs, dtype=bool)

                    for gene, (operator_str, value) in gene_filters.items():
                        gene_expr = adata[:, gene].X
                        gene_expr = gene_expr.toarray().flatten() if hasattr(gene_expr, "toarray") else np.array(gene_expr).flatten()

                        mask &= ops[operator_str](gene_expr, value)

                    adata.obs[annotation_name] = mask

                    st.session_state.adata = adata
                    st.success(f"Annotation '{annotation_name}' added to `adata.obs`.")

                    # show .obs preview
                    st.subheader('Cell-level info (.obs)')
                    df = adata.obs.head(20).copy()
                    for col in df.select_dtypes(include=bool).columns:
                        df[col] = df[col].astype(str)
                    st.dataframe(df)
                            
        elif task == 'Re-analyze clusters':            
            st.subheader('Re-analyze clusters')

            # get clustering keys (e.g., 'leiden', 'louvain')
            clustering_cols = [col for col in adata.obs.columns if col.startswith("leiden") or col.startswith("louvain")] 
            clustering_key = st.selectbox('Choose clustering key:', clustering_cols)

            # Get unique cluster labels
            clusters = adata.obs[clustering_key].unique().tolist()
            selected_clusters = st.multiselect('Choose clusters to re-analyze:', sorted(clusters))
            
            st.subheader('PCA')
            n_pcs_input = st.text_input("Number of principal components (default: None)", "")
            if n_pcs_input.strip() == "":
                n_pcs = None
            else:
                try:
                    n_pcs = int(n_pcs_input)
                    if n_pcs < 1:
                        st.error("Number of PCs must be positive.")
                        n_pcs = None
                except ValueError:
                    st.error("Please enter a valid integer.")
                    n_pcs = None

            st.subheader('Non-linear Dimensionality Reduction')
            embedding_method = st.selectbox("Choose embedding method:", ["UMAP", "t-SNE"])

            if embedding_method == 'UMAP':
                umap_n_pcs = st.text_input('Number of PCs for UMAP (default: None)', '')
                if umap_n_pcs.strip() == "":
                    n_pcs_umap = None
                else:
                    try:
                        n_pcs_umap = int(umap_n_pcs)
                        if n_pcs_umap < 1:
                            st.error("Number of PCs must be positive.")
                            n_pcs_umap = None
                    except ValueError:
                        st.error("Please enter a valid integer.")
                        n_pcs_umap = None

            elif embedding_method == 't-SNE':
                col1, col2 = st.columns(2)
                with col1:
                    perplexity = st.slider("Perplexity", 5, 100, 30)
                    early_exaggeration = st.slider("Early exaggeration", 5, 50, 12)
                with col2:
                    learning_rate = st.slider("Learning rate", 10, 1000, 200)

            st.subheader('Clustering')
            clustering_methods = st.selectbox('Choose clustering method:', ['leiden', 'louvain'])
            col1, col2 = st.columns(2)
            with col1:
                resolution = st.slider("Resolution", 0.1, 2.0, 0.5, 0.1)
            with col2:
                n_iterations = st.number_input("Number of iterations", 1, 10, 2)

            if st.button('Re-analyze'):
                with st.spinner("Re-analyzing selected clusters..."):
                    random.seed(0)
                    # Subset the data
                    subset_adata = adata[adata.obs[clustering_key].isin(selected_clusters)].copy()

                    # Reset to raw if available
                    if subset_adata.raw is not None:
                        subset_adata.X = subset_adata.raw.X

                        # Normalize and log-transform
                        sc.pp.normalize_total(subset_adata, target_sum=1e4)
                        sc.pp.log1p(subset_adata)
                    else:
                        st.warning("Raw data not found — using current X matrix without re-normalization.")

                    # Convert sparse to dense for safe checking
                    X_dense = subset_adata.X.toarray() if scipy.sparse.issparse(subset_adata.X) else subset_adata.X

                    # NaN or zero check
                    if np.any(np.isnan(X_dense)):
                        st.error("Expression matrix contains NaNs. Please check raw data or pre-processing.")
                    elif np.all(X_dense == 0):
                        st.error("All-zero expression matrix. Nothing to analyze.")
                    else:
                        # regress_out, scale, PCA
                        sc.pp.regress_out(subset_adata, ['total_counts'])
                        sc.pp.scale(subset_adata)
                        sc.tl.pca(subset_adata, n_comps=n_pcs, svd_solver='arpack')

                        # umap and t-sne
                        if embedding_method == "UMAP" and 'X_pca' in subset_adata.obsm:
                            sc.pp.neighbors(subset_adata, n_pcs=n_pcs_umap)
                            sc.tl.umap(subset_adata)
                            st.session_state.subset_adata = subset_adata
                            st.success("UMAP complete!")

                        elif embedding_method == "t-SNE" and 'X_pca' in subset_adata.obsm:
                            sc.tl.tsne(subset_adata, perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate)
                            st.session_state.subset_adata = subset_adata
                            st.success("t-SNE complete!")
                            
                            # plot t-sne
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sc.pl.tsne(subset_adata, ax=ax, show=False, frameon=False)
                            st.pyplot(fig)
                            plt.clf()
                        
                        # clustering
                        if clustering_methods == 'leiden':
                            sc.tl.leiden(subset_adata, resolution=resolution, n_iterations=n_iterations, key_added=f'leiden_{resolution}')
                            st.success("Leiden clustering complete!")
                        elif clustering_methods == 'louvain':
                            sc.tl.louvain(subset_adata, resolution=resolution, key_added=f'louvain_{resolution}')
                            st.success("Louvain clustering complete!")
                        
                        # save the subset for visualization
                        subset_key = f"reanalyzed_{clustering_methods}_{resolution}"
                        if 'saved_subsets' not in st.session_state:
                            st.session_state.saved_subsets = {}
                        st.session_state.saved_subsets[subset_key] = subset_adata
                        st.success(f"Subset saved under key: {subset_key}")

                        # plot PCA results from subset
                        if 'X_pca' in subset_adata.obsm:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("PCA (Subset)")
                                sc.pl.pca(subset_adata, show=False)
                                st.pyplot(plt.gcf())
                                plt.clf()
                            
                            with col2:
                                st.subheader("PCA Plot (Subset) Colored by Clusters")
                                sc.pl.pca(subset_adata, show=False, color='sample' if 'sample' in adata.obs else None,
                                        title='',
                                        legend_loc='upper right')
                                st.pyplot(plt.gcf())
                                plt.clf()

                            col3, col4 = st.columns(2)

                            with col3:
                                st.subheader(f"{embedding_method} Plot (Subset)")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                if embedding_method == 'UMAP':
                                    sc.pl.umap(subset_adata, ax=ax, show=False)
                                else:
                                    sc.pl.tsne(subset_adata, ax=ax, show=False)
                                st.pyplot(fig)
                                plt.clf()
                            
                            with col4:
                                st.subheader(f"{embedding_method} Colored by Clusters (Subset)")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                if embedding_method == 'UMAP':
                                    sc.pl.umap(subset_adata, color=[f'leiden_{resolution}' if clustering_methods == 'leiden' else f'louvain_{resolution}'], 
                                            ax=ax, 
                                            title='',
                                            legend_loc='upper right',
                                            show=False)
                                else:
                                    sc.pl.tsne(subset_adata, color=[f'leiden_{resolution}' if clustering_methods == 'leiden' else f'louvain_{resolution}'], ax=ax, show=False)
                                st.pyplot(fig)
                                plt.clf()

elif mode == 'Advanced Analysis' and selected_step == 'Gene-level Manipulation':
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Gene-level Manipulation</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Gene-level manipulation allows you to run differential gene expression and co-expression analyses.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.adata.copy()

        task = st.selectbox('Choose a task:', ['Differential gene expression analysis', 'Co-expression analysis'])

        if task == 'Differential gene expression analysis':
            st.subheader('Differential Gene Expression Analysis')
            
            # Get categorical columns for grouping
            categorical_cols = adata.obs.select_dtypes(['category', 'object']).columns.tolist()
            if not categorical_cols:
                st.error("No categorical columns found in adata.obs for grouping.")
                st.stop()
            
            groupby = st.selectbox("Group by (`.obs` column):", categorical_cols)
            
            # Ensure the selected column is categorical
            if not pd.api.types.is_categorical_dtype(adata.obs[groupby]):
                adata.obs[groupby] = adata.obs[groupby].astype('category')
            
            group_vals = adata.obs[groupby].cat.categories.tolist()
            
            # Show group distribution
            with st.expander("View group distribution"):
                group_counts = adata.obs[groupby].value_counts()
                st.write("Group sizes:")
                st.dataframe(group_counts.reset_index())
                
                # Warning for small groups
                small_groups = group_counts[group_counts < 10]
                if len(small_groups) > 0:
                    st.warning(f"Small groups detected (< 10 cells): {small_groups.index.tolist()}")
            
            st.subheader('Comparison Setup')
            with st.expander("How to set up comparisons", expanded=False):
                st.markdown("""
                **Reference Group**: The baseline for comparison (denominator in fold change)
                - **"Rest"**: Compare each group against all other groups combined
                - **Specific group**: Use one group as the reference baseline
                
                **Target Group**: The group(s) being tested (numerator in fold change)  
                - **"All"**: Test all groups (except reference if specified)
                - **Specific group**: Test only one group against the reference
                
                **Example**: Reference="Control", Target="Treatment" → finds genes upregulated in Treatment vs Control
                """)
            col1, col2 = st.columns(2)
            
            with col1:
                reference_group = st.selectbox("Reference group:", ['Rest'] + group_vals)
                
            with col2:
                if reference_group == 'Rest':
                    target_group = st.selectbox("Target group:", ['All'] + group_vals, index=0)
                else:
                    target_options = ['All'] + [g for g in group_vals if g != reference_group]
                    target_group = st.selectbox("Target group:", target_options, index=0)
            
            # Advanced options
            with st.expander("Advanced Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    method = st.selectbox("Test method:", 
                                        ["wilcoxon", "t-test", "logreg", "t-test_overestim_var"],
                                        help="Wilcoxon is recommended for single-cell data")
                    use_raw = st.checkbox("Use raw counts", value=False, 
                                        help="Use raw counts instead of normalized data")
                    
                with col2:
                    n_genes = st.number_input('Number of top genes to show:', 
                                            min_value=1, max_value=100, value=10)
                    min_fold_change = st.number_input('Minimum log fold change:', 
                                                    min_value=0.0, max_value=5.0, value=0.25, step=0.25)
            
            comparison_name = st.text_input("Name this comparison:", 
                                        value=f'DEG_{groupby}_{reference_group}_vs_{target_group}')
            
            # Validation
            if not comparison_name.strip():
                st.error("Please provide a comparison name.")
                st.stop()
            
            if comparison_name in adata.uns:
                st.warning(f"Comparison '{comparison_name}' already exists and will be overwritten.")
            
            if st.button("Run Differential Expression Analysis"):
                with st.spinner('Identifying differentially expressed genes...'):
                    try:
                        # Validate inputs
                        if reference_group != 'Rest' and reference_group not in group_vals:
                            st.error(f"Reference group `{reference_group}` not found in `{groupby}` column.")
                            st.stop()
                        
                        if target_group != 'All' and target_group not in group_vals:
                            st.error(f"Target group `{target_group}` not found in `{groupby}` column.")
                            st.stop()
                        
                        # Set up parameters
                        rank_params = {
                            'adata': adata,
                            'groupby': groupby,
                            'method': method,
                            'key_added': comparison_name,
                            'min_fold_change': min_fold_change
                        }
                        
                        # Run differential expression based on comparison type
                        if reference_group == 'Rest':
                            if target_group == 'All':
                                rank_params['reference'] = 'rest'
                            else:
                                rank_params['reference'] = 'rest'
                                rank_params['groups'] = [target_group]
                        else:
                            rank_params['reference'] = reference_group
                            if target_group != 'All':
                                rank_params['groups'] = [target_group]
                            else:
                                rank_params['groups'] = 'all'
                        
                        # Run the analysis
                        sc.tl.rank_genes_groups(**rank_params)
                        
                        # Update session state
                        st.session_state.adata = adata
                        
                        st.success(f"Differential expression analysis completed!")
                        
                        # Display results
                        st.subheader(f'Results: {comparison_name}')
                        
                        # Get results
                        results = adata.uns[comparison_name]
                        
                        # Check if results exist
                        if 'names' not in results:
                            st.error("No results found. This might indicate an issue with the analysis.")
                            st.stop()
                        
                        groups = results['names'].dtype.names if hasattr(results['names'], 'dtype') else [comparison_name]
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Groups compared", len(groups))
                        with col2:
                            st.metric("Genes per group", len(results['names'][groups[0]] if groups else []))
                        with col3:
                            st.metric("Test method", method.upper())
                        
                        # Plot top genes
                        if n_genes > 0:
                            st.subheader(f'Top {n_genes} Differentially Expressed Genes')
                            try:
                                fig = sc.pl.rank_genes_groups(
                                    adata,
                                    n_genes=n_genes,
                                    sharey=False,
                                    key=comparison_name,
                                    show=False,
                                    return_fig=True
                                )
                                
                                if isinstance(fig, list):
                                    fig = fig[0].get_figure()
                                elif hasattr(fig, 'get_figure'):
                                    fig = fig.get_figure()
                                
                                st.pyplot(fig)
                                plt.clf()
                                
                            except Exception as e:
                                st.error(f"Error creating plot: {str(e)}")
                        
                        # Create comprehensive results table
                        st.subheader('Detailed Results')
                        
                        top_genes = []
                        for group in groups:
                            names = results['names'][group]
                            scores = results['scores'][group]
                            
                            # Handle different p-value types
                            if 'pvals_adj' in results:
                                pvals = results['pvals_adj'][group]
                                pval_type = 'Adjusted p-value'
                            else:
                                pvals = results['pvals'][group]
                                pval_type = 'p-value'
                            
                            # Add log fold changes if available (handle recarray properly)
                            logfoldchanges = None
                            if 'logfoldchanges' in results:
                                logfoldchanges = results['logfoldchanges'][group]
                            else:
                                logfoldchanges = [None] * len(names)
                            
                            for rank, (gene, score, pval, lfc) in enumerate(zip(names, scores, pvals, logfoldchanges), start=1):
                                if pd.isna(gene) or gene == '' or str(gene) == 'nan':  # Skip invalid entries
                                    continue
                                    
                                gene_data = {
                                    'Group': group,
                                    'Rank': rank,
                                    'Gene': gene,
                                    'Score': float(score) if pd.notna(score) else 0.0,
                                    pval_type: float(pval) if pd.notna(pval) else 1.0,
                                }
                                
                                if lfc is not None and pd.notna(lfc):
                                    gene_data['Log Fold Change'] = float(lfc)
                                    
                                top_genes.append(gene_data)
                        
                        if top_genes:
                            deg_df = pd.DataFrame(top_genes)
                            
                            # Format p-values in scientific notation
                            pval_col = 'Adjusted p-value' if 'Adjusted p-value' in deg_df.columns else 'p-value'
                            deg_df[pval_col] = deg_df[pval_col].apply(lambda x: f'{x:.2e}' if pd.notna(x) else 'N/A')
                            
                            # Add filtering options
                            col1, col2 = st.columns(2)
                            with col1:
                                if len(groups) > 1:
                                    selected_groups = st.multiselect("Filter by group:", groups, default=groups)
                                    filtered_df = deg_df[deg_df['Group'].isin(selected_groups)]
                                else:
                                    selected_groups = groups
                                    filtered_df = deg_df

                            with col2:
                                max_rank = st.number_input("Show top N genes per group:", 
                                                        min_value=1, max_value=len(filtered_df), 
                                                        value=min(20, len(filtered_df)))
                                display_df = filtered_df.groupby('Group').head(max_rank)

                            # Display the filtered/limited dataframe
                            st.dataframe(display_df, use_container_width=True)

                            # Download option - use the full results, not the display-limited version
                            download_df = deg_df[deg_df['Group'].isin(selected_groups)] if len(groups) > 1 else deg_df
                            csv = download_df.to_csv(index=False)
                            st.download_button(
                                label=f"Download all results as CSV ({len(download_df)} genes)",
                                data=csv,
                                file_name=f"{comparison_name}_results.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No valid results to display.")
                            
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
                        st.error("Please check your data and parameters.")
                        
            # Show existing comparisons
            if adata.uns:
                deg_keys = [key for key in adata.uns.keys() if key.startswith('DEG_') or 'rank_genes_groups' in key]
                if deg_keys:
                    with st.expander(f"View Previous Comparisons ({len(deg_keys)} found)"):
                        for key in deg_keys:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"• {key}")
                            with col2:
                                if st.button("Delete", key=f"del_{key}"):
                                    del adata.uns[key]
                                    st.session_state.adata = adata
                                    st.rerun()
                
        elif task == 'Co-expression analysis':
            gene_list = adata.var_names.tolist()

            analysis_type = st.radio("Analysis type", ["Pairwise correlation", "Gene scatter plot"])

            if analysis_type == "Pairwise correlation":
                selected_genes = st.multiselect("Select gene names:", gene_list)
                method = st.radio("Correlation method:", ["pearson", "spearman"])

                if st.button('Run co-expression analysis:'):

                    if len(selected_genes) >= 2:
                        expr = adata[:, selected_genes].X.toarray() if scipy.sparse.issparse(adata.X) else adata[:, selected_genes].X
                        df = pd.DataFrame(expr, columns=selected_genes)
                        corr = df.corr(method=method)

                        st.subheader("Correlation matrix")
                        st.dataframe(corr.round(2))

                        fig, ax = plt.subplots()
                        sns.heatmap(corr, annot=True, cmap="vlag", ax=ax)
                        st.pyplot(fig)
                    else:
                        st.warning("Please enter at least two valid genes for correlation.")

            elif analysis_type == "Gene scatter plot":
                col1, col2 = st.columns(2)
                with col1:
                    gene_x = st.selectbox("Gene X", gene_list)
                with col2:
                    gene_y = st.selectbox("Gene Y", gene_list)

                if st.button('Run co-expression analysis:'):
                    x = adata[:, gene_x].X.toarray().flatten() if scipy.sparse.issparse(adata.X) else adata[:, gene_x].X.flatten()
                    y = adata[:, gene_y].X.toarray().flatten() if scipy.sparse.issparse(adata.X) else adata[:, gene_y].X.flatten()

                    fig, ax = plt.subplots()
                    ax.scatter(x, y, s=10, alpha=0.4)
                    ax.set_xlabel(gene_x)
                    ax.set_ylabel(gene_y)
                    ax.set_title(f"Expression: {gene_x} vs {gene_y}")
                    st.pyplot(fig)


elif mode == 'Advanced Analysis' and selected_step == 'Visualization':
    if 'adata' not in st.session_state or st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Gene-level Manipulation</h2>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Visualize anything you'd like!
        </div>
        """, unsafe_allow_html=True)

        # collect all available datasets: original + optional saved subsets
        datasets = {"Original dataset": st.session_state.adata}
        if 'saved_subsets' in st.session_state and isinstance(st.session_state.saved_subsets, dict):
            datasets.update(st.session_state.saved_subsets)

        # select which one to visualize
        selected_key = st.selectbox("Select a dataset to visualize:", list(datasets.keys()))

        # load the selected dataset
        adata = datasets[selected_key].copy()
        adata.var_names = adata.var_names.str.lower()

        # viz options
        viz_type = st.selectbox(
            "Choose visualization type:",
            ["Gene expression", 'Clusters and cell groups', "Quality metrics"]
        )

        if viz_type == 'Gene expression':
            st.subheader('Gene expression visualization')

            # Use normalized data if available
            if 'norm_log' in adata.layers:
                adata_for_plot = adata.copy()
                adata_for_plot.X = adata.layers['norm_log'].copy()

                # set .raw to normalized data
                adata_for_plot.raw = adata_for_plot
            else:   
                st.warning('Normalized and logarithmized data not found. Please make sure you run Normalization & Logarithmization for your dataset.')
                adata_for_plot = adata

            # Gene list (keep original casing, allow case-insensitive matching)
            gene_list = adata_for_plot.var_names.tolist()
            gene_names = st.multiselect('Select gene names:', gene_list)

            if not gene_names:
                st.info("Please select at least one gene.")
            else:
                plot_type = st.selectbox('Plot type:', [
                    'Scatter plot', 'Violin plot', 'Dot plot', 
                    'Heatmap', 'Matrix plot', 'Stacked violin plot'
                ])

                if plot_type == 'Scatter plot':
                    projection_type = st.selectbox("Projection type:", ["UMAP", "t-SNE", "PCA"])

                    with st.expander('Plot Options', expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            point_size = st.slider('Point size:', 1, 20, 5)
                            alpha = st.slider('Point transparency:', 0.0, 1.0, 0.8, step=0.1)
                        with col2:
                            color_map = st.selectbox('Color map:', [
                                'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gnuplot',
                                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 
                            ], index=0)

                    if st.button('Plot Gene Expression') and projection_type is not None:
                        n_genes = len(gene_names)
                        n_cols = min(3, n_genes)
                        n_rows = (n_genes + n_cols - 1) // n_cols

                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                        axes = axes.flatten() if n_genes > 1 else [axes]

                        for i, gene in enumerate(gene_names):
                            ax = axes[i]
                            if projection_type == "UMAP" and "X_umap" in adata_for_plot.obsm:
                                sc.pl.umap(adata_for_plot, color=gene, ax=ax, ncols=3, frameon=False, show=False, size=point_size, alpha=alpha, cmap=color_map)
                            elif projection_type == "t-SNE" and "X_tsne" in adata_for_plot.obsm:
                                sc.pl.tsne(adata_for_plot, color=gene, ax=ax, ncols=3, frameon=False, show=False, size=point_size, alpha=alpha, cmap=color_map)
                            elif projection_type == "PCA" and "X_pca" in adata_for_plot.obsm:
                                sc.pl.pca(adata_for_plot, color=gene, ax=ax, ncols=3, frameon=False, show=False, size=point_size, alpha=alpha, cmap=color_map)
                            ax.set_title(gene)

                        for j in range(i + 1, len(axes)):
                            fig.delaxes(axes[j])  # remove empty subplots

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                elif plot_type == 'Violin plot':
                    if st.button("Plot Gene Expression"):
                            n_genes = len(gene_names)
                            n_cols = min(3, n_genes)
                            n_rows = (n_genes + n_cols - 1) // n_cols
                            
                            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                            axes = axes.flatten() if n_genes > 1 else [axes]
                            
                            for i, gene in enumerate(gene_names):
                                ax = axes[i]
                                sc.pl.violin(adata_for_plot, keys=gene, ax=ax, show=False)
                                ax.set_title(gene)
                            
                            for j in range(i + 1, len(axes)):
                                fig.delaxes(axes[j])  # remove empty subplots

                            plt.tight_layout()
                            st.pyplot(fig)

                elif plot_type == 'Dot plot':
                    clustering_keys = [col for col in adata.obs.columns if col.startswith("leiden") or col.startswith("louvain")]
                    if clustering_keys:
                        group = st.selectbox("Group by:", clustering_keys)
                    else:
                        st.warning("No clustering keys found in adata.obs.")
                        group = None

                    if st.button("Plot Gene Expression") and group is not None:
                        n_genes = len(gene_names)
                        n_cols = min(3, n_genes)
                        n_rows = (n_genes + n_cols - 1) // n_cols
                        
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                        axes = axes.flatten() if n_genes > 1 else [axes]
                        
                        for i, gene in enumerate(gene_names):
                            ax = axes[i]
                            sc.pl.dotplot(adata_for_plot, var_names=gene, groupby=group, ax=ax, show=False, dendrogram=True)
                            ax.set_title(gene)
                        
                        for j in range(i + 1, len(axes)):
                            fig.delaxes(axes[j])  # remove empty subplots

                        plt.tight_layout()
                        st.pyplot(fig)

                elif plot_type == 'Heatmap':
                    clustering_keys = [col for col in adata.obs.columns if col.startswith("leiden") or col.startswith("louvain")]
                    if clustering_keys:
                        group = st.selectbox("Group by:", clustering_keys)
                    else:
                        st.warning("No clustering keys found in adata.obs.")
                        group = None

                    if st.button("Plot Gene Expression") and group is not None:
                        n_genes = len(gene_names)
                        width = 8  # Fixed width
                        height = max(4, n_genes * 0.5 + 2)

                        sc.pl.heatmap(adata_for_plot, var_names=gene_names, groupby=group, dendrogram=True, show=False, figsize=(width, height), swap_axes=True)
                        fig = plt.gcf()  
                        st.pyplot(fig)
                        plt.close(fig)

                elif plot_type == 'Matrix plot':
                    clustering_keys = [col for col in adata.obs.columns if col.startswith("leiden") or col.startswith("louvain")]
                    if clustering_keys:
                        group = st.selectbox("Group by:", clustering_keys)
                    else:
                        st.warning("No clustering keys found in adata.obs.")
                        group = None

                    if st.button("Plot Gene Expression") and group is not None:
                        n_genes = len(gene_names)
                        width = 8  # Fixed width
                        height = max(4, n_genes * 0.5 + 2)

                        sc.pl.matrixplot(adata_for_plot, var_names=gene_names, groupby=group, dendrogram=True, show=False, figsize=(width, height), swap_axes=True)
                        fig = plt.gcf() 
                        st.pyplot(fig)
                        plt.close(fig)

                elif plot_type == 'Stacked violin plot':
                    clustering_keys = [col for col in adata.obs.columns if col.startswith("leiden") or col.startswith("louvain")]
                    if clustering_keys:
                        group = st.selectbox("Group by:", clustering_keys)
                    else:
                        st.warning("No clustering keys found in adata.obs.")
                        group = None

                    if st.button("Plot Gene Expression") and group is not None:
                        n_genes = len(gene_names)
                        width = 8  # Fixed width
                        height = max(4, n_genes * 0.5 + 2)

                        sc.pl.stacked_violin(adata_for_plot, var_names=gene_names, groupby=group, dendrogram=True, show=False, figsize=(width, height), swap_axes=True)
                        fig = plt.gcf() 
                        st.pyplot(fig)
                        plt.close(fig)

        # elif viz_type == "Cluster analysis":
        #     st.subheader('Visualize clusters')

        #     # Find clustering keys
        #     cluster_keys = [col for col in adata.obs.columns if col.startswith("leiden") or col.startswith("louvain")]

        #     # Ensure projection options are available
        #     projection_options = [proj.replace('X_', '') for proj in adata.obsm.keys() if proj.startswith('X_')]
        #     if not projection_options:
        #         st.warning("No projection found in `adata.obsm` starting with 'X_'.")
        #     else:
        #         projection_type = st.selectbox('Projection type:', projection_options)

        #         if not cluster_keys:
        #             st.warning("No cluster annotations found.")
        #         else:
        #             cluster_key = st.selectbox("Select clustering key:", cluster_keys)
        #             if st.button("Plot Clusters"):
        #                 fig, ax = plt.subplots(figsize=(8, 6))
        #                 if projection_type.lower() == 'umap':
        #                     sc.pl.umap(adata, color=cluster_key, ax=ax, frameon=False, show=False)
        #                 elif projection_type.lower() == 'tsne':
        #                     sc.pl.tsne(adata, color=cluster_key, ax=ax, frameon=False, show=False)
        #                 elif projection_type.lower() == 'pca':
        #                     sc.pl.pca(adata, color=cluster_key, ax=ax, frameon=False, show=False)
        #                 else:
        #                     st.warning(f"No plotting function defined for projection: {projection_type}")
        #                 st.pyplot(fig)
        
        # elif viz_type == 'Cell group annotations':
        #     st.subheader("Visualize cell groups")

        #     # Get all boolean columns from adata.obs
        #     bool_cols = [col for col in adata.obs.columns if pd.api.types.is_bool_dtype(adata.obs[col]) or pd.api.types.is_categorical_dtype(adata.obs[col])]

        #     if not bool_cols:
        #         st.warning("No boolean annotations found in `adata.obs`. Please annotate cells first.")
        #     else:
        #         annotation_key = st.selectbox("Select annotation to visualize:", bool_cols)

        #         # Get available projections from adata.obsm
        #         projection_options = [proj.replace("X_", "") for proj in adata.obsm.keys() if proj.startswith("X_")]
        #         if not projection_options:
        #             st.warning("No dimensionality reduction projections found (e.g., UMAP, tSNE in `adata.obsm`).")
        #         else:
        #             projection_type = st.selectbox("Select projection type:", projection_options)

        #             if st.button("Plot Annotated Cells"):
        #                 fig, ax = plt.subplots(figsize=(8, 6))

        #                 if projection_type.lower() == "umap":
        #                     sc.pl.umap(adata, color=annotation_key, ax=ax, frameon=False, show=False)
        #                 elif projection_type.lower() == "tsne":
        #                     sc.pl.tsne(adata, color=annotation_key, ax=ax, frameon=False, show=False)
        #                 else:
        #                     # Fallback to generic embedding
        #                     embedding_key = f"X_{projection_type}"
        #                     if embedding_key in adata.obsm:
        #                         sc.pl.embedding(adata, basis=embedding_key.replace("X_", ""), color=annotation_key, ax=ax, frameon=False, show=False)
        #                     else:
        #                         st.error(f"Projection '{projection_type}' not found in `adata.obsm`.")

        #                 st.pyplot(fig)
        elif viz_type == "Clusters and cell groups":
            st.subheader('Visualize clusters and cell groups')
            
            # get all categorical columns (including boolean, string categories, and clustering)
            categorical_cols = []
            for col in adata.obs.columns:
                if (pd.api.types.is_categorical_dtype(adata.obs[col]) or 
                    pd.api.types.is_bool_dtype(adata.obs[col]) or
                    pd.api.types.is_string_dtype(adata.obs[col]) or
                    col.startswith("leiden") or 
                    col.startswith("louvain")):
                    categorical_cols.append(col)
            
            # ensure projection options are available
            projection_options = [proj.replace('X_', '') for proj in adata.obsm.keys() if proj.startswith('X_')]
            
            if not projection_options:
                st.warning("No projection found in `adata.obsm` starting with 'X_'.")
            elif not categorical_cols:
                st.warning("No categorical annotations found in `adata.obs`.")
            else:
                annotation_key = st.selectbox("Select annotation to visualize:", categorical_cols)
                projection_type = st.selectbox('Projection type:', projection_options)
                
                if st.button("Plot Annotation"):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    if projection_type.lower() == 'umap':
                        sc.pl.umap(adata, color=annotation_key, ax=ax, frameon=False, show=False)
                    elif projection_type.lower() == 'tsne':
                        sc.pl.tsne(adata, color=annotation_key, ax=ax, frameon=False, show=False)
                    elif projection_type.lower() == 'pca':
                        sc.pl.pca(adata, color=annotation_key, ax=ax, frameon=False, show=False)
                    else:
                        # Fallback to generic embedding
                        embedding_key = f"X_{projection_type}"
                        if embedding_key in adata.obsm:
                            sc.pl.embedding(adata, basis=projection_type, color=annotation_key, ax=ax, frameon=False, show=False)
                        else:
                            st.error(f"Projection '{projection_type}' not found in `adata.obsm`.")
                            st.stop()
                    
                    st.pyplot(fig)

        elif viz_type == "Quality metrics":
            st.subheader('Visualize quality metrics')

            # Filter for common QC metrics patterns in observations
            qc_obs_patterns = ['n_genes', 'n_counts', 'total_counts', 'pct_counts', 'percent_', 'pct_', 'mt_', 'ribo_', 'hb_', 'doublet', 'scrublet']
            qc_metrics_col = [col for col in adata.obs.columns 
                            if any(pattern in col.lower() for pattern in qc_obs_patterns) 
                            and not col.startswith(('leiden_', 'louvain_',)) 
                            and 'highly_variable' not in col.lower()]
            
            # Filter for common QC metrics patterns in variables
            qc_var_patterns = ['mean', 'std', 'var', 'cv', 'dropout', 'highly_variable', 'dispersions', 'pct_dropout']
            qc_metrics_var = [col for col in adata.var.columns 
                            if any(pattern in col.lower() for pattern in qc_var_patterns)]

            all_qc_metrics = qc_metrics_col + qc_metrics_var

            if not all_qc_metrics:
                st.warning("No QC metrics found in the data.")

            qc_genes = st.multiselect("Select QC metrics to visualize:", all_qc_metrics)

            if qc_genes:
                # Separate obs and var metrics
                obs_metrics = [metric for metric in qc_genes if metric in adata.obs.columns]
                var_metrics = [metric for metric in qc_genes if metric in adata.var.columns]
                
                # Plot observation metrics (if any)
                if obs_metrics:
                    st.write("**Cell-level QC Metrics (Observation Data):**")
                    sc.pl.violin(
                        adata,
                        obs_metrics,
                        jitter=0.4,
                        multi_panel=True,
                        show=False
                    )
                    fig = plt.gcf()
                    st.pyplot(fig)
                    plt.close(fig)
        
                # Plot variable metrics (if any)
                if var_metrics:
                    st.write("**Gene-level QC Metrics (Variable Data):**")
                    for metric in var_metrics:
                        # Create histogram for each variable metric
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(adata.var[metric].dropna(), bins=50, alpha=0.7, edgecolor='black')
                        ax.set_xlabel(metric)
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Distribution of {metric}')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)


elif mode == 'Advanced Analysis' and selected_step == 'Export Results':
    st.subheader("Export AnnData Objects")
    
    # Collect top-level AnnData objects
    top_level_keys = {key: st.session_state[key] for key in st.session_state if isinstance(st.session_state[key], ad.AnnData)}
    
    # Collect subset AnnData objects from st.session_state["saved_subsets"]
    subset_dict = st.session_state.get("saved_subsets", {})
    subset_keys = {f"subset_{key}": val for key, val in subset_dict.items() if isinstance(val, ad.AnnData)}
    
    # Combine all into one dictionary
    all_adata_dict = {**top_level_keys, **subset_keys}
    
    if not all_adata_dict:
        st.warning("No AnnData objects found in session state.")
    else:
        selected_keys = st.multiselect(
            "Select AnnData objects to export:", 
            list(all_adata_dict.keys()), 
            default=list(all_adata_dict.keys())
        )
        
        if selected_keys:
            # Export method selection
            export_method = st.radio(
                "Export method:",
                ["Single combined file", "Individual files"],
                help="Single file combines all selected objects, individual files exports each separately"
            )
            
            if st.button("Export Results", key="export_results"):
                timestamp = int(time.time())
                downloads_path = os.path.expanduser("~/Downloads")
                
                with st.spinner("Exporting to Downloads folder..."):
                    try:
                        if export_method == "Single combined file":
                            # Export single combined file (using first selected as primary)
                            primary_adata = all_adata_dict[selected_keys[0]]
                            filename = f"exported_results_{timestamp}.h5ad"
                            full_path = os.path.join(downloads_path, filename)
                            primary_adata.write_h5ad(full_path)
                            
                            file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
                            st.success(f"Export complete!")
                            st.info(f"**File:** `{filename}`")
                            st.info(f"**Size:** {file_size_mb:.1f} MB")
                            st.info(f"**Location:** `{full_path}`")
                            
                        else:  # Individual files
                            exported_files = []
                            total_size_mb = 0
                            
                            for key in selected_keys:
                                adata_obj = all_adata_dict[key]
                                filename = f"{key}_{timestamp}.h5ad"
                                full_path = os.path.join(downloads_path, filename)
                                adata_obj.write_h5ad(full_path)
                                
                                file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
                                total_size_mb += file_size_mb
                                exported_files.append((filename, file_size_mb))
                            
                            st.success(f"Export complete! {len(exported_files)} files exported.")
                            st.info(f"**Total size:** {total_size_mb:.1f} MB")
                            st.info(f"**Location:** `{downloads_path}`")
                            
                            # Show details of exported files
                            with st.expander("Exported files details"):
                                for filename, size_mb in exported_files:
                                    st.write(f"• `{filename}` ({size_mb:.1f} MB)")
                                    
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
                        st.error("Please check that the Downloads folder is accessible and you have write permissions.")
    
    st.markdown("---")
    
    # Reset button
    if st.button("Reset Analysis"):
        # Store keys to preserve (if any)
        keys_to_preserve = {"selected_step"}  # add keys that users want to keep
        
        # Create a list of keys to delete (avoiding dict modification during iteration)
        keys_to_reset = [key for key in st.session_state.keys() if key not in keys_to_preserve]
        
        # Clear session state variables
        for key in keys_to_reset:
            if key in st.session_state:  # Extra safety check
                del st.session_state[key]
        
        # Explicitly reset critical variables
        st.session_state.adata = None
        st.success("Session has been reset. Please upload a new dataset to begin again.")
        st.rerun()







            
        


                



            

        
                    
                









                



































