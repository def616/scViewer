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
warnings.filterwarnings('ignore')
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['figure.facecolor'] = 'white'

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
st.markdown('<h1 class="main-header">scViewer: Single Cell Analysis Made Easy</h1>', unsafe_allow_html=True)

# Sidebar for navigation
mode = st.sidebar.radio("Choose mode:", ["Standard Pipeline", "Advanced Analysis"], key="mode_selector")
# st.sidebar.title("Standard Analysis Pipeline")
pipeline_steps = [
    "Data Loading",
    "Quality Control",
    "Pre-processing", 
    "Normalization & Transformation",
    "Feature Selection",
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
        st.metric("Total Counts", f"{adata.X.sum():.0f}")

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
        sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts", color="pct_counts_mt", ax=axes[2,1])
        axes[2,1].set_title('Total Counts vs Genes (Colored by Mito %)')
    
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


# content based on selected step
if mode == "Standard Pipeline" and selected_step == "Data Loading":
    st.markdown('<h2 class="step-header">Step 1: Data Loading</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>What we're doing:</strong> Loading your single-cell RNA-seq data into an AnnData object.
    Supported format: H5AD
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your single-cell data file",
        type=['h5ad']
    )
    
    if uploaded_file is not None:
        try:
            # load h5ad file
            uploaded_file.name.endswith('.h5ad')
            adata = sc.read_h5ad(uploaded_file)
            
            # Store in session state
            st.session_state.adata = adata
            
            st.success(f"Successfully loaded {uploaded_file.name}")
            
            # Display data summary
            st.subheader("Dataset Overview")
            display_data_summary(adata)
            
            # Show first few genes and cells
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("First 10 Genes")
                st.dataframe(adata.var.head(10).reset_index())

            with col2:
                st.subheader("First 10 Cells")
                st.dataframe(adata.obs.head(10))
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}") 

elif mode == "Standard Pipeline" and selected_step == "Quality Control":
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Step 2: Quality Control</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Calculating quality control metrics. Scanpy's function, calculate_qc_metrics(), calculates common QC metrics. 
                    We can input a specific gene population in order to calculate proportions of counts for this population. For example, we can calculate common QC metrics for mitochondrial, hemoglobin, and ribosomal genes.
                    We also will predict doublet cells.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.adata.copy()
        
        # Calculate QC metrics
        species = st.selectbox('Species', ['Mouse', 'Human'])
        mito_gene = st.checkbox('Annotate mitochondrial genes', value=True)
        hemo_gene = st.checkbox('Annotate hemoglobin genes', value=True)
        ribo_gene = st.checkbox('Annotate ribosomal genes', value=True)

        qc_button = st.button("Calculate QC Metrics")

        if qc_button:
            with st.spinner("Calculating quality control metrics..."):
                adata.var_names_make_unique()

                # Determine species-specific mitochondrial gene prefix
                mito_prefix = 'mt-' if species == 'Mouse' else 'Mt-'
                adata.var['mt'] = adata.var_names.str.startswith(mito_prefix)

                # Add optional QC gene markers
                if ribo_gene:
                    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
                if hemo_gene:
                    if species == 'Mouse':
                        adata.var["hb"] = adata.var_names.str.contains(r"^Hb[ab]")
                    elif species == 'Human':
                        adata.var["hb"] = adata.var_names.str.contains(r"^HB[AB]")

                # Build qc_vars list dynamically
                qc_vars = ['mt']
                if ribo_gene:
                    qc_vars.append('ribo')
                if hemo_gene:
                    qc_vars.append('hb')

                # Run QC
                sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, log1p=True, inplace=True)

                # Predict doublets
                sc.pp.scrublet(adata, batch_key="sample")

                # Store result
                st.session_state.adata = adata
                st.success("✅ QC metrics calculated!")


        # Display QC metrics if available
        if 'total_counts' in adata.obs.columns:
            st.subheader("Quality Control Metrics")
            
            # Plot QC metrics
            fig = plot_qc_metrics(adata)
            st.pyplot(fig)

            # columns for viewing .obs and .var
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

elif mode == "Standard Pipeline" and selected_step == "Pre-processing":
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Step 3: Pre-processing</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Filtering out low-quality cells and genes based on QC metrics.
        This step removes cells and genes that don't meet our quality thresholds.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.adata.copy()
        
        if 'total_counts' not in adata.obs.columns:
            st.warning("Please calculate QC metrics first!")
        else:
            # Filtering parameters
            st.subheader("Filtering Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                min_genes = st.number_input("Min genes per cell", min_value=0, value=200)
            
            with col2:
                min_cells = st.number_input("Min cells per gene", min_value=0, value=3)
            
            if st.button("Apply Filtering"):
                with st.spinner("Filtering cells and genes..."):
                    n_cells_before = adata.n_obs
                    n_genes_before = adata.n_vars
                    
                    # Filter cells
                    sc.pp.filter_cells(adata, min_genes=min_genes)
                    sc.pp.filter_genes(adata, min_cells=min_cells)
                    
                    # Store filtered data
                    st.session_state.adata = adata
                    
                    st.success(f"✅ Filtering complete!")
                    st.info(f"Cells: {n_cells_before} → {adata.n_obs} ({n_cells_before - adata.n_obs} removed)")
                    st.info(f"Genes: {n_genes_before} → {adata.n_vars} ({n_genes_before - adata.n_vars} removed)")
                    
                    # Display updated summary
                    st.subheader("Filtered Dataset Summary")
                    display_data_summary(adata)

elif mode == "Standard Pipeline" and selected_step == "Normalization & Transformation":
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Step 4: Normalization & Transformation</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Normalizing gene expression data to a scale factor. Scanpy applies median count depth normalization with log1p transformation. 
                    In this step, Scanpy also removes effects of sequencing depth via the function regress_out(). It also centers each gene to a mean of zero and scales each gene to unit variance.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.adata.copy()
        
        # Normalization options
        st.subheader("Normalization Method")

        # input for normalization target sum
        target_sum = st.number_input("Target sum", min_value=0.0, value=1e4, step=1e3, format="%.0f")
        
        # checkbox for log1p
        log_transform = st.checkbox("Apply log(x+1) transformation", value=True)

        # checkbox for regress_out
        regress = st.checkbox('Regress out', value=True)

        # checkbox for scaling
        scaling = st.checkbox('Scale data', value=True)
        
        if st.button("Apply Normalization"):
            with st.spinner("Normalizing data..."):
                # Save raw data if not already saved
                if adata.raw is None:
                    adata.raw = adata.copy()
                
                # Normalize
                if target_sum == 0:
                    computed_target = np.median(adata.X.sum(axis=1))
                    sc.pp.normalize_total(adata, target_sum=computed_target)
                else:
                    sc.pp.normalize_total(adata, target_sum=target_sum)
                
                # Log transform
                if log_transform:
                    sc.pp.log1p(adata)

                # regress out
                if regress:
                    sc.pp.regress_out(adata, ['total_counts'])
                
                # scale
                if scaling:
                    sc.pp.scale(adata, max_value=10)
                
                # Store updated data
                st.session_state.adata = adata
                
                st.success("✅ Normalization complete!")

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

                # # show updated .obs and .var
                # col1, col2 = st.columns(2)

                # with col1:
                #     st.subheader('Cell-level info (.obs)')
                #     df = adata.obs.head(20).copy()
                #     for col in df.select_dtypes(include=bool).columns:
                #         df[col] = df[col].astype(str)
                #     st.dataframe(df)
            
                # with col2:
                #     st.subheader('Gene-level info (.var)')
                #     df = adata.var.head(20).copy()
                #     for col in df.select_dtypes(include=bool).columns:
                #         df[col] = df[col].astype(str)
                #     st.dataframe(df)


elif mode == "Standard Pipeline" and selected_step == "Feature Selection":
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Step 5: Feature Selection</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Identifying highly variable genes (HVGs) that drive cell-to-cell variation.
        These genes are most informative for downstream analysis.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.adata.copy()
        
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
        
                n_hvg = adata.var['highly_variable'].sum()
                st.success(f"✅ Found {n_hvg} highly variable genes!")
                
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


elif mode == "Standard Pipeline" and selected_step == "Dimensionality Reduction":
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Step 6: Dimensionality Reduction</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Reducing the dimensionality of the data using PCA and UMAP/t-SNE.
        This helps visualize the data and identify cell populations.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.adata.copy()
        
        # Check if HVGs are available
        if 'highly_variable' not in adata.var.columns:
            st.warning("Please perform feature selection first!")
        else:
            # PCA parameters
            st.subheader("Principal Component Analysis (PCA)")

            n_pcs = st.number_input("Number of principal components (default: 50)", 10, 100, 50)
            use_hvg = st.checkbox("Use only highly variable genes", value=True)
            zero_center = st.checkbox("Zero center", value=True)
                
            if st.button("Run PCA"):
                with st.spinner("Running PCA..."):
                    # Keep only HVGs for PCA
                    if use_hvg:
                        adata_pca = adata[:, adata.var.highly_variable]
                    else:
                        adata_pca = adata
                    
                    # Run PCA
                    sc.tl.pca(adata, n_comps=n_pcs, zero_center=zero_center, svd_solver='arpack', use_highly_variable=use_hvg)
                    
                    # Store results
                    st.session_state.adata = adata
                    
                    st.success("✅ PCA complete!")
                    
            # plotting pca
            if 'X_pca' in adata.obsm:
                col1, col2 = st.columns(2)
                # Plot PCA variance ratio
                with col1:
                    st.subheader("PCA Variance Ratio")
                    sc.pl.pca_variance_ratio(adata, log=False, n_pcs=20, show=False)
                    fig1 = plt.gcf()
                    st.pyplot(fig1)
                    plt.clf()

                # Plot PCA
                with col2:
                    st.subheader("PCA Scatter Plot")
                    sc.pl.pca(adata, show=False)
                    fig2 = plt.gcf()
                    st.pyplot(fig2)
                    plt.clf()
            
        # UMAP/t-SNE parameters
        if 'X_pca' in adata.obsm:
            st.subheader("Non-linear Dimensionality Reduction")
            
            embedding_method = st.selectbox("Choose method:", ["UMAP", "t-SNE"])
            
            if embedding_method == "UMAP":
                col1, col2 = st.columns(2)
                with col1:
                    n_neighbors = st.slider("Number of neighbors", 5, 100, 15)
                    min_dist = st.slider("Minimum distance", 0.0, 1.0, 0.5)
                with col2:
                    spread = st.slider("Spread", 0.5, 3.0, 1.0)
                    
                if st.button("Run UMAP"):
                    with st.spinner("Running UMAP..."):
                        # Compute neighborhood graph
                        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
                        # Run UMAP
                        sc.tl.umap(adata, min_dist=min_dist, spread=spread)
                        
                        # Store results
                        st.session_state.adata = adata
                        
                        st.success("✅ UMAP complete!")
                        
                        # Plot UMAP
                        col = st.columns(1)[0]
                        with col:
                            fig, ax = plt.subplots(figsize=(8,6))
                            sc.pl.umap(adata, ax=ax)
                            st.pyplot(fig)
            
            elif embedding_method == "t-SNE":
                col1, col2 = st.columns(2)
                with col1:
                    perplexity = st.slider("Perplexity", 5, 100, 30)
                    early_exaggeration = st.slider("Early exaggeration", 5, 50, 12)
                with col2:
                    learning_rate = st.slider("Learning rate", 10, 1000, 200)
                    
                if st.button("Run t-SNE"):
                    with st.spinner("Running t-SNE..."):
                        # Run t-SNE
                        sc.tl.tsne(adata, perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate)
                        
                        # Store results
                        st.session_state.adata = adata
                        
                        st.success("✅ t-SNE complete!")
                        
                        # Plot t-SNE
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sc.pl.tsne(adata, ax=ax)
                        st.pyplot(fig)

            # show updated .obs and .var
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

elif mode == "Standard Pipeline" and selected_step == "Clustering":
    if st.session_state.adata is None:
        st.warning("Please load data first!")
    else:
        st.markdown('<h2 class="step-header">Step 7: Clustering</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Grouping cells into clusters based on their gene expression similarity.
        This helps identify distinct cell types or states.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.adata.copy()
        
        # Check if neighborhood graph exists
        if 'neighbors' not in adata.uns:
            st.warning("Please run PCA and compute neighborhood graph first!")
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
                    
                    n_clusters = len(adata.obs[cluster_key].unique())
                    st.success(f"✅ {clustering_method} clustering complete! Identified {n_clusters} clusters.")
                    
                    # Plot clustering results
                    if 'X_umap' in adata.obsm:
                        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # UMAP colored by clusters
                        sc.pl.umap(adata, color=cluster_key, ax=axes[0], legend_loc='right', frameon=False)
                        axes[0].set_title(f'{clustering_method} Clusters')
                        
                        # UMAP colored by total counts
                        sc.pl.umap(adata, color='sample', ax=axes[1], legend_loc='right', frameon=False)
                        axes[1].set_title('Total Counts')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    elif 'X_tsne' in adata.obsm:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sc.pl.tsne(adata, color=cluster_key, ax=ax)
                        st.pyplot(fig)
                    
                    col1, col2 = st.columns(2)

                    clustering_keys = [col for col in adata.obs.columns if col.startswith("leiden_") or col.startswith("louvain_")]
                    if not clustering_keys:
                        st.warning("No clustering found. Please run clustering first.")
                    else:
                        cluster_key = clustering_keys[0]  # or let user select with st.selectbox()

                        with col1:
                            # Cluster composition
                            st.subheader("Cell counts per cluster")
                            cluster_counts = adata.obs[cluster_key].value_counts().sort_index()
                            cluster_df = cluster_counts.reset_index()
                            cluster_df.columns = ['Cluster', 'Cell count']

                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.barplot(x=cluster_counts.index.astype(str), y=cluster_counts.values, ax=ax, color="steelblue")
                            ax.set_xlabel("Cluster")
                            ax.set_ylabel("Cell count")
                            ax.tick_params(left=False, bottom=False)
                            for spine in ax.spines.values():
                                spine.set_visible(False)
                            plt.xticks(rotation=0)
                            st.pyplot(fig)

                            # Cluster composition colored by sample
                            grouped_count = (
                                adata.obs[[cluster_key, 'sample']]
                                .value_counts()
                                .reset_index(name='count')
                            )
                            pivot_df = grouped_count.pivot(index=cluster_key, columns='sample', values='count').fillna(0)

                        # Plot stacked bar chart
                        with col2:
                            st.subheader("Cell counts per cluster by samples")

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
                        
                # show updated .obs and .var
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
        st.markdown('<h2 class="step-header">Step 8: Differential Gene Expression </h2>', unsafe_allow_html=True)

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
        st.markdown('<h2 class="step-header">Step 8: Visualization</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Creating various visualizations to explore your data.
        This includes gene expression plots, cluster comparisons, and quality metrics.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.adata.copy()
        
        # Visualization options
        viz_type = st.selectbox(
            "Choose visualization type:",
            ["Gene Expression", "Cluster Analysis", "Quality Metrics"]
        )
        
        if viz_type == "Gene Expression":
            st.subheader("Gene Expression Visualization")
            
            # Gene selection
            gene_input = st.text_input("Enter gene names (e.g. Ebf2, Sox9, Gata6):", "")
            
            if gene_input:
                genes = [gene.strip() for gene in gene_input.split(',')]
                available_genes = [gene for gene in genes if gene in adata.var_names]
                unavailable_genes = [gene for gene in genes if gene not in adata.var_names]

                if unavailable_genes:
                    st.warning(f"The following genes were not found and will be ignored: {', '.join(unavailable_genes)}")

                if available_genes:
                    plot_type = st.selectbox('Plot type:', ['Scatter plot', 'Violin plot', 'Dot plot', 'Heatmap', 'Matrix plot', 'Stacked violin plot'])

                    if plot_type == 'Scatter plot':
                        projection_type = st.selectbox("Projection type:", ["UMAP", "t-SNE", "PCA"])

                        if st.button("Plot Gene Expression"):
                            n_genes = len(available_genes)
                            n_cols = min(3, n_genes)
                            n_rows = (n_genes + n_cols - 1) // n_cols
                            
                            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                            axes = axes.flatten() if n_genes > 1 else [axes]
                            
                            for i, gene in enumerate(available_genes):
                                ax = axes[i]
                                if projection_type == "UMAP" and "X_umap" in adata.obsm:
                                    sc.pl.umap(adata, color=gene, ax=ax, ncols=3, frameon=False, show=False)
                                elif projection_type == "t-SNE" and "X_tsne" in adata.obsm:
                                    sc.pl.tsne(adata, color=gene, ax=ax, ncols=3, frameon=False, show=False)
                                elif projection_type == "PCA" and "X_pca" in adata.obsm:
                                    sc.pl.pca(adata, color=gene, ax=ax, ncols=3, frameon=False, show=False)

                                ax.set_title(gene)
                            
                            for j in range(i + 1, len(axes)):
                                fig.delaxes(axes[j])  # remove empty subplots

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
        st.markdown('<h2 class="step-header">Step 9: Export Results </h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>What we're doing:</strong> Export results as a .h5ad file that can be reuploaded into scViewer for other analysis and visualizations. Click reset analysis to redo the analysis.
        </div>
        """, unsafe_allow_html=True)
        
        adata = st.session_state.adata.copy()

        # Write to a temporary file on disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad") as tmp:
            adata.write_h5ad(tmp.name)
            tmp_path = tmp.name

        # Read it back into memory as bytes
        with open(tmp_path, "rb") as f:
            h5ad_bytes = f.read()

        # Clean up temporary file
        os.remove(tmp_path)

        # download button
        st.download_button(
            label="Download .h5ad",
            data=h5ad_bytes,
            file_name="exported_results.h5ad",
            mime="application/octet-stream"
        )

        # reset button
        if st.button("Reset Analysis"):
            st.session_state.adata = None

            # clear other session state variables
            keys_to_reset = [key for key in st.session_state.keys() if key != "selected_step"]
            for key in keys_to_reset:
                del st.session_state[key]

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
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your single-cell data file",
        type=['h5ad']
    )
    
    if uploaded_file is not None:
        try:
            # load h5ad file
            uploaded_file.name.endswith('.h5ad')
            adata = sc.read_h5ad(uploaded_file)

            # Patch: Ensure adata.X is sparse if needed
            if isinstance(adata.X, np.ndarray):
                adata.X = scipy.sparse.csr_matrix(adata.X)

            # Optional: patch raw
            if adata.raw is not None and isinstance(adata.raw.X, np.ndarray):
                adata.raw._X = scipy.sparse.csr_matrix(adata.raw.X)

            # Store in session state
            st.session_state.adata = adata
            
            st.success(f"Successfully loaded {uploaded_file.name}")
            
            # Display data summary
            st.subheader("Dataset Overview")
            display_data_summary_advanced(adata)
            
            # Show first few genes and cells
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("First 10 Observations (.obs)")
                df = adata.obs.head(10).copy()
                for col in df.select_dtypes(include=bool).columns:
                    df[col] = df[col].astype(str)
                st.dataframe(df)

            with col2:
                st.subheader("First 10 Observations (.var)")
                df = adata.var.head(10).copy()
                for col in df.select_dtypes(include=bool).columns:
                    df[col] = df[col].astype(str)
                st.dataframe(df)
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

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

            genes = adata.var_names.tolist()

            gene_input = st.multiselect('Select gene names:', genes)

            if gene_input:
                # genes = [gene.strip() for gene in gene_input.split(',')]
                # available_genes = [gene for gene in genes if gene in adata.var_names]
                # unavailable_genes = [gene for gene in genes if gene not in adata.var_names]

                # if unavailable_genes:
                #     st.warning(f"The following genes were not found and will be ignored: {', '.join(unavailable_genes)}")

                # if available_genes:
                #     st.markdown("### Expression Filters for Selected Genes")
                #     gene_filters = {}
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
                    import operator as op

                    ops = {
                        '>': op.gt,
                        '>=': op.ge,
                        '<': op.lt,
                        '<=': op.le,
                        '=': op.eq,
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

            # Get clustering keys (e.g., 'leiden', 'louvain')
            clustering_cols = [col for col in adata.obs.columns if col.startswith("leiden") or col.startswith("louvain")]
            clustering_key = st.selectbox('Choose clustering key:', clustering_cols)

            # Get unique cluster labels
            clusters = adata.obs[clustering_key].unique().tolist()
            selected_clusters = st.multiselect('Choose clusters to re-analyze:', sorted(clusters))

            st.subheader('Highly variable genes')
            calculate_hvg = st.checkbox('Calculate highly variable genes', value=True)
            use_hvg = st.checkbox("Use only highly variable genes", value=True)
            
            st.subheader('PCA')
            n_pcs = st.slider('Number of PCs', 10, 100, 50)
            zero_center = st.checkbox("Zero center PCA", value=True)

            st.subheader('Non-linear Dimensionality Reduction')
            embedding_method = st.selectbox("Choose embedding method:", ["UMAP", "t-SNE"])

            if embedding_method == 'UMAP':
                col1, col2 = st.columns(2)
                with col1:
                    n_neighbors = st.slider("Number of neighbors", 5, 100, 15)
                    min_dist = st.slider("Minimum distance", 0.0, 1.0, 0.5)
                with col2:
                    spread = st.slider("Spread", 0.5, 3.0, 1.0)
            elif embedding_method == 't-SNE':
                col1, col2 = st.columns(2)
                with col1:
                    perplexity = st.slider("Perplexity", 5, 100, 30)
                    early_exaggeration = st.slider("Early exaggeration", 5, 50, 12)
                with col2:
                    learning_rate = st.slider("Learning rate", 10, 1000, 200)


            if st.button('Re-analyze'):
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
                    # Calculate HVGs
                    if calculate_hvg:
                        sc.pp.highly_variable_genes(subset_adata, flavor='seurat', n_top_genes=2000)
                        if use_hvg:
                            subset_adata = subset_adata[:, subset_adata.var.highly_variable]

                    # Scale and PCA
                    sc.pp.scale(subset_adata)
                    sc.tl.pca(subset_adata, n_comps=n_pcs, zero_center=zero_center, svd_solver='arpack')

                    # umap and t-sne
                    if embedding_method == "UMAP" and 'X_pca' in subset_adata.obsm:
                        sc.pp.neighbors(subset_adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
                        sc.tl.umap(subset_adata, min_dist=min_dist, spread=spread)
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

                    # Plot PCA results from subset
                    if 'X_pca' in subset_adata.obsm:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("PCA Scatter Plot (Subset)")
                            sc.pl.pca(subset_adata, show=False)
                            st.pyplot(plt.gcf())
                            plt.clf()
                        
                        with col2:
                            st.subheader(f"{embedding_method} Plot (Subset)")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            if embedding_method == 'UMAP':
                                sc.pl.umap(subset_adata, ax=ax, show=False, frameon=False)
                            else:
                                sc.pl.tsne(subset_adata, ax=ax, show=False, frameon=False)
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

        # subset and filter based on columns in .var
        # if task == 'Subset and filter':
        #     var_col = adata.var.columns.tolist()
        #     subset = st.selectbox('Subset by:', var_col)

        #     selected_values = None
        #     subset_series = adata.var[subset]
        #     col_type = subset_series.dtype
        #     st.write(f"Selected column `{subset}` dtype: {col_type}")

        #     if pd.api.types.is_bool_dtype(col_type) or pd.api.types.is_categorical_dtype(col_type) or not pd.api.types.is_numeric_dtype(col_type):
        #         unique_values = subset_series.unique().tolist()
        #         selected_values = st.multiselect(f'Select  `{subset}` values to keep:', unique_values)

        #     elif pd.api.types.is_numeric_dtype(col_type):
        #         col1, col2, col3 = st.columns([1,1,2])
        #         with col1:
        #             operator = st.selectbox('Operator:', ['=', '>', '<', '<=', '>='])
                
        #         with col2:
        #             try:
        #                 default_val = float(np.nanmedian(subset_series))
        #             except Exception:
        #                 default_val = 0.0
        #             threshold = st.number_input('Value:', value=default_val)
                
        #         with col3:
        #             st.write(f"Range: {np.nanmin(subset_series):.2f} to {np.nanmax(subset_series):.2f}")

        #     # apply filter
        #     if st.button('Subset'):
        #         subset_adata = None

        #         if pd.api.types.is_numeric_dtype(col_type):
        #             try:
        #                 if operator == '=':
        #                     mask = subset_series == threshold
        #                 elif operator == '>':
        #                     mask = subset_series > threshold
        #                 elif operator == '<':
        #                     mask = subset_series < threshold
        #                 elif operator == '>=':
        #                     mask = subset_series >= threshold
        #                 elif operator == '<=':
        #                     mask = subset_series <= threshold
        #                 else:
        #                     st.warning("Invalid operator selected.")
        #                     mask = None

        #                 if mask is not None:
        #                     subset_adata = adata[:, mask].copy()
        #             except NameError:
        #                 st.warning("Numeric filter not fully defined. Please select operator and value.")

        #         elif selected_values:
        #             mask = subset_series.isin(selected_values)
        #             subset_adata = adata[:, mask].copy()

        #         else:
        #             st.warning("No subsetting applied. Please check your selections.")

        #         if subset_adata is not None:
        #             st.success(f'Subset applied. {subset_adata.n_obs} cells retained. {subset_adata.n_vars} genes retained.')

        #             # show data preview
        #             df = subset_adata.var[[subset]].copy()
        #             for col in df.select_dtypes(include=bool).columns:
        #                 df[col] = df[col].astype(str)
        #             st.dataframe(df.fhead(20))

        #             # name the subset + save
        #             st.markdown('### Save this subset')
        #             if 'saved_subsets' not in st.session_state:
        #                 st.session_state['saved_subsets'] = {}
        #             subset_name = st.text_input('Name this subset:', value=f'{subset}_filtered')

        #             if st.button('Save subset'):
        #                 if subset_name.strip() != '':
        #                     st.session_state['saved_subsets'][subset_name] = subset_adata
        #                     st.success(f'Subset saved as `{subset_name}`.')
        #                 else:
        #                     st.warning('Please enter a valid name.')
        #             else:
        #                 st.warning("No subsetting applied. Please check your selections.")

        if task == 'Differential gene expression analysis':
            groupby = st.selectbox("Group by (`.obs` column):", adata.obs.select_dtypes(['category', 'object']).columns)

            group_vals = adata.obs[groupby].cat.categories.tolist()

            reference_group = st.selectbox("Reference group:", ['Rest'] + group_vals)
            # if ref = 'rest', then groups = 'all'
            if reference_group == 'Rest':
                target_group = st.multiselect("Target group:", ['All'])
            else: 
                target_group = st.multiselect("Target group:", ['All'] + group_vals, default=['All'])
            
            n_genes = st.number_input('Select number of genes to show:', min_value=None, max_value=None, value=0)
            method = st.selectbox("Test method:", ["t-test", "wilcoxon", "logreg", "t-test_overestim_var"])
            comparison_name = st.text_input("Name this comparison:", value=f'')

            if st.button("Run Differential Expression"):
                with st.spinner('Identifying DEGs...'):
                    #Ensure groupby column is categorical
                    adata.obs[groupby] = adata.obs[groupby].astype('category')
                    group_values = adata.obs[groupby].cat.categories.tolist()

                    # Show available group names (debugging help)
                    # st.write("Available groups:", group_values)

                    # Validate reference group
                    if reference_group != 'Rest' and reference_group not in group_values:
                        st.error(f"Reference group `{reference_group}` not found in `{groupby}` column.")
                    elif target_group != 'All' and target_group not in group_values:
                        st.error(f"Target group `{target_group}` not found in `{groupby}` column.")
                    else:
                        if reference_group == 'Rest':
                            sc.tl.rank_genes_groups(
                                adata,
                                groupby = groupby,
                                use_raw = False,
                                reference = 'rest',
                                method = method,
                                key_added = comparison_name
                            )
                        elif reference_group != 'Rest' and target_group != 'All':
                            sc.tl.rank_genes_groups(
                                adata, 
                                groupby = groupby,
                                use_raw = False,
                                reference = reference_group,
                                groups = [target_group],
                                method = method,
                                key_added = comparison_name
                            )
                        elif reference_group != 'Rest' and target_group == 'All':
                            sc.tl.rank_genes_groups(
                                adata, 
                                groupby = groupby,
                                use_raw = False,
                                reference = reference_group,
                                groups = 'all',
                                method = method,
                                key_added = comparison_name
                            )

                        st.session_state.adata = adata
                        st.success(f"Differential expression run complete!")
            
                # show top n_gene results as plot
                st.subheader(f'Showing top {n_genes} DEGs from {comparison_name}')
                axes_list = sc.pl.rank_genes_groups(
                    adata,
                    n_genes = n_genes,
                    sharey = False,
                    key = comparison_name,
                    show = False,
                    return_fig = True
                )
                # convert to a Figure
                if isinstance(axes_list, list):
                    fig = axes_list[0].get_figure()  # use the first Axes
                else:
                    fig = axes_list.get_figure()
                st.pyplot(fig)

                # show degs in table
                results = adata.uns[comparison_name]
                groups = results['names'].dtype.names # cluster names
                # showing top genes
                top_genes = []
                for group in groups:
                    names = results['names'][group]
                    scores = results['scores'][group]
                    pvals = results['pvals_adj'][group] if 'pvals_adj' in results else results['pvals'][group]

                    for rank, (gene, score, pval) in enumerate(zip(names, scores, pvals), start=1):
                        top_genes.append({
                            'Cluster': group,
                            'Rank': rank,
                            'Gene': gene,
                            'Score': score,
                            'Adjusted p-value': pval
                        })
                # convert to df
                deg_df = pd.DataFrame(top_genes)
                # convert to scientific notation
                deg_df['Adjusted p-value'] = deg_df['Adjusted p-value'].apply(lambda x: f'{x:.2e}')

                st.subheader('DEGs per cluster (table)')
                st.dataframe(deg_df)
                
        elif task == 'Co-expression analysis':
            gene_list = adata.var_names.tolist()

            analysis_type = st.radio("Analysis type", ["Pairwise correlation", "Gene scatter plot"])

            if analysis_type == "Pairwise correlation":
                selected_genes = st.multiselect("Select gene names:", gene_list)
                method = st.radio("Correlation method:", ["pearson", "spearman"])

                if st.button('Run co-expression analysis:'):
                    # if selected_genes:
                    #     genes = [gene.strip() for gene in selected_genes.split(',')]
                    #     available_genes = [gene for gene in genes if gene in adata.var_names]
                    #     unavailable_genes = [gene for gene in genes if gene not in adata.var_names]
                        
                    #     if unavailable_genes:
                    #         st.warning(f"The following genes were not found and will be ignored: {', '.join(unavailable_genes)}")

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
    if 'adata' not in st.session_state:
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

        # viz options
        viz_type = st.selectbox(
            "Choose visualization type:",
            ["Gene expression", "Cluster analysis", 'Cell group annotations', "Quality metrics"]
        )

        if viz_type == 'Gene expression':
            st.subheader('Gene expression visualization')

            # select genes
            gene_list = adata.var_names.tolist()
            gene_names = st.multiselect('Select gene names:', gene_list)

            if gene_names:
                plot_type = st.selectbox('Plot type:', ['Scatter plot', 'Violin plot', 'Dot plot', 'Heatmap', 'Matrix plot', 'Stacked violin plot'])

                if plot_type == 'Scatter plot':
                    projection_type = st.selectbox("Projection type:", ["UMAP", "t-SNE", "PCA"])

                    if st.button('Plot Gene Expression') and projection_type is not None:
                        n_genes = len(gene_names)
                        n_cols = min(3, n_genes)
                        n_rows = (n_genes + n_cols - 1) // n_cols
                        
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                        axes = axes.flatten() if n_genes > 1 else [axes]
                        
                        for i, gene in enumerate(gene_names):
                            ax = axes[i]
                            if projection_type == "UMAP" and "X_umap" in adata.obsm:
                                sc.pl.umap(adata, color=gene, ax=ax, ncols=3, frameon=False, show=False)
                            elif projection_type == "t-SNE" and "X_tsne" in adata.obsm:
                                sc.pl.tsne(adata, color=gene, ax=ax, ncols=3, frameon=False, show=False)
                            elif projection_type == "PCA" and "X_pca" in adata.obsm:
                                sc.pl.pca(adata, color=gene, ax=ax, ncols=3, frameon=False, show=False)

                            ax.set_title(gene)
                        
                        for j in range(i + 1, len(axes)):
                            fig.delaxes(axes[j])  # remove empty subplots

                        plt.tight_layout()
                        st.pyplot(fig)
                elif plot_type == 'Violin plot':
                        if st.button("Plot Gene Expression"):
                            n_genes = len(gene_names)
                            n_cols = min(3, n_genes)
                            n_rows = (n_genes + n_cols - 1) // n_cols
                            
                            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                            axes = axes.flatten() if n_genes > 1 else [axes]
                            
                            for i, gene in enumerate(gene_names):
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
                            n_genes = len(gene_names)
                            n_cols = min(3, n_genes)
                            n_rows = (n_genes + n_cols - 1) // n_cols
                            
                            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                            axes = axes.flatten() if n_genes > 1 else [axes]
                            
                            for i, gene in enumerate(gene_names):
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
                        n_genes = len(gene_names)
                        width = 8  # Fixed width
                        height = max(4, n_genes * 0.5 + 2)

                        sc.pl.heatmap(adata, var_names=gene_names, groupby=group, dendrogram=True, show=False, figsize=(width, height), swap_axes=True)
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
                        n_genes = len(gene_names)
                        width = 8  # Fixed width
                        height = max(4, n_genes * 0.5 + 2)

                        sc.pl.matrixplot(adata, var_names=gene_names, groupby=group, dendrogram=True, show=False, figsize=(width, height), swap_axes=True)
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
                        n_genes = len(gene_names)
                        width = 8  # Fixed width
                        height = max(4, n_genes * 0.5 + 2)

                        sc.pl.stacked_violin(adata, var_names=gene_names, groupby=group, dendrogram=True, show=False, figsize=(width, height), swap_axes=True)
                        fig = plt.gcf()  # Get current figure
                        st.pyplot(fig)
                        plt.close(fig)

                else:
                    st.warning("No valid gene names entered.")
            
        elif viz_type == "Cluster analysis":
            st.subheader('Visualize clusters')

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
        
        elif viz_type == 'Cell group annotations':
            st.subheader("Visualize cell groups")

            # Get all boolean columns from adata.obs
            bool_cols = [col for col in adata.obs.columns if pd.api.types.is_bool_dtype(adata.obs[col]) or pd.api.types.is_categorical_dtype(adata.obs[col])]

            if not bool_cols:
                st.warning("No boolean annotations found in `adata.obs`. Please annotate cells first.")
            else:
                annotation_key = st.selectbox("Select annotation to visualize:", bool_cols)

                # Get available projections from adata.obsm
                projection_options = [proj.replace("X_", "") for proj in adata.obsm.keys() if proj.startswith("X_")]
                if not projection_options:
                    st.warning("No dimensionality reduction projections found (e.g., UMAP, tSNE in `adata.obsm`).")
                else:
                    projection_type = st.selectbox("Select projection type:", projection_options)

                    if st.button("Plot Annotated Cells"):
                        fig, ax = plt.subplots(figsize=(8, 6))

                        if projection_type.lower() == "umap":
                            sc.pl.umap(adata, color=annotation_key, ax=ax, frameon=False, show=False)
                        elif projection_type.lower() == "tsne":
                            sc.pl.tsne(adata, color=annotation_key, ax=ax, frameon=False, show=False)
                        else:
                            # Fallback to generic embedding
                            embedding_key = f"X_{projection_type}"
                            if embedding_key in adata.obsm:
                                sc.pl.embedding(adata, basis=embedding_key.replace("X_", ""), color=annotation_key, ax=ax, frameon=False, show=False)
                            else:
                                st.error(f"Projection '{projection_type}' not found in `adata.obsm`.")

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
        selected_keys = st.multiselect("Select AnnData objects to export:", list(all_adata_dict.keys()), default=list(all_adata_dict.keys()))

        if selected_keys:
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                for key in selected_keys:
                    adata = all_adata_dict[key]

                    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
                        adata.write(tmp_file.name)
                        with open(tmp_file.name, "rb") as f:
                            zipf.writestr(f"{key}.h5ad", f.read())

            zip_buffer.seek(0)

            st.download_button(
                label="Download selected .h5ad files as ZIP",
                data=zip_buffer,
                file_name="exported_adata_files.zip",
                mime="application/zip"
            )
        else:
            st.info("Please select at least one object to export.")







            
        


                



            

        
                    
                









                



































