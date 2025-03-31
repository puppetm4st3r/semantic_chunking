import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from typing import List, Dict
import datamapplot
import seaborn as sns
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class SemanticGroupUtils:
    """
    A set of tools to manipulate and visualize semantic text splits.
    """

    @staticmethod
    def plot_chunk_differences_and_breakpoints(
        split_texts: List[str],
        break_points: np.ndarray,
        chunk_cosine_distances: np.ndarray,
    ) -> None:
        """
        Plots the differences and breakpoints of text chunks based on cosine distances.

        Args:
            split_texts (List[str]): A list of text splits.
            break_points (np.ndarray): An array of break point indices.
            chunk_cosine_distances (np.ndarray): An array of cosine distances between text splits.

        Returns:
            None
        """
        cumulative_len = np.cumsum([len(x) for x in split_texts])
        
        # Create interactive Plotly figure
        fig = make_subplots(rows=1, cols=1)
        
        # Add line trace for cosine distances
        fig.add_trace(
            go.Scatter(
                x=cumulative_len[:-1],
                y=chunk_cosine_distances,
                mode='lines',
                name='Cosine Distance',
                line=dict(color='blue')
            )
        )
        
        # Add markers for each point
        fig.add_trace(
            go.Scatter(
                x=cumulative_len[:-1],
                y=chunk_cosine_distances,
                mode='markers',
                marker=dict(color='red', symbol='x', size=8, opacity=0.7),
                name='Data Points'
            )
        )
        
        # Add vertical lines for breakpoints
        for bp in break_points:
            fig.add_shape(
                type="line",
                x0=cumulative_len[bp],
                y0=0,
                x1=cumulative_len[bp],
                y1=1.1 * max(chunk_cosine_distances),
                line=dict(color="black", width=1.5, dash="dash"),
            )
        
        # Calculate a reasonable initial zoom level (showing about 10% of the data)
        total_range = cumulative_len[-1]
        initial_range_start = 0
        initial_range_end = total_range * 0.1  # Show first 10% of data initially
        
        # Update layout
        fig.update_layout(
            title='Agrupación de fragmentos similares',
            xaxis_title='Posición en el corpus de texto',
            yaxis_title='Distancia coseno entre fragmentos',
            height=600,
            width=1400,
            showlegend=True,
            dragmode='pan',
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='linear',
                range=[initial_range_start, initial_range_end]
            )
        )
        
        # Show figure
        fig.show()

    @staticmethod
    def plot_2d_semantic_embeddings(
        semantic_embeddings: List[np.ndarray], semantic_text_groups: List[str], umap_neighbors: int = 5
    ) -> None:
        """
        Creates a 3D plot of semantic embeddings using Plotly.

        Args:
            semantic_embeddings (List[np.ndarray]): Embeddings of text chunks.
            semantic_text_groups (List[str]): A list of text chunks.
            umap_neighbors (int, optional): Number of neighbors for UMAP. Defaults to 5.

        Returns:
            None
        """
        dimension_reducer = UMAP(
            n_neighbors=umap_neighbors, n_components=3, min_dist=0.0, metric="cosine", random_state=0
        )
        reduced_embeddings = dimension_reducer.fit_transform(semantic_embeddings)

        # Create DataFrame with renamed columns
        splits_df = pd.DataFrame(
            {
                "umap_reduction_x": reduced_embeddings[:, 0],
                "umap_reduction_y": reduced_embeddings[:, 1],
                "umap_reduction_z": reduced_embeddings[:, 2],
                "idx": np.arange(len(reduced_embeddings[:, 0])),
            }
        )

        splits_df["chunk_end"] = np.cumsum([len(x) for x in semantic_text_groups])

        # Create Plotly figure
        fig = go.Figure()
        
        # Add 3D scatter plot colored by index
        fig.add_trace(
            go.Scatter3d(
                x=splits_df["umap_reduction_x"],
                y=splits_df["umap_reduction_y"],
                z=splits_df["umap_reduction_z"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=splits_df["idx"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Index")
                ),
                text=[f"Index: {i}" for i in splits_df["idx"]],
                hoverinfo="text",
                name="Chunks"
            )
        )
        
        # Add 3D lines connecting points in sequence with opacity 0.6
        fig.add_trace(
            go.Scatter3d(
                x=reduced_embeddings[:, 0],
                y=reduced_embeddings[:, 1],
                z=reduced_embeddings[:, 2],
                mode="lines",
                line=dict(color="rgba(0,0,0,0.4)", width=1),
                opacity=0.6,
                showlegend=False
            )
        )
        
        # Calculate data ranges to optimize zoom sensitivity
        x_range = splits_df["umap_reduction_x"].max() - splits_df["umap_reduction_x"].min()
        y_range = splits_df["umap_reduction_y"].max() - splits_df["umap_reduction_y"].min()
        z_range = splits_df["umap_reduction_z"].max() - splits_df["umap_reduction_z"].min()
        
        # Update layout
        fig.update_layout(
            title="3D Semantic Embeddings",
            scene=dict(
                xaxis_title="UMAP Reduction X",
                yaxis_title="UMAP Reduction Y",
                zaxis_title="UMAP Reduction Z",
                aspectmode="cube",
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.2),
                    projection=dict(type="orthographic")
                ),
                aspectratio=dict(x=1, y=1, z=1),
                # Set specific axis ranges based on data
                xaxis=dict(range=[splits_df["umap_reduction_x"].min() - x_range*0.1, 
                                  splits_df["umap_reduction_x"].max() + x_range*0.1]),
                yaxis=dict(range=[splits_df["umap_reduction_y"].min() - y_range*0.1, 
                                  splits_df["umap_reduction_y"].max() + y_range*0.1]),
                zaxis=dict(range=[splits_df["umap_reduction_z"].min() - z_range*0.1, 
                                  splits_df["umap_reduction_z"].max() + z_range*0.1]),
            ),
            height=800,
            width=1000,
            template="plotly_white"
        )
        
        # Configure enhanced zoom behavior
        fig.show(config=dict(
            scrollZoom=True,
            doubleClick="reset",
            displayModeBar=True,
            modeBarButtonsToAdd=["resetCameraLastSave3d"],
            # Increase scroll zoom sensitivity
            edits=dict(
                shapePosition=True,
                annotationPosition=True,
            )
        ))

    @staticmethod
    def plot_2d_semantic_embeddings_with_clusters(
        semantic_embeddings: List[np.ndarray],
        semantic_text_groups: List[str],
        linkage: np.ndarray,
        n_clusters: int = 30,
        umap_neighbors: int = 5
    ) -> pd.DataFrame:
        """
        Creates a 3D plot of reduced embeddings with cluster labels using Plotly.

        Args:
            semantic_embeddings (List[np.ndarray]): Embeddings of text chunks.
            semantic_text_groups (List[str]): A list of text chunks.
            linkage (np.ndarray): Linkage matrix for hierarchical clustering.
            n_clusters (int, optional): Number of clusters. Defaults to 30.
            umap_neighbors (int, optional): Number of neighbors for UMAP. Defaults to 5.

        Returns:
            pd.DataFrame: DataFrame containing reduced embeddings and cluster labels.
        """
        cluster_labels = hierarchy.cut_tree(linkage, n_clusters=n_clusters).ravel()
        dimension_reducer = UMAP(
            n_neighbors=umap_neighbors, n_components=3, min_dist=0.0, metric="cosine", random_state=0
        )
        reduced_embeddings = dimension_reducer.fit_transform(semantic_embeddings)

        splits_df = pd.DataFrame(
            {
                "umap_reduction_x": reduced_embeddings[:, 0],
                "umap_reduction_y": reduced_embeddings[:, 1],
                "umap_reduction_z": reduced_embeddings[:, 2],
                "cluster_label": cluster_labels,
            }
        )

        splits_df["chunk_end"] = np.cumsum([len(x) for x in semantic_text_groups])

        # Create Plotly figure
        fig = go.Figure()
        
        # Get unique clusters for color assignments
        unique_clusters = sorted(splits_df["cluster_label"].unique())
        
        # Create a fixed discrete color palette with 16 distinctive colors
        discrete_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # Blue, Orange, Green, Red, Purple
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',  # Brown, Pink, Gray, Olive, Cyan
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',  # Light Blue, Light Orange, Light Green, Light Red, Light Purple
            '#c49c94'  # Light Brown (16th color)
        ]
        
        # Create a color mapping for each cluster
        color_mapping = {cluster_id: discrete_colors[i % len(discrete_colors)] 
                         for i, cluster_id in enumerate(unique_clusters)}
        
        # Map colors to each data point based on cluster
        point_colors = [color_mapping[cluster] for cluster in splits_df["cluster_label"]]
        
        # Add 3D scatter plot with discrete colors - hide from legend
        fig.add_trace(
            go.Scatter3d(
                x=splits_df["umap_reduction_x"],
                y=splits_df["umap_reduction_y"],
                z=splits_df["umap_reduction_z"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=point_colors,
                ),
                text=[f"Cluster: {label}" for label in splits_df["cluster_label"]],
                hoverinfo="text",
                name="Clusters",
                showlegend=False  # Hide main trace from legend
            )
        )
        
        # Add 3D lines connecting points in sequence with opacity 0.6
        fig.add_trace(
            go.Scatter3d(
                x=reduced_embeddings[:, 0],
                y=reduced_embeddings[:, 1],
                z=reduced_embeddings[:, 2],
                mode="lines",
                line=dict(color="rgba(0,0,0,0.4)", width=1),
                opacity=0.6,
                showlegend=False
            )
        )
        
        # Add a trace for each cluster in the legend
        # Create a mapping to track which clusters we've already added to the legend
        cluster_legend_entries = {}
        
        for i, cluster_id in enumerate(unique_clusters):
            # Only create entry if not already added
            if cluster_id not in cluster_legend_entries:
                fig.add_trace(
                    go.Scatter3d(
                        x=[None], y=[None], z=[None],
                        mode="markers",
                        marker=dict(size=10, color=color_mapping[cluster_id]),
                        name=f"Cluster {cluster_id}",
                        legendgroup=f"cluster_{cluster_id}",  # Group by cluster
                        showlegend=True
                    )
                )
                cluster_legend_entries[cluster_id] = True
        
        # Calculate data ranges to optimize zoom sensitivity
        x_range = splits_df["umap_reduction_x"].max() - splits_df["umap_reduction_x"].min()
        y_range = splits_df["umap_reduction_y"].max() - splits_df["umap_reduction_y"].min()
        z_range = splits_df["umap_reduction_z"].max() - splits_df["umap_reduction_z"].min()
        
        # Update layout
        fig.update_layout(
            title="3D Semantic Embeddings with Clusters",
            scene=dict(
                xaxis_title="UMAP Reduction X",
                yaxis_title="UMAP Reduction Y",
                zaxis_title="UMAP Reduction Z",
                aspectmode="cube",
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.2),
                    projection=dict(type="orthographic")
                ),
                aspectratio=dict(x=1, y=1, z=1),
                # Set specific axis ranges based on data
                xaxis=dict(range=[splits_df["umap_reduction_x"].min() - x_range*0.1, 
                                  splits_df["umap_reduction_x"].max() + x_range*0.1]),
                yaxis=dict(range=[splits_df["umap_reduction_y"].min() - y_range*0.1, 
                                  splits_df["umap_reduction_y"].max() + y_range*0.1]),
                zaxis=dict(range=[splits_df["umap_reduction_z"].min() - z_range*0.1, 
                                  splits_df["umap_reduction_z"].max() + z_range*0.1]),
            ),
            height=800,
            width=1000,
            template="plotly_white",
            legend=dict(
                itemsizing="constant",
                font=dict(size=10),
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                traceorder="grouped"  # Group legend items
            )
        )
        
        # Configure enhanced zoom behavior
        fig.show(config=dict(
            scrollZoom=True,
            doubleClick="reset",
            displayModeBar=True,
            modeBarButtonsToAdd=["resetCameraLastSave3d"],
            # Increase scroll zoom sensitivity
            edits=dict(
                shapePosition=True,
                annotationPosition=True,
            )
        ))

        return splits_df

    @staticmethod
    def create_hierarchical_clustering(
        semantic_group_embeddings: List[np.ndarray],
        n_components_reduced: int = 4,
        plot: bool = True,
        umap_neighbors: int = 5
    ) -> np.ndarray:
        """
        Creates hierarchical clustering from semantic group embeddings.

        Args:
            semantic_group_embeddings (List[np.ndarray]): Embeddings of semantic groups.
            n_components_reduced (int, optional): Number of components for dimensionality reduction. Defaults to 10.
            plot (bool, optional): Whether to plot the clustering. Defaults to True.

        Returns:
            np.ndarray: Linkage matrix for hierarchical clustering.
        """
        dimension_reducer_clustering = UMAP(
            n_neighbors=umap_neighbors,
            n_components=n_components_reduced,
            min_dist=0.0,
            metric="cosine",
            random_state=0
        )
        reduced_embeddings_clustering = dimension_reducer_clustering.fit_transform(
            semantic_group_embeddings
        )

        row_linkage = hierarchy.linkage(
            pdist(reduced_embeddings_clustering),
            method="average",
            optimal_ordering=True,
        )

        if plot:
            g = sns.clustermap(
                pd.DataFrame(reduced_embeddings_clustering),
                row_linkage=row_linkage,
                row_cluster=True,
                col_cluster=False,
                annot=True,
                linewidth=0.5,
                annot_kws={"size": 8, "color": "white"},
                cbar_pos=None,
                dendrogram_ratio=0.25
            )

            g.ax_heatmap.set_yticklabels(
                g.ax_heatmap.get_yticklabels(), rotation=0, size=8
            )

        return row_linkage

    @staticmethod
    def plot_chunks_and_summaries(
        semantic_group_embeddings: List[np.ndarray],
        semantic_group_descriptions: List[str],
        umap_neighbors: int = 5
    ) -> None:
        """
        Plots the reduced embeddings and their descriptions.

        Args:
            semantic_group_embeddings (List[np.ndarray]): Embeddings of semantic groups.
            semantic_group_descriptions (List[str]): Descriptions of semantic groups.

        Returns:
            None
        """
        dimension_reducer = UMAP(
            n_neighbors=umap_neighbors, n_components=2, min_dist=0.0, metric="cosine", random_state=0
        )
        reduced_embeddings = dimension_reducer.fit_transform(semantic_group_embeddings)

        fig, ax = datamapplot.create_plot(
            reduced_embeddings,
            labels=semantic_group_descriptions,
            force_matplotlib=True,
            label_wrap_width=10,
            font_family="Urbanist",
            color_label_text=False,
            add_glow=False,
            figsize=(12, 8),
            label_font_size=12,
        )

    @staticmethod
    def plot_corpus_and_clusters(
        splits_df: pd.DataFrame, cluster_summaries: Dict[int, Dict[str, str]] = {}
    ) -> pd.DataFrame:
        """
        Plots the progression of a corpus with cluster labels using a swimlane chart visualization.
        Each cluster gets its own lane, making it easier to see where each topic appears in the text.

        Args:
            splits_df (pd.DataFrame): A DataFrame containing the corpus data with a 'cluster_label' column.
                                    It should also include a 'chunk_end' column indicating the end of each chunk.
            cluster_summaries (Dict[int, Dict[str, str]], optional): A dictionary containing cluster summaries.
                                    The keys are cluster IDs and the values are dictionaries with a 'summary' key.

        Returns:
            pd.DataFrame: A DataFrame with 'cluster_label' and 'index_span' columns, where 'index_span' is a tuple
                            indicating the start and end indices of each cluster segment.
        """
        df = splits_df

        # Identify the start of a new segment
        df["shifted"] = df["cluster_label"].shift(1)
        df["is_new_segment"] = (df["cluster_label"] != df["shifted"]).astype(int)
        segment_groups = df["is_new_segment"].cumsum()

        # Group by cluster label and segment group
        result = df.groupby(["cluster_label", segment_groups]).apply(
            lambda g: (g.index.min() - 1, g.index.max())
        )

        result = result.reset_index()[["cluster_label", 0]].rename(
            columns={0: "index_span"}
        )
        
        # Get unique clusters and sort them
        unique_clusters = sorted(result["cluster_label"].unique())
        n_clusters = len(unique_clusters)
        
        # Colorbrewer qualitative palette for better distinction between clusters
        # Using Tab20 for up to 20 colors, more clusters will cycle through colors
        color_map = px.colors.qualitative.Dark24
        
        # Create plotly figure
        fig = go.Figure()
        
        # Get the full range of x values to set initial zoom
        x_values = []
        
        # Map cluster IDs to lane positions (y-axis)
        cluster_to_lane = {cluster_id: i for i, cluster_id in enumerate(unique_clusters)}
        
        # Create lanes for each cluster
        for i, row in result.iterrows():
            cluster_id = row["cluster_label"]
            span = row["index_span"]
            lane_position = cluster_to_lane[cluster_id]
            
            # Calculate xmin and xmax
            if span[0] == -1:
                xmin = 0
            else:
                xmin = splits_df.iloc[span[0]]["chunk_end"]
                
            xmax = splits_df.iloc[span[1]]["chunk_end"]
            x_values.extend([xmin, xmax])
            
            # Get cluster name/summary if available
            cluster_name = f"Cluster {cluster_id}"
            if cluster_summaries and cluster_id in cluster_summaries:
                if isinstance(cluster_summaries[cluster_id], dict) and "summary" in cluster_summaries[cluster_id]:
                    cluster_name = cluster_summaries[cluster_id]["summary"]
                else:
                    cluster_name = str(cluster_summaries[cluster_id])
                    
            # Truncate long cluster names
            if len(cluster_name) > 50:
                cluster_name = cluster_name[:47] + "..."
            
            # Add a colored rectangle for this segment in the appropriate lane
            fig.add_trace(
                go.Scatter(
                    x=[xmin, xmax, xmax, xmin, xmin],
                    y=[lane_position - 0.4, lane_position - 0.4, lane_position + 0.4, lane_position + 0.4, lane_position - 0.4],
                    fill="toself",
                    fillcolor=color_map[cluster_id % len(color_map)],
                    line=dict(color="black", width=1),
                    mode="lines",
                    name=cluster_name,
                    hoverinfo="text",
                    hovertext=f"{cluster_name}<br>Posición: {xmin} - {xmax}",
                    showlegend=False
                )
            )
        
        # Add cluster names as y-axis labels
        y_ticks = []
        y_tick_texts = []
        
        for cluster_id in unique_clusters:
            lane = cluster_to_lane[cluster_id]
            y_ticks.append(lane)
            # Only use cluster ID number for Y-axis
            y_tick_texts.append(str(cluster_id))
        
        # Add legend trace for each cluster
        for cluster_id in unique_clusters:
            cluster_name = f"Cluster {cluster_id}"
            if cluster_summaries and cluster_id in cluster_summaries:
                if isinstance(cluster_summaries[cluster_id], dict) and "summary" in cluster_summaries[cluster_id]:
                    cluster_name = cluster_summaries[cluster_id]["summary"]
                else:
                    cluster_name = str(cluster_summaries[cluster_id])
            
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color_map[cluster_id % len(color_map)]),
                    name=cluster_name,
                    showlegend=True
                )
            )
        
        # Set initial zoom range (show first 10% of data)
        if x_values:
            total_range = max(x_values)
            initial_range_start = 0
            initial_range_end = total_range * 0.1
        else:
            initial_range_start = 0
            initial_range_end = 100
            
        # Update layout
        fig.update_layout(
            title={
                'text': 'Agrupación de fragmentos similares por temas',
                'y': 1,  # Move title further down
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Progresión en el corpus de texto',
            yaxis_title='Clusters',
            height=max(600, n_clusters * 40 + 200),  # Adjust height based on number of clusters
            width=1400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.9,  # Move legend higher
                xanchor="right",
                x=1
            ),
            margin=dict(t=150),  # Increase top margin significantly
            dragmode='pan',
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='linear',
                range=[initial_range_start, initial_range_end]
            ),
            yaxis=dict(
                tickvals=y_ticks,
                ticktext=y_tick_texts,
                range=[-0.5, len(unique_clusters) - 0.5]
            )
        )
        
        # Print cluster summaries if provided
        if cluster_summaries:
            print("Resumen de clusters:")
            cluster_summaries_to_print = {}
            for k, v in cluster_summaries.items():
                if isinstance(v, dict) and "summary" in v:
                    cluster_summaries_to_print[k] = v["summary"]
                else:
                    cluster_summaries_to_print[k] = str(v)
                    
            for k, v in cluster_summaries_to_print.items():
                print(f"Cluster {k}: {v}")
                
        # Show figure
        fig.show()
            
        return result
