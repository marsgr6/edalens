import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from scipy.cluster import hierarchy
import warnings
import re
from scipy.stats import t

from edalens.utils.plot_utils import (
    FIT_METHODS, HEATMAP_ROW_ORDERS, HEATMAP_SCALINGS, MARGINAL_KINDS,
    numeric_heatmap_figure, regression_figure,
)

warnings.filterwarnings('ignore')

def visualization():
    #st.subheader("🧪 Visualization: Interactive Data Exploration")

    # Check if filtered data exists in session state
    if "filtered_df" not in st.session_state:
        st.warning("Please upload and process a dataset in the Home section first.")
        return

    # Work on a copy: presentation changes below must not alter the shared
    # working frame used by Home, Missing Data, undo, and downloads.
    data = st.session_state["filtered_df"].copy()
    
    # Check if imputation has been applied based on operation history
    if "operations" in st.session_state and any(op["type"] == "impute" for op in st.session_state["operations"]):
        st.success("Using dataset with imputed values for visualization.")

    # Check if current_numeric_cols and current_categorical_cols exist
    if "current_numeric_cols" not in st.session_state or "current_categorical_cols" not in st.session_state:
        st.warning("Column type information is missing. Please assign variable types in the Home section.")
        return

    numeric_cols = st.session_state["current_numeric_cols"]
    categorical_cols = st.session_state["current_categorical_cols"]

    # Convert categorical columns to strings
    for col in data.select_dtypes(include='category').columns:
        data[col] = data[col].astype('str')

    # Define plot types
    PLOT_TYPES = [
        'bars', 'boxes', 'lineplot', 'histogram', 'density 1', 'density 2',
        'scatter', 'catplot', 'missingno', 'correlation', 'clustermap',
        'pairplot', 'regression', "heatmap", 'ridges'
    ]

    if not numeric_cols:
        PLOT_TYPES = [
            'bars', 'boxes', 'histogram', 'density 2',
            'scatter', 'catplot', 'missingno'
        ]

    # --- SESSION STATE PERSISTENCE LOGIC ---
    if "viz_state" not in st.session_state:
        st.session_state.viz_state = {
            "plot_type": "bars",
            "x": numeric_cols[0] if numeric_cols else (categorical_cols[0] if categorical_cols else None),
            "y": numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else None),
            "color": "None",
            "facet_col": "None",
            "facet_row": "None",
            "size": "None"
        }

    if "_plot_type_selector" not in st.session_state:
        st.session_state._plot_type_selector = st.session_state.viz_state["plot_type"]

    def handle_plot_change():
        new_plot = st.session_state.get("_plot_type_selector", st.session_state.viz_state.get("plot_type", "bars"))
        state = st.session_state.viz_state
        state["plot_type"] = new_plot

        # Helpers to get default fallback columns safely
        first_cat = categorical_cols[0] if categorical_cols else (data.columns[0] if len(data.columns) > 0 else None)
        first_num = numeric_cols[0] if numeric_cols else (data.columns[0] if len(data.columns) > 0 else None)
        second_num = numeric_cols[1] if len(numeric_cols) > 1 else first_num

        # Smart defaults and persistence depending on exact plot requirements
        if new_plot in ["bars", "catplot", "lineplot"]:
            if state["x"] not in categorical_cols: 
                state["x"] = first_cat
            if new_plot == "bars": 
                state["y"] = "None"
            else: 
                if state["y"] not in numeric_cols: 
                    state["y"] = first_num
            
        elif new_plot == "boxes":
            # Boxplots: optional categorical grouping on X (or None for single box),
            # required numerical variable on Y for the distribution.
            # (Supports horizontal orientation internally if roles are swapped via Risk It All.)
            if state["x"] not in categorical_cols + ["None"]:
                state["x"] = first_cat if first_cat else "None"
            # Ensure Y is always a numeric column when possible (do not allow None)
            if state["y"] not in numeric_cols:
                state["y"] = first_num if first_num else (data.columns[0] if len(data.columns) > 0 else "None")
                
        elif new_plot in ["scatter", "regression", "density 2"]:
            # These plots strictly require two numerical variables
            if state["x"] not in numeric_cols: 
                state["x"] = first_num
            if state["y"] not in numeric_cols: 
                state["y"] = second_num
        
        elif new_plot in ["histogram", "density 1"]:
            # Single numerical variable plots
            if state["x"] not in numeric_cols: 
                state["x"] = first_num
            state["y"] = "None" # Y is ignored or forced to count/density

        elif new_plot == "heatmap":
            if state["x"] not in categorical_cols + ["None"]: 
                state["x"] = "None"
            if state["y"] not in numeric_cols: 
                state["y"] = first_num

        elif new_plot == "ridges":
            # Ridges plot distributions (numeric) over categories (categorical)
            if state["x"] not in numeric_cols: 
                state["x"] = first_num
            if state["y"] not in categorical_cols: 
                state["y"] = first_cat

    def check_cardinality(var_name):
        if var_name and var_name != "None" and var_name in categorical_cols:
            n_unique = data[var_name].nunique()
            if n_unique > 50:
                st.warning(f"⚠️ High cardinality in `{var_name}` ({n_unique} unique values). Rendering may be slow or cluttered.")

    # --- SIDEBAR CONTROLS (Plot Type & Risk) ---
    with st.sidebar:
        st.markdown("### Controls")
        risk_it_all = st.checkbox("Risk It All (Unlock all variable mappings)", value=False)

    # --- VARIABLE POOL (VISUAL DISPLAY) ---
    with st.expander("🗂️ Variables Pool", expanded=False):
        cat_badges = "".join([f"<span style='background-color:#FF7F0E; color:white; padding:4px 10px; border-radius:12px; margin:4px; display:inline-block; font-size:12px;'>{c} (cat)</span>" for c in categorical_cols])
        num_badges = "".join([f"<span style='background-color:#1F77B4; color:white; padding:4px 10px; border-radius:12px; margin:4px; display:inline-block; font-size:12px;'>{n} (num)</span>" for n in numeric_cols])
        
        # Added max-height and overflow-y to enable vertical scrolling
        st.markdown(
            f"""
            <div style='
                padding: 10px; 
                border: 1px dashed #ccc; 
                border-radius: 5px; 
                text-align: center; 
                max-height: 120px; 
                overflow-y: auto; 
                margin-bottom: 10px;
            '>
                {cat_badges}{num_badges}
            </div>
            """, 
            unsafe_allow_html=True
        )

    # --- MAPPING ZONES (HORIZONTAL DROPDOWNS) ---
    current_plot = st.session_state._plot_type_selector
    
    # Determine allowed options based on current plot and risk_it_all
    if risk_it_all:
        x_opts = data.columns.tolist()
        y_opts = data.columns.tolist()
    else:
        # Smart X mapping based on standard statistical practices
        if current_plot in ['bars', 'boxes', 'catplot', 'lineplot', 'heatmap']:
            x_opts = categorical_cols.copy() if categorical_cols else data.columns.tolist()
        else:  # scatter, histogram, density 1/2, regression, ridges
            x_opts = numeric_cols.copy() if numeric_cols else data.columns.tolist()

        # Smart Y mapping based on standard statistical practices
        if current_plot in ['ridges']:
            y_opts = categorical_cols.copy() if categorical_cols else data.columns.tolist()
        elif current_plot in ['histogram', 'density 1']:
            y_opts = []  # Force to None below
        elif current_plot == 'boxes':
            # Boxes require a numeric Y (distribution); X is optional categorical grouping
            y_opts = numeric_cols.copy() if numeric_cols else data.columns.tolist()
        else:
            y_opts = numeric_cols.copy() if numeric_cols else data.columns.tolist()

    # Fallback overrides to guarantee UI doesn't crash on empty selections
    if not x_opts:
        x_opts = data.columns.tolist()
    if not y_opts:
        y_opts = data.columns.tolist()

    # Append 'None' allowances where applicable
    # For boxes: X may be None (single boxplot) or categorical; Y must stay numeric (no None)
    if current_plot in ['boxes', 'heatmap']:
        if 'None' not in x_opts:
            x_opts = ['None'] + x_opts
    if current_plot == 'bars':
        if 'None' not in y_opts:
            y_opts = ['None'] + y_opts
    # Note: intentionally do NOT allow None for Y when plot_type == 'boxes'
    if current_plot in ['histogram', 'density 1']:
        y_opts = ['None']  # Completely overrides Y to None

    cat_opts = ["None"] + (data.columns.tolist() if risk_it_all else categorical_cols)
    size_opts = ["None"] + (data.columns.tolist() if risk_it_all else numeric_cols)
    
    # Safely retrieve states for index mapping
    s_x = st.session_state.viz_state["x"] if st.session_state.viz_state["x"] in x_opts else (x_opts[0] if x_opts else None)
    s_y = st.session_state.viz_state["y"] if st.session_state.viz_state["y"] in y_opts else (y_opts[0] if y_opts else None)
    s_color = st.session_state.viz_state["color"] if st.session_state.viz_state["color"] in cat_opts else "None"
    s_facet_c = st.session_state.viz_state["facet_col"] if st.session_state.viz_state["facet_col"] in cat_opts else "None"
    s_facet_r = st.session_state.viz_state["facet_row"] if st.session_state.viz_state["facet_row"] in cat_opts else "None"
    s_size = st.session_state.viz_state["size"] if st.session_state.viz_state["size"] in size_opts else "None"

    z1, z2, z3, z4, z5, z6, z7 = st.columns(7)
    
    # Create a dictionary to map your original names to the icon version
    # Updated dictionary with clearer visual metaphors
    PLOT_ICONS = {
        'bars': '📊 bars', 
        'boxes': '📦 boxes', 
        'lineplot': '📈 lineplot', 
        'histogram': '📶 histogram', 
        'density 1': '⛰️ density 1',    # Represents a single peak/bell curve
        'density 2': '🌐 density 2',    # Represents a 3D contour map or globe grid lines
        'scatter': '🌌 scatter',  # Represents a field of scattered points/stars 
        'catplot': '🔀 catplot', 
        'missingno': '🕳️ missingno', 
        'correlation': '🔗 correlation',  # Represents attraction/linkage between variables
        'clustermap': '🌳 clustermap',  # Represents a hierarchical tree/dendrogram structure
        'pairplot': '🪟 pairplot',      # Represents a matrix grid of window panes
        'regression': '📉 regression', 
        "heatmap": '🔥 heatmap', 
        'ridges': '🏔️ ridges'
    }


    with z1:
        plot_idx = PLOT_TYPES.index(st.session_state.viz_state["plot_type"]) if st.session_state.viz_state["plot_type"] in PLOT_TYPES else 0
        st.markdown("**Plot**")
        st.selectbox(
            "Plot", 
            PLOT_TYPES, 
            key="_plot_type_selector", 
            label_visibility="collapsed",
            index=plot_idx, 
            on_change=handle_plot_change,
            format_func=lambda x: PLOT_ICONS.get(x, x)  # Shows icons without changing the actual value
        )
    with z2:
        st.markdown("**X**")
        st.selectbox("X", x_opts, key="x_zone", label_visibility="collapsed", index=x_opts.index(s_x) if s_x in x_opts else 0)
    with z3:
        st.markdown("**Y**")
        st.selectbox("Y", y_opts, key="y_zone", label_visibility="collapsed", index=y_opts.index(s_y) if s_y in y_opts else 0)
    with z4:
        st.markdown("**Color (Hue)**")
        st.selectbox("Color", cat_opts, key="color_zone", label_visibility="collapsed", index=cat_opts.index(s_color))
    with z5:
        st.markdown("**Facet Col**")
        st.selectbox("Facet Col", cat_opts, key="facet_col_zone", label_visibility="collapsed", index=cat_opts.index(s_facet_c))
    with z6:
        st.markdown("**Facet Row**")
        st.selectbox("Facet Row", cat_opts, key="facet_row_zone", label_visibility="collapsed", index=cat_opts.index(s_facet_r))
    with z7:
        st.markdown("**Size**")
        st.selectbox("Size", size_opts, key="size_zone", label_visibility="collapsed", index=size_opts.index(s_size))

    # Synchronize state on change
    st.session_state.viz_state["x"] = st.session_state.x_zone
    st.session_state.viz_state["y"] = st.session_state.y_zone
    st.session_state.viz_state["color"] = st.session_state.color_zone
    st.session_state.viz_state["facet_col"] = st.session_state.facet_col_zone
    st.session_state.viz_state["facet_row"] = st.session_state.facet_row_zone
    st.session_state.viz_state["size"] = st.session_state.size_zone

    check_cardinality(st.session_state.color_zone)
    check_cardinality(st.session_state.facet_col_zone)
    check_cardinality(st.session_state.facet_row_zone)

    # Map UI zones to plotting logic variables
    plot_type = current_plot
    global_var_x = st.session_state.x_zone if st.session_state.x_zone != "None" else None
    global_var_y = st.session_state.y_zone if st.session_state.y_zone != "None" else None
    global_hue = st.session_state.color_zone if st.session_state.color_zone != "None" else None
    global_facet_col = st.session_state.facet_col_zone if st.session_state.facet_col_zone != "None" else None
    global_facet_row = st.session_state.facet_row_zone if st.session_state.facet_row_zone != "None" else None
    global_size = st.session_state.size_zone if st.session_state.size_zone != "None" else None
    
    PALETTE = px.colors.qualitative.Set2

    def render_side_by_side(config_fn, plot_fn, config_width=0.32):
        config_col, plot_col = st.columns([config_width, 1 - config_width], gap="large")
        with config_col:
            config_fn()
        with plot_col:
            plot_fn()

    # Plot rendering function
    def render_plot():
        # Non-interactive plot types
        if plot_type == "correlation":
            data_c = data.drop(data.columns[data.nunique() == 1], axis=1)
            corr_matrix = data_c.select_dtypes(include=np.number).corr()
            fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Viridis', 
                           title="Correlation Matrix", width=800, height=600)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "clustermap":
            with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                z_score = st.selectbox("Z-Score", [None, 0, 1], index=0)
                standard_scale = st.selectbox("Standard Scale", [None, 0, 1], index=0)
                
            data_c = data.drop(data.columns[data.nunique() == 1], axis=1)
            numeric_data = data_c.select_dtypes(include='number').dropna()

            if z_score is not None:
                numeric_data = (numeric_data - numeric_data.mean()) / numeric_data.std()
            if standard_scale is not None:
                numeric_data = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())
            
            corr_matrix = numeric_data.corr()
            labels = corr_matrix.columns
            row_linkage = hierarchy.linkage(corr_matrix, method='average')
            col_linkage = hierarchy.linkage(corr_matrix.T, method='average')
            row_order = hierarchy.leaves_list(row_linkage)
            col_order = hierarchy.leaves_list(col_linkage)
            corr_matrix = corr_matrix.iloc[row_order, col_order]
            labels = corr_matrix.columns
            
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=labels,
                    y=labels,
                    colorscale='Viridis',
                    showscale=True,
                    text=corr_matrix.values.round(2),
                    texttemplate="%{text}",
                    textfont={"size": 10}
                )
            )
            fig.update_layout(
                title="Clustermap with Clustering",
                width=800,
                height=800,
                xaxis=dict(side="top", showticklabels=True),
                yaxis=dict(showticklabels=True),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "pairplot":
            numeric_cols_list = list(data.select_dtypes(include='number').columns)
            cat_cols = list(data.select_dtypes(include=['object', 'category', 'bool']).columns)

            if not numeric_cols_list:
                st.warning("No numeric columns available for pairplot.")
                return

            hue_var = global_hue

            n_cols = len(numeric_cols_list)
            fig = make_subplots(
                rows=n_cols, cols=n_cols,
                row_titles=numeric_cols_list,
                column_titles=numeric_cols_list,
                vertical_spacing=0.05,
                horizontal_spacing=0.05,
                shared_xaxes=False,
                shared_yaxes=False
            )

            for i in range(n_cols):
                col_i = numeric_cols_list[i]
                for j in range(n_cols):
                    col_j = numeric_cols_list[j]

                    if i == j:
                        if hue_var is None:
                            subset = data[col_i].dropna()
                            if len(subset) >= 2:
                                kde = gaussian_kde(subset)
                                x_values = np.linspace(min(subset), max(subset), 100)
                                density = kde(x_values)
                                density = density / density.max()
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_values, y=density, mode='lines', name=col_i,
                                        line=dict(color=PALETTE[0]), showlegend=(i == 0 and j == 0)
                                    ), row=i+1, col=j+1
                                )
                        else:
                            for k, hue_val in enumerate(data[hue_var].dropna().unique()):
                                subset = data[data[hue_var] == hue_val][col_i].dropna()
                                if len(subset) < 2: continue
                                try:
                                    kde = gaussian_kde(subset)
                                    x_values = np.linspace(min(subset), max(subset), 100)
                                    density = kde(x_values)
                                    density = density / density.max()
                                    fig.add_trace(
                                        go.Scatter(
                                            x=x_values, y=density, mode='lines', name=str(hue_val),
                                            line=dict(color=PALETTE[k % len(PALETTE)]), showlegend=(i == 0 and j == 0)
                                        ), row=i+1, col=j+1
                                    )
                                except Exception:
                                    continue
                    else:
                        if hue_var is None:
                            subset = data[[col_j, col_i]].dropna()
                            if subset.empty: continue
                            fig.add_trace(
                                go.Scatter(
                                    x=subset[col_j], y=subset[col_i], mode='markers',
                                    marker=dict(color=PALETTE[0], size=5), name="Data", showlegend=False
                                ), row=i+1, col=j+1
                            )
                        else:
                            for k, hue_val in enumerate(data[hue_var].dropna().unique()):
                                subset = data[data[hue_var] == hue_val][[col_j, col_i]].dropna()
                                if subset.empty: continue
                                fig.add_trace(
                                    go.Scatter(
                                        x=subset[col_j], y=subset[col_i], mode='markers',
                                        marker=dict(color=PALETTE[k % len(PALETTE)], size=5), name=str(hue_val), showlegend=False
                                    ), row=i+1, col=j+1
                                )

            fig.update_layout(title="Pairplot (KDE Diagonals, Scatter Off-Diagonals)", width=800, height=800, showlegend=True)

            for i in range(n_cols):
                for j in range(n_cols):
                    col_i = numeric_cols_list[i]
                    col_j = numeric_cols_list[j]
                    x_data = data[col_j].dropna()
                    y_data = data[col_i].dropna()
                    x_range = [min(x_data), max(x_data)] if not x_data.empty else [0, 1]
                    y_range = [0, 1.2] if i == j else [min(y_data), max(y_data)] if not y_data.empty else [0, 1]
                    x_margin = (x_range[1] - x_range[0]) * 0.05
                    y_margin = (y_range[1] - y_range[0]) * 0.05 if i != j else 0.1
                    x_range = [x_range[0] - x_margin, x_range[1] + x_margin]
                    y_range = [y_range[0] - y_margin, y_range[1] + y_margin]
                    fig.update_xaxes(range=x_range, row=i+1, col=j+1)
                    fig.update_yaxes(range=y_range, row=i+1, col=j+1)

            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "missingno":
            with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                tplot = st.selectbox("Plot Type", ["matrix", "bars", "heatmap", "dendrogram"])
                if tplot == "bars":
                    hide_no_missing = st.checkbox("Hide Variables Without Missing Values", value=False)
                else:
                    hide_no_missing = False

            mapped_data = (~data.isna()).astype(int)

            if tplot == "matrix":
                fig = px.imshow(
                    mapped_data.values, labels=dict(x="Columns", y="Rows", color="Present"),
                    x=mapped_data.columns, y=mapped_data.index,
                    color_continuous_scale="viridis", range_color=[0, 1], aspect='auto', width=1000, height=600
                )
                fig.update_layout(title="Missing Data Matrix (1 = Present, 0 = Missing)")
                st.plotly_chart(fig, use_container_width=True)

            elif tplot == "bars":
                mapped_data = data.notnull().astype(int)
                missing_counts = mapped_data.sum(axis=0)

                if hide_no_missing:
                    missing_counts = missing_counts[missing_counts < mapped_data.shape[0]]

                if missing_counts.empty:
                    st.warning("No variables with missing data to plot.")
                    st.stop()

                missing_counts = missing_counts.sort_values(ascending=False)
                df_missing = pd.DataFrame({"Variable": missing_counts.index, "NonMissingCount": missing_counts.values})

                fig = px.bar(
                    df_missing, x="Variable", y="NonMissingCount", color="NonMissingCount",
                    color_continuous_scale="Viridis", text="NonMissingCount",
                    title="Non-Missing Data Count per Variable", width=800, height=600
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(xaxis_title="Variables", yaxis_title="Count of Non-Missing Values", margin=dict(l=80, r=20, t=50, b=150), xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

            elif tplot == "heatmap":
                from scipy.spatial.distance import squareform
                from scipy.cluster.hierarchy import linkage, leaves_list

                mapped_data = data.notnull().astype(int)
                cols_with_missing = mapped_data.columns[mapped_data.nunique() > 1]
                mapped_data = mapped_data[cols_with_missing]

                if mapped_data.shape[1] < 2:
                    st.warning("Not enough variables with missing data to plot a heatmap.")
                    st.stop()

                corr_matrix = mapped_data.corr()
                corr_matrix = (corr_matrix + corr_matrix.T) / 2
                distance_matrix = 1 - corr_matrix
                np.fill_diagonal(distance_matrix.values, 0)
                condensed_distance = squareform(distance_matrix.values)

                linkage_matrix = linkage(condensed_distance, method='ward')
                ordered_indices = leaves_list(linkage_matrix)
                ordered_columns = corr_matrix.columns[ordered_indices]
                sorted_corr = corr_matrix.loc[ordered_columns, ordered_columns]
                sorted_corr_masked = sorted_corr.copy()
                sorted_corr_masked.values[np.triu_indices_from(sorted_corr_masked, 1)] = np.nan

                fig = px.imshow(
                    sorted_corr_masked, text_auto=".2f", color_continuous_scale="Viridis",
                    zmin=0, zmax=1, title="Lower Triangular Missingness Correlation Clustermap", width=800, height=800
                )
                fig.update_layout(xaxis_title="Variables", yaxis_title="Variables", xaxis_side="bottom", yaxis_autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)

            elif tplot == "dendrogram":
                import plotly.figure_factory as ff
                from scipy.spatial.distance import squareform
                from scipy.cluster.hierarchy import linkage, leaves_list

                mapped_data = data.notnull().astype(int)
                cols_with_missing = mapped_data.columns[mapped_data.nunique() > 1]
                mapped_data = mapped_data[cols_with_missing]

                if mapped_data.shape[1] < 2:
                    st.warning("Not enough variables with missing data to plot a dendrogram.")
                    st.stop()

                corr_matrix = mapped_data.corr()
                corr_matrix = (corr_matrix + corr_matrix.T) / 2
                distance_matrix = 1 - corr_matrix
                np.fill_diagonal(distance_matrix.values, 0)
                condensed_distance = squareform(distance_matrix.values)

                linkage_matrix = linkage(condensed_distance, method="ward")
                ordered_indices = leaves_list(linkage_matrix)
                ordered_columns = corr_matrix.columns[ordered_indices]
                sorted_corr = corr_matrix.loc[ordered_columns, ordered_columns]

                dendro = ff.create_dendrogram(
                    mapped_data.T.values, orientation='right', labels=list(mapped_data.columns),
                    linkagefun=lambda _: linkage_matrix, color_threshold=None
                )
                heatmap = go.Heatmap(
                    z=sorted_corr.values, x=ordered_columns, y=ordered_columns,
                    colorscale='Viridis', colorbar=dict(title="Missingness Corr"), zmin=0, zmax=1
                )

                fig = go.Figure()
                for trace in dendro['data']: fig.add_trace(trace)
                for trace in fig['data']:
                    if trace['xaxis'] == 'x2': trace['xaxis'] = 'x1'
                    if trace['yaxis'] == 'y2': trace['yaxis'] = 'y1'
                fig.add_trace(heatmap)
                fig.update_layout(
                    width=1000, height=1000, showlegend=False,
                    xaxis=dict(domain=[0.3, 1], tickmode='array', tickvals=list(range(len(ordered_columns))), ticktext=ordered_columns, tickangle=45),
                    yaxis=dict(domain=[0, 0.7], tickmode='array', tickvals=list(range(len(ordered_columns))), ticktext=ordered_columns[::-1], autorange='reversed'),
                    xaxis2=dict(domain=[0, 0.2]), yaxis2=dict(domain=[0.7, 1]), margin=dict(l=100, t=50, b=100)
                )
                st.plotly_chart(fig, use_container_width=True)

        # Interactive plot types
        else:
            if plot_type == 'bars':
                var_x = global_var_x
                hue = global_hue
                facet_col = global_facet_col if global_facet_col else 'None'
                facet_row = global_facet_row if global_facet_row else 'None'

                with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                    tplot = st.selectbox("Plot Type", ["bars", "heatmap"])
                    bar_mode = st.selectbox("Bars Mode", ["Histogram", "Raw Values (height = value in order)"])
                    time_series = st.checkbox("Time is here", value=False) if tplot == "bars" else False
                    var_y = global_var_y if time_series else None
                    
                    category_orders = {}
                    plot_data = data.copy()

                    if hue is not None: plot_data[hue] = plot_data[hue].astype(str)

                    if var_x in plot_data.columns and plot_data[var_x].dtype.name in ['object', 'category']:
                        custom_order_x = st.multiselect(f"Custom Order for X: {var_x}", options=plot_data[var_x].dropna().unique().tolist(), default=sorted(plot_data[var_x].dropna().unique().tolist()))
                        plot_data[var_x] = pd.Categorical(plot_data[var_x], categories=custom_order_x, ordered=True)
                        category_orders[var_x] = custom_order_x

                    if hue and hue != 'None' and hue in plot_data.columns and plot_data[hue].dtype.name in ['object', 'category']:
                        custom_order_hue = st.multiselect(f"Custom Order for Hue: {hue}", options=plot_data[hue].dropna().unique().tolist(), default=sorted(plot_data[hue].dropna().unique().tolist()))
                        plot_data[hue] = pd.Categorical(plot_data[hue], categories=custom_order_hue, ordered=True)
                        category_orders[hue] = custom_order_hue

                    if facet_col != 'None' and facet_col in plot_data.columns and plot_data[facet_col].dtype.name in ['object', 'category']:
                        custom_order_col = st.multiselect(f"Custom Order for Col: {facet_col}", options=plot_data[facet_col].dropna().unique().tolist(), default=sorted(plot_data[facet_col].dropna().unique().tolist()))
                        plot_data[facet_col] = pd.Categorical(plot_data[facet_col], categories=custom_order_col, ordered=True)
                        category_orders[facet_col] = custom_order_col

                    if facet_row != 'None' and facet_row in plot_data.columns and plot_data[facet_row].dtype.name in ['object', 'category']:
                        custom_order_row = st.multiselect(f"Custom Order for Row: {facet_row}", options=plot_data[facet_row].dropna().unique().tolist(), default=sorted(plot_data[facet_row].dropna().unique().tolist()))
                        plot_data[facet_row] = pd.Categorical(plot_data[facet_row], categories=custom_order_row, ordered=True)
                        category_orders[facet_row] = custom_order_row

                if tplot == "bars":
                    if time_series:
                        if var_y is None:
                            st.warning("Please select a Y variable in the Mapping Zones for Time Series mode.")
                            return

                        is_object = False
                        if isinstance(plot_data[var_x].dtype, pd.CategoricalDtype):
                            if plot_data[var_x].cat.categories.dtype.name == 'object': is_object = True
                        else:
                            if plot_data[var_x].dtype.name == 'object': is_object = True

                        if not is_object:
                            st.warning(f"Time series mode requires '{var_x}' to be an object column (e.g., dates).")
                            return

                        if not pd.api.types.is_numeric_dtype(plot_data[var_y]):
                            st.warning(f"Time series mode requires '{var_y}' to be a numerical column.")
                            return

                        facet_col_vals = plot_data[facet_col].dropna().unique() if facet_col != 'None' else [None]
                        facet_row_vals = plot_data[facet_row].dropna().unique() if facet_row != 'None' else [None]

                        n_cols = max(len(facet_col_vals), 1)
                        n_rows = max(len(facet_row_vals), 1)

                        fig = make_subplots(
                            rows=n_rows, cols=n_cols, shared_yaxes=True, shared_xaxes=True,
                            horizontal_spacing=0.1, vertical_spacing=0.12,
                            subplot_titles=[
                                f"{facet_row}: {r} | {facet_col}: {c}" if facet_row != 'None' and facet_col != 'None' else
                                f"{facet_col}: {c}" if facet_col != 'None' else
                                f"{facet_row}: {r}" if facet_row != 'None' else ""
                                for r in facet_row_vals for c in facet_col_vals
                            ]
                        )

                        hue_values = plot_data[hue].dropna().unique() if hue is not None else [None]
                        color_map = {hv: PALETTE[idx % len(PALETTE)] for idx, hv in enumerate(hue_values)}

                        for i_row, row_val in enumerate(facet_row_vals):
                            for i_col, col_val in enumerate(facet_col_vals):
                                row_idx = i_row + 1
                                col_idx = i_col + 1
                                sub_data = plot_data.copy()
                                if facet_row != 'None': sub_data = sub_data[sub_data[facet_row] == row_val]
                                if facet_col != 'None': sub_data = sub_data[sub_data[facet_col] == col_val]

                                if sub_data.empty: continue

                                for hue_val in hue_values:
                                    subset = sub_data.copy()
                                    if hue_val is not None: subset = subset[subset[hue] == hue_val]
                                    if subset.empty: continue

                                    fig.add_trace(
                                        go.Bar(
                                            x=subset[var_x], y=subset[var_y], name=str(hue_val) if hue_val else "Bars",
                                            marker_color=color_map[hue_val], showlegend=(row_idx == 1 and col_idx == 1)
                                        ), row=row_idx, col=col_idx
                                    )

                        fig.update_layout(width=800, height=600, title=f"Time Series Bar Plot: {var_y} vs {var_x}", xaxis_title=var_x, yaxis_title=var_y, legend_title=hue, showlegend=True, barmode='group')

                    else:
                        if bar_mode == "Histogram":
                            fig = px.histogram(
                                plot_data, x=var_x, color=hue, barmode='group',
                                facet_col=facet_col if facet_col != 'None' else None,
                                facet_row=facet_row if facet_row != 'None' else None,
                                color_discrete_sequence=PALETTE, category_orders=category_orders, width=800, height=600
                            )
                            fig.update_traces(texttemplate='%{y}', textposition='auto')
                        else:
                            df_plot = plot_data.reset_index()
                            x_axis = 'index'
                            y_axis = var_x
                            fig = px.bar(
                                df_plot, x=x_axis, y=y_axis, color=hue,
                                facet_col=facet_col if facet_col != 'None' else None,
                                facet_row=facet_row if facet_row != 'None' else None,
                                color_discrete_sequence=PALETTE, category_orders=category_orders, width=800, height=600
                            )
                            fig.update_layout(xaxis_title="Index", yaxis_title=y_axis)

                else:
                    if hue is not None and var_x is not None:
                        if pd.api.types.is_numeric_dtype(plot_data[var_x]): plot_data[var_x] = pd.cut(plot_data[var_x], bins=10)
                        if pd.api.types.is_numeric_dtype(plot_data[hue]): plot_data[hue] = pd.cut(plot_data[hue], bins=10)
                        df_2dhist = pd.crosstab(plot_data[var_x], plot_data[hue])
                        fig = px.imshow(
                            df_2dhist, text_auto='.0f', color_continuous_scale='Viridis', title=f"Heatmap: {hue} vs {var_x}", width=800, height=600
                        )
                        fig.update_xaxes(title=hue)
                        fig.update_yaxes(title=var_x)
                    else:
                        st.warning("You must select both X and Hue variables to create a heatmap.")
                        return

                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == 'boxes':
                var_x_used = global_var_x
                var_y_used = global_var_y
                hue_used = global_hue
                facet_col_used = global_facet_col
                facet_row_used = global_facet_row

                # Enforce expected roles when Risk It All is off:
                # - Y must be numeric (distribution values)
                # - X may be None or categorical (grouping)
                # When Risk It All is on, allow role swap which triggers horizontal orientation.
                x_is_numeric = var_x_used is not None and var_x_used in numeric_cols
                y_is_numeric = var_y_used is not None and var_y_used in numeric_cols

                if not risk_it_all:
                    # Strict mode: require numeric Y; treat non-categorical X as invalid → None
                    if not y_is_numeric:
                        st.warning("Boxplots require a numeric Y variable. Please select a numeric column for Y.")
                        return
                    if var_x_used is not None and var_x_used not in categorical_cols:
                        st.warning(f"X variable '{var_x_used}' is not categorical. Treating as no grouping (single box).")
                        var_x_used = None
                    orientation = 'v'
                    num_var = var_y_used
                    cat_var = var_x_used
                else:
                    # Risk mode: support both vertical (X=cat, Y=num) and horizontal (X=num, Y=cat)
                    if not (x_is_numeric or y_is_numeric):
                        st.warning("At least one of X or Y must be numeric for a boxplot.")
                        return
                    orientation = 'h' if (x_is_numeric and not y_is_numeric) else 'v'
                    num_var = var_x_used if orientation == 'h' else var_y_used
                    cat_var = var_y_used if orientation == 'h' else var_x_used
                    if num_var is None or (num_var not in numeric_cols and not pd.api.types.is_numeric_dtype(data[num_var])):
                        st.warning("Could not determine a numeric variable for the boxplot distribution.")
                        return

                show_swarm_checkbox = (facet_col_used is None and facet_row_used is None and (hue_used is None or hue_used == cat_var))

                with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                    tplot = st.selectbox("Plot Type", ["boxplot", "violin"])
                    swarm_points = st.checkbox("Overlay Swarm of Points", value=False) if show_swarm_checkbox else False
                    band_interval = st.checkbox("Show Band Interval (CI around means)", value=False)
                    # leftover time_series checkbox removed (only relevant for lineplot)
                    ci_level = st.selectbox("Confidence Interval Level", ["68%", "95%", "99%"], index=1) if band_interval else "95%"
                    ci_level_value = {"68%": 0.68, "95%": 0.95, "99%": 0.99}[ci_level]

                    plot_data = data.copy()
                    category_orders = {}
                    
                    if hue_used is not None and hue_used in plot_data.columns: plot_data[hue_used] = plot_data[hue_used].astype(str)

                    for axis_var, label in [(cat_var, 'x' if orientation == 'v' else 'y'), (hue_used, 'hue'), (facet_col_used, 'col'), (facet_row_used, 'row')]:
                        if axis_var and axis_var in plot_data.columns and plot_data[axis_var].dtype.name in ['object', 'category']:
                            opts = plot_data[axis_var].dropna().unique().tolist()
                            order = st.multiselect(f"Custom Order for {label}: {axis_var}", options=opts, default=sorted(opts))
                            plot_data[axis_var] = pd.Categorical(plot_data[axis_var], categories=order, ordered=True)
                            category_orders[axis_var] = order

                point_mode = 'all' if swarm_points else False
                jitter_val = 0.4
                point_kwargs = dict(jitter=jitter_val, pointpos=0, marker=dict(size=5, opacity=0.5, line=dict(width=0)))

                if hue_used:
                    groups = plot_data[hue_used].dropna().unique().tolist()
                    if hue_used in category_orders: groups = [g for g in category_orders[hue_used] if g in groups]
                elif cat_var:
                    groups = plot_data[cat_var].dropna().unique().tolist()
                    if cat_var in category_orders: groups = [g for g in category_orders[cat_var] if g in groups]
                else:
                    groups = [None]

                fig = go.Figure()

                for i, grp in enumerate(groups):
                    if hue_used:
                        mask = plot_data[hue_used] == grp
                        sub = plot_data[mask]
                        if cat_var:
                            cats = sub[cat_var].dropna().unique().tolist()
                            if cat_var in category_orders: cats = [c for c in category_orders[cat_var] if c in cats]
                            values, cat_values = sub[num_var], sub[cat_var]
                        else:
                            values, cat_values = sub[num_var], None
                    elif cat_var:
                        mask = plot_data[cat_var] == grp
                        sub = plot_data[mask]
                        values, cat_values = sub[num_var], None
                    else:
                        sub = plot_data
                        values, cat_values = sub[num_var], None

                    trace_name = str(grp) if grp is not None else num_var
                    color = PALETTE[i % len(PALETTE)]

                    if orientation == 'v':
                        if hue_used and cat_var: tx, ty = sub[cat_var], values
                        elif cat_var: tx, ty = None, values
                        else: tx, ty = [0] * len(values), values
                    else:
                        if hue_used and cat_var: tx, ty = values, sub[cat_var]
                        elif cat_var: tx, ty = values, None
                        else: tx, ty = values, [0] * len(values)

                    common = dict(name=trace_name, marker_color=color, legendgroup=trace_name, **point_kwargs)
                    if tx is not None: common['x'] = tx
                    if ty is not None: common['y'] = ty
                    if orientation == 'h': common['orientation'] = 'h'

                    if tplot == "boxplot":
                        fig.add_trace(go.Box(**common, boxpoints=point_mode))
                    else:
                        fig.add_trace(go.Violin(**common, points=point_mode, box_visible=True, meanline_visible=True, spanmode='manual', span=[values.min(), values.max()]))

                single_var = cat_var is None

                if band_interval and cat_var and not single_var:
                    from scipy import stats as _stats
                    cat_order = category_orders.get(cat_var, sorted(plot_data[cat_var].dropna().unique().tolist()))
                    line_groups = plot_data[hue_used].dropna().unique().tolist() if hue_used else [None]
                    if hue_used and hue_used in category_orders: line_groups = [g for g in category_orders[hue_used] if g in line_groups]

                    for li, lgrp in enumerate(line_groups):
                        sub_lg = plot_data[plot_data[hue_used] == lgrp] if lgrp is not None else plot_data
                        means, lo, hi = [], [], []

                        for cat in cat_order:
                            vals = sub_lg.loc[sub_lg[cat_var] == cat, num_var].dropna()
                            if len(vals) < 2:
                                means.append(float('nan')); lo.append(float('nan')); hi.append(float('nan'))
                                continue
                            m = vals.mean()
                            se = _stats.sem(vals)
                            z = _stats.norm.ppf(1 - (1 - ci_level_value) / 2)
                            means.append(m); lo.append(m - z * se); hi.append(m + z * se)

                        line_color = "black"
                        group_label = str(lgrp) if lgrp is not None else num_var
                        if orientation == 'v': xs_line, ys_line, xs_band, ys_band = cat_order, means, cat_order + cat_order[::-1], hi + lo[::-1]
                        else: xs_line, ys_line, xs_band, ys_band = means, cat_order, hi + lo[::-1], cat_order + cat_order[::-1]

                        fig.add_trace(go.Scatter(x=xs_band, y=ys_band, fill='toself', fillcolor=line_color, opacity=0.15, line=dict(width=0), hoverinfo='skip', showlegend=False, legendgroup=group_label))
                        fig.add_trace(go.Scatter(x=xs_line, y=ys_line, mode='lines+markers', line=dict(color=line_color, width=2), marker=dict(size=7, color=line_color), name=f"{group_label} mean", legendgroup=group_label, showlegend=True))

                elif band_interval and single_var:
                    from scipy import stats as _stats
                    hue_groups = plot_data[hue_used].dropna().unique().tolist() if hue_used else [None]
                    if hue_used and hue_used in category_orders: hue_groups = [g for g in category_orders[hue_used] if g in hue_groups]
                    n_groups = len(hue_groups)
                    
                    for hi_, hgrp in enumerate(hue_groups):
                        sub_hg = plot_data[plot_data[hue_used] == hgrp] if hgrp is not None else plot_data
                        vals = sub_hg[num_var].dropna()
                        if len(vals) < 2: continue
                        m = vals.mean()
                        se = _stats.sem(vals)
                        z = _stats.norm.ppf(1 - (1 - ci_level_value) / 2)
                        err = z * se
                        pos = 0 if n_groups == 1 else (hi_ - (n_groups - 1) / 2) * (0.8 / n_groups)
                        label = str(hgrp) if hgrp is not None else num_var

                        if orientation == 'v':
                            fig.add_trace(go.Scatter(x=[pos], y=[m], mode='markers', marker=dict(size=12, color='crimson', symbol='circle', line=dict(width=2, color='white')), error_y=dict(type='data', array=[err], color='crimson', thickness=3, width=8), name=f"{label} mean", legendgroup=label, showlegend=True))
                        else:
                            fig.add_trace(go.Scatter(x=[m], y=[pos], mode='markers', marker=dict(size=12, color='crimson', symbol='circle', line=dict(width=2, color='white')), error_x=dict(type='data', array=[err], color='crimson', thickness=3, width=8), name=f"{label} mean", legendgroup=label, showlegend=True))

                fig.update_layout(width=400 if (single_var and orientation == 'v') else 800, height=300 if (single_var and orientation == 'h') else 600, boxmode='group' if hue_used else 'overlay', violinmode='group' if hue_used else 'overlay')
                if single_var:
                    if orientation == 'v': fig.update_xaxes(showticklabels=False, title_text="", range=[-0.5, 0.5])
                    else: fig.update_yaxes(showticklabels=False, title_text="", range=[-0.5, 0.5])

                if facet_col_used or facet_row_used:
                    plot_kwargs = dict(data_frame=plot_data, color_discrete_sequence=PALETTE, width=800, height=600, category_orders=category_orders)
                    if var_y_used: plot_kwargs['y'] = var_y_used
                    if var_x_used: plot_kwargs['x'] = var_x_used
                    if hue_used: plot_kwargs['color'] = hue_used
                    if facet_col_used: plot_kwargs['facet_col'] = facet_col_used
                    if facet_row_used: plot_kwargs['facet_row'] = facet_row_used
                    fig = px.box(**plot_kwargs) if tplot == "boxplot" else px.violin(**plot_kwargs, box=True)
                    if tplot == "violin":
                        for trace in fig.data:
                            ys = trace.y if orientation == 'v' else trace.x
                            if ys is not None and len(ys) > 0: trace.update(spanmode='manual', span=[min(ys), max(ys)])

                st.plotly_chart(fig, use_container_width=(not single_var))

            elif plot_type == 'lineplot':
                var_x = global_var_x
                var_y = global_var_y
                hue_used = global_hue
                facet_col_used = global_facet_col
                facet_row_used = global_facet_row

                with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                    style_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                    width_cols = ['None'] + (data.columns.tolist() if risk_it_all else numeric_cols)
                    
                    style = st.selectbox("Line Style (Categorical)", style_cols, index=0)
                    width = st.selectbox("Line Width (Numeric)", width_cols, index=0)
                    show_markers = st.checkbox("Show Markers", value=True)
                    band_interval = st.checkbox("Show Band in Lineplot (CI)", value=False)
                    time_series = st.checkbox("Time is here", value=False)
                    aggregation_method = st.selectbox("Aggregation Method", ["Mean", "Sum"], index=0) if not time_series else "Raw"
                    ci_level = st.selectbox("Confidence Interval Level", ["68%", "95%", "99%"], index=1) if (band_interval and not time_series) else "95%"
                    
                    style_used = None if style == 'None' else style
                    width_used = None if width == 'None' else width
                    ci_level_value = {"68%": 0.68, "95%": 0.95, "99%": 0.99}.get(ci_level, 0.95)
                    
                    plot_data = data.copy()
                    if hue_used is not None and hue_used in plot_data.columns: plot_data[hue_used] = plot_data[hue_used].astype(str)

                    category_orders = {}
                    for var_name, prefix in [(var_x, "x"), (hue_used, "hue"), (facet_col_used, "col"), (facet_row_used, "row")]:
                        if var_name and var_name in plot_data.columns and plot_data[var_name].dtype.name in ['object', 'category']:
                            c_order = st.multiselect(f"Custom Order for {prefix}: {var_name}", options=plot_data[var_name].dropna().unique().tolist(), default=sorted(plot_data[var_name].dropna().unique().tolist()))
                            plot_data[var_name] = pd.Categorical(plot_data[var_name], categories=c_order, ordered=True)
                            category_orders[var_name] = c_order

                facet_col_vals = plot_data[facet_col_used].dropna().unique().tolist() if facet_col_used else [None]
                facet_row_vals = plot_data[facet_row_used].dropna().unique().tolist() if facet_row_used else [None]
                n_cols = len(facet_col_vals) if facet_col_used else 1
                n_rows = len(facet_row_vals) if facet_row_used else 1

                fig = make_subplots(
                    rows=n_rows, cols=n_cols, shared_yaxes=True, shared_xaxes=True, horizontal_spacing=0.1, vertical_spacing=0.12,
                    subplot_titles=[
                        f"{facet_row_used}: {r} | {facet_col_used}: {c}" if facet_row_used and facet_col_used else
                        f"{facet_col_used}: {c}" if facet_col_used else f"{facet_row_used}: {r}" if facet_row_used else ""
                        for r in facet_row_vals for c in facet_col_vals
                    ]
                )

                hue_values = plot_data[hue_used].dropna().unique() if (hue_used and hue_used != var_x) else [None]
                color_map = {hv: PALETTE[idx % len(PALETTE)] for idx, hv in enumerate(hue_values)}

                line_styles = {'solid': 'solid', 'dash': 'dash', 'dot': 'dot', 'dashdot': 'dashdot'}
                if style_used and style_used in plot_data.columns and plot_data[style_used].dtype.name in ['object', 'category']:
                    style_values = plot_data[style_used].dropna().unique()
                    style_map = {v: list(line_styles.values())[i % len(line_styles)] for i, v in enumerate(style_values)}
                else:
                    style_map = {None: 'solid'}

                if width_used and width_used in plot_data.columns and np.issubdtype(plot_data[width_used].dtype, np.number):
                    width_map = plot_data[width_used].dropna().to_dict()
                else: width_map = {None: 2}

                seen_hue_values = set()

                for i_row, row_val in enumerate(facet_row_vals):
                    for i_col, col_val in enumerate(facet_col_vals):
                        row_idx, col_idx = i_row + 1, i_col + 1
                        sub_data = plot_data.copy()
                        if facet_row_used: sub_data = sub_data[sub_data[facet_row_used] == row_val]
                        if facet_col_used: sub_data = sub_data[sub_data[facet_col_used] == col_val]
                        if sub_data.empty: continue

                        for hue_val in hue_values:
                            subset = sub_data.copy()
                            if hue_used and hue_used != var_x: subset = subset[subset[hue_used] == hue_val]
                            if subset.empty: continue
                            
                            line_color = color_map[hue_val]
                            show_in_legend = hue_val not in seen_hue_values
                            if show_in_legend: seen_hue_values.add(hue_val)

                            if time_series:
                                is_object = False
                                if isinstance(plot_data[var_x].dtype, pd.CategoricalDtype):
                                    if plot_data[var_x].cat.categories.dtype.name == 'object': is_object = True
                                else:
                                    if plot_data[var_x].dtype.name == 'object': is_object = True

                                if not is_object:
                                    st.warning(f"Time series requires '{var_x}' to be an object (dates).")
                                    continue
                                if not np.issubdtype(plot_data[var_y].dtype, np.number):
                                    st.warning(f"Time series requires '{var_y}' to be numeric.")
                                    continue

                                scatter_kwargs = dict(
                                    x=subset[var_x], y=subset[var_y], mode='lines' + ('+markers' if show_markers else ''),
                                    name=str(hue_val) if hue_val else "Line", legendgroup=str(hue_val) if hue_val else None,
                                    marker=dict(size=6, color=line_color),
                                    line=dict(color=line_color, dash=style_map.get(subset[style_used].iloc[0] if style_used else None, 'solid'), width=width_map.get(subset[width_used].iloc[0] if width_used else None, 2)),
                                    showlegend=show_in_legend
                                )
                            else:
                                group_cols = list(dict.fromkeys([col for col in [var_x, hue_used, facet_col_used, facet_row_used] if col is not None and col != var_x]))
                                group_cols = [var_x] + group_cols
                                grouped = plot_data.groupby(group_cols)[var_y].agg(['mean', 'std', 'count', 'sum']).reset_index()

                                sub = grouped.copy()
                                if hue_used and hue_used != var_x: sub = sub[sub[hue_used] == hue_val]
                                if facet_row_used: sub = sub[sub[facet_row_used] == row_val]
                                if facet_col_used: sub = sub[sub[facet_col_used] == col_val]

                                if sub.empty or sub['count'].iloc[0] < 2: continue

                                sub['se'] = sub['std'] / np.sqrt(sub['count'])
                                sub['ci_lower'], sub['ci_upper'] = t.interval(ci_level_value, sub['count'] - 1, loc=sub['mean'], scale=sub['se'])
                                sub['ci'] = (sub['ci_upper'] - sub['ci_lower']) / 2
                                y_values = sub['sum'] if aggregation_method == "Sum" else sub['mean']

                                scatter_kwargs = dict(
                                    x=sub[var_x], y=y_values, mode='lines' + ('+markers' if show_markers else ''),
                                    name=str(hue_val) if hue_val else "Line", legendgroup=str(hue_val) if hue_val else None,
                                    marker=dict(size=6, color=line_color),
                                    line=dict(color=line_color, dash=style_map.get(sub[style_used].iloc[0] if style_used else None, 'solid'), width=width_map.get(sub[width_used].iloc[0] if width_used else None, 2)),
                                    showlegend=show_in_legend
                                )
                                if band_interval and aggregation_method == "Mean":
                                    scatter_kwargs['error_y'] = dict(type='data', symmetric=True, array=sub['ci'], thickness=1.5, width=5, color=line_color)

                            fig.add_trace(go.Scatter(**scatter_kwargs), row=row_idx, col=col_idx)

                fig.update_layout(width=1000, height=400 * n_rows, title=f"{'Time Series ' if time_series else ''}Lineplot: {aggregation_method} of {var_y} vs {var_x}", showlegend=True, legend_title=hue_used)
                if facet_row_used or facet_col_used:
                    for i in range(1, n_rows + 1):
                        for j in range(1, n_cols + 1):
                            fig.update_xaxes(title_text=var_x, row=i, col=j, showticklabels=True)
                            fig.update_yaxes(title_text=f"{aggregation_method} of {var_y}", row=i, col=j, showticklabels=True)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == 'ridges':
                var_x = global_var_x
                var_y = global_var_y
                hue = global_hue

                cat_cols = data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
                only_numeric = len(cat_cols) == 0

                with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                    height = st.slider("Height (each row)", 5, 500, 200, 5)
                    
                    plot_data = data.copy()
                    global_hue_categories = [None]
                    y_values_unique = []
                    hue_param = hue

                    if not only_numeric:
                        if var_y:
                            y_values_unique = plot_data[var_y].dropna().unique()
                            n_rows = len(y_values_unique)
                            if n_rows > 100:
                                st.warning(f"Too many categories in {var_y} ({n_rows}). Plotting numeric columns instead.")
                                only_numeric = True
                            else:
                                no_hue = st.checkbox("No Hue?", value=False)
                                hue_param = var_y if no_hue else hue

                                custom_order_y = st.multiselect(f"Custom Order for Y: {var_y}", options=sorted(y_values_unique.tolist()), default=sorted(y_values_unique.tolist()))
                                plot_data[var_y] = pd.Categorical(plot_data[var_y], categories=custom_order_y, ordered=True)
                                y_values_unique = custom_order_y

                                if hue_param != var_y and hue_param in plot_data.columns:
                                    hue_values_unique = plot_data[hue_param].dropna().unique()
                                    custom_order_hue = st.multiselect(f"Custom Order for Hue: {hue_param}", options=sorted(hue_values_unique.tolist()), default=sorted(hue_values_unique.tolist()))
                                    plot_data[hue_param] = pd.Categorical(plot_data[hue_param], categories=custom_order_hue, ordered=True)
                                    global_hue_categories = custom_order_hue

                if only_numeric:
                    with st.sidebar.expander("⚙️ Numeric Columns Config", expanded=True):
                        selected_columns = st.multiselect("Select Numeric Columns for Ridge Plot", options=plot_data.select_dtypes(include=['number']).columns.tolist(), default=plot_data.select_dtypes(include=['number']).columns.tolist())
                    
                    n_rows = len(selected_columns)
                    if n_rows == 0:
                        st.warning("Please select at least one numeric variable.")
                        st.stop()

                    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_titles=[str(c) for c in selected_columns])
                    for idx, col in enumerate(selected_columns):
                        x_data = plot_data[col].dropna()
                        if len(x_data) < 2: continue
                        kde = gaussian_kde(x_data)
                        x_values = np.linspace(x_data.min(), x_data.max(), 200)
                        density = kde(x_values)
                        fig.add_trace(go.Scatter(x=x_values, y=density / density.max(), mode='lines', fill='tozeroy', name=col, line=dict(color=PALETTE[idx % len(PALETTE)]), showlegend=False), row=idx+1, col=1)
                        fig.update_yaxes(title="Density", row=idx+1, col=1)

                    fig.update_layout(title="Ridge Plot of Selected Numeric Variables", width=900, height=max(300, n_rows * height), showlegend=False)
                    fig.update_xaxes(title="Value", row=n_rows, col=1)

                else:
                    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_titles=[str(y) for y in y_values_unique])
                    x_range = [plot_data[var_x].dropna().min(), plot_data[var_x].dropna().max()]
                    x_values = np.linspace(x_range[0], x_range[1], 200)

                    for idx, y_val in enumerate(y_values_unique):
                        row_idx = idx + 1
                        subset = plot_data[plot_data[var_y] == y_val]
                        if subset.empty: continue
                        hue_values = subset[hue_param].dropna().unique() if (hue_param != var_y and hue_param in subset.columns) else [None]

                        for j, hue_val in enumerate(hue_values):
                            hue_subset = subset[subset[hue_param] == hue_val] if hue_val is not None else subset
                            x_data = hue_subset[var_x].dropna()
                            if len(x_data) < 2: continue
                            kde = gaussian_kde(x_data)
                            density = kde(x_values)
                            color_idx = global_hue_categories.index(hue_val) if hue_val in global_hue_categories else 0

                            fig.add_trace(go.Scatter(x=x_values, y=density / density.max(), mode='lines', fill='tozeroy', name=f"{hue_val}" if hue_val is not None else str(y_val), line=dict(color=PALETTE[color_idx % len(PALETTE)]), showlegend=(row_idx == 1)), row=row_idx, col=1)
                        fig.update_yaxes(title="Density", row=row_idx, col=1)

                    fig.update_layout(title=f"Ridge Plot Faceted by {var_y}", width=900, height=max(300, n_rows * height), showlegend=True)
                    fig.update_xaxes(title=var_x, row=n_rows, col=1)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == 'histogram':
                var_x = global_var_x
                hue_var = global_hue if global_hue else 'None'
                facet_col = global_facet_col if global_facet_col else 'None'
                facet_row = global_facet_row if global_facet_row else 'None'

                with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                    multiple = st.selectbox("Multiple", ["layer", "dodge", "stack"])
                    stat = st.selectbox("Stat", ["count", "probability", "percent", "density"])
                    element = st.selectbox("Element", ["bars", "step"])
                    common_norm = st.checkbox("Common Norm", value=False)
                    cumulative = st.checkbox("Cumulative", value=False)
                    
                    barmode = 'overlay' if multiple == 'layer' else multiple
                    plot_data = data.copy()
                    if hue_var != "None" and hue_var in plot_data.columns: plot_data[hue_var] = plot_data[hue_var].astype(str)

                    category_orders = {}
                    for v_name, label in [(var_x, 'X'), (hue_var, 'Hue'), (facet_col, 'Col'), (facet_row, 'Row')]:
                        if v_name != 'None' and v_name in plot_data.columns and plot_data[v_name].dtype.name in ['object', 'category']:
                            c_order = st.multiselect(f"Custom Order for {label}: {v_name}", options=plot_data[v_name].dropna().unique().tolist(), default=sorted(plot_data[v_name].dropna().unique().tolist()), key=f"hist_ord_{label}")
                            plot_data[v_name] = pd.Categorical(plot_data[v_name], categories=c_order, ordered=True)
                            category_orders[v_name] = c_order

                if element == "step":
                    fig = go.Figure()
                    for hue_val in plot_data[hue_var].dropna().unique() if hue_var != "None" else [None]:
                        subset = plot_data[plot_data[hue_var] == hue_val] if hue_var != "None" else plot_data
                        fig.add_trace(go.Histogram(x=subset[var_x], histnorm=stat if stat != "count" else None, cumulative_enabled=cumulative, opacity=0.75 if multiple == 'layer' else 1.0, name=str(hue_val) if hue_val is not None else var_x))
                    fig.update_layout(barmode=barmode, width=800, height=600, title="Step Histogram", xaxis_title=var_x, yaxis_title=stat.capitalize(), showlegend=True)
                else:
                    fig = px.histogram(
                        plot_data, x=var_x, color=hue_var if hue_var != "None" else None,
                        facet_col=facet_col if facet_col != "None" else None, facet_row=facet_row if facet_row != "None" else None,
                        barmode=barmode, histnorm=stat if stat != "count" else None, cumulative=cumulative,
                        category_orders=category_orders, color_discrete_sequence=PALETTE, width=800, height=600
                    )
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type in ["density 1", "density 2"]:
                var_x = global_var_x
                var_y = global_var_y if plot_type == "density 2" else None
                hue_var = global_hue if global_hue else 'None'
                facet_col = global_facet_col if global_facet_col else 'None'
                facet_row = global_facet_row if global_facet_row else 'None'

                plot_data = data.copy()
                category_orders = {}

                with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                    if plot_type == "density 1":
                        multiple = st.selectbox("Multiple", ["layer", "stack"])
                        common_norm = st.checkbox("Common Norm (Normalize)", value=False)
                        cumulative = st.checkbox("Cumulative Density", value=False)
                        show_scatter = False
                        kde_style = "Filled"
                        gradient_alpha = False
                    else:
                        kind = st.selectbox("Kind", ["hist", "kde"])
                        common_norm = st.checkbox("Common Norm (Normalize)", value=False)
                        rug = st.checkbox("Add Rug Plot?", value=False)
                        
                        show_scatter = st.checkbox("Show Scatter Plot Overlay", value=False) if kind == "kde" else False
                        #kde_style = st.selectbox("2D KDE Style", ["Filled (Gradient Alpha)", "Contour Lines Only"]) if kind == "kde" else "Filled (Gradient Alpha)"
                        kde_style = "Filled (Gradient Alpha)"
                        gradient_alpha = st.checkbox("Gradient Transparency (Density-based)", value=False) if (kind == "kde" and kde_style == "Filled (Gradient Alpha)") else False


                    def add_category_order(var_name, prefix):
                        if var_name and var_name != 'None' and plot_data[var_name].dtype.name in ['object', 'category']:
                            c_order = st.multiselect(f"Custom Order for {prefix}: {var_name}", options=plot_data[var_name].dropna().unique().tolist(), default=sorted(plot_data[var_name].dropna().unique().tolist()), key=f"den_ord_{prefix}")
                            plot_data[var_name] = pd.Categorical(plot_data[var_name], categories=c_order, ordered=True)
                            category_orders[var_name] = c_order
                    
                    add_category_order(var_x, 'x')
                    if var_y: add_category_order(var_y, 'y')
                    add_category_order(hue_var, 'hue')
                    add_category_order(facet_col, 'col')
                    add_category_order(facet_row, 'row')

                if plot_type == "density 1":
                    facet_active = (facet_col != 'None') or (facet_row != 'None')
                    if facet_active:
                        facets = [col for col in [facet_row, facet_col] if col != 'None']
                        if facets:
                            filtered_plot_data = plot_data.copy()
                            for facet in facets:
                                if facet in category_orders and category_orders[facet]:
                                    filtered_plot_data = filtered_plot_data[filtered_plot_data[facet].isin(category_orders[facet])]
                                filtered_plot_data = filtered_plot_data.dropna(subset=[facet])

                            if not filtered_plot_data.empty:
                                facet_combinations = filtered_plot_data[facets].drop_duplicates()
                                for facet in facets:
                                    if facet in category_orders:
                                        facet_combinations[facet] = pd.Categorical(facet_combinations[facet], categories=category_orders[facet], ordered=True)
                                facet_combinations = facet_combinations.sort_values(by=facets)
                                n_rows = facet_combinations[facet_row].nunique() if facet_row != 'None' else 1
                                n_cols = facet_combinations[facet_col].nunique() if facet_col != 'None' else 1
                            else:
                                facet_combinations = pd.DataFrame([{}])
                                n_rows, n_cols, facet_active = 1, 1, False
                        else:
                            facet_combinations = pd.DataFrame([{}])
                            n_rows, n_cols, facet_active = 1, 1, False
                    else: n_rows, n_cols = 1, 1

                    subplot_titles = []
                    if facet_active and facets:
                        for _, row in facet_combinations.iterrows():
                            title = []
                            if facet_row != 'None': title.append(f"{facet_row}: {row[facet_row]}")
                            if facet_col != 'None': title.append(f"{facet_col}: {row[facet_col]}")
                            subplot_titles.append(" | ".join(title))

                    if facet_active and subplot_titles:
                        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, horizontal_spacing=0.08, vertical_spacing=0.1, shared_xaxes=True)
                    else:
                        fig = make_subplots(rows=1, cols=1, horizontal_spacing=0.08, vertical_spacing=0.1, shared_xaxes=True)
                        n_rows, n_cols = 1, 1

                    x_range = np.linspace(plot_data[var_x].dropna().min(), plot_data[var_x].dropna().max(), 200)

                    if facet_active and facets and not facet_combinations.empty and not facet_combinations.equals(pd.DataFrame([{}])):
                        facet_groups = [(tuple(row[facets]), filtered_plot_data[(filtered_plot_data[facets].eq(row[facets]).all(axis=1)) if len(facets) > 1 else (filtered_plot_data[facets[0]] == row[facets[0]])]) for _, row in facet_combinations.iterrows()]
                    else:
                        facet_groups = [((), plot_data)]

                    for idx, (facet_vals, group_df) in enumerate(facet_groups):
                        if facet_active and facets:
                            if group_df.empty: continue
                            facet_vals = list(facet_vals) if isinstance(facet_vals, tuple) else [facet_vals]
                            row_idx = list(facet_combinations[facet_row].unique()).index(facet_vals[0]) + 1 if facet_row != 'None' else 1
                            col_idx = list(facet_combinations[facet_col].unique()).index(facet_vals[-1]) + 1 if facet_col != 'None' else 1
                        else: row_idx, col_idx = 1, 1

                        hue_values = group_df[hue_var].dropna().unique() if hue_var != 'None' else [None]
                        if hue_var != 'None' and hue_var in category_orders: hue_values = [v for v in category_orders[hue_var] if v in hue_values]

                        offset = 0
                        for j, hue_val in enumerate(hue_values):
                            subset = group_df[group_df[hue_var] == hue_val] if hue_var != 'None' else group_df
                            x_data = subset[var_x].dropna()
                            if len(x_data) < 2: continue
                            try:
                                kde = gaussian_kde(x_data)
                                density = kde(x_range)
                                if cumulative:
                                    density = np.cumsum(density)
                                    density = density / density[-1]
                                if common_norm and not cumulative: density = density / density.max()
                                y_values = density + offset if multiple == "stack" else density

                                fig.add_trace(
                                    go.Scatter(x=x_range, y=y_values, mode='lines', fill='tozeroy' if multiple == "stack" else None, name=f"{hue_val}" if hue_var != 'None' else str(var_x), line=dict(color=PALETTE[j % len(PALETTE)]), showlegend=(idx == 0)),
                                    row=row_idx, col=col_idx
                                )
                                if multiple == "stack": offset += y_values.max()
                            except Exception: continue

                    fig.update_layout(title=f"Density Plot of {var_x}", width=900, height=300 * max(n_rows, 1), showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    if kind == "hist":
                        fig = px.density_heatmap(
                            plot_data, x=var_x, y=var_y, facet_col=facet_col if facet_col != 'None' else None, facet_row=facet_row if facet_row != 'None' else None,
                            color_continuous_scale='Viridis', histnorm='probability density' if not common_norm else None,
                            marginal_x='rug' if rug else None, marginal_y='rug' if rug else None, width=800, height=600, category_orders=category_orders
                        )
                    else:
                        fig = px.density_contour(
                            plot_data, x=var_x, y=var_y, color=hue_var if hue_var != 'None' else None, facet_col=facet_col if facet_col != 'None' else None, facet_row=facet_row if facet_row != 'None' else None,
                            color_discrete_sequence=PALETTE, marginal_x='rug' if rug else None, marginal_y='rug' if rug else None, width=800, height=600, category_orders=category_orders
                        )
                        
                        # Clean up layout: hide useless colorbar and update contour fills
                        fig.update_layout(coloraxis_showscale=False)
                        
                        if kde_style == "Contour Lines Only":
                            for trace in fig.data:
                                if hasattr(trace, 'contours'):
                                    trace.update(fillcolor=None)
                        elif gradient_alpha:
                            for trace in fig.data:
                                if hasattr(trace, 'contours'):
                                    trace.update(contours=dict(coloring='heatmap'), showscale=False, showlegend=False)

                        # Render clean scatter overlay while respecting category orders and subplots
                        if show_scatter:
                            sub_scatter = plot_data.dropna(subset=[var_x, var_y])
                            if not sub_scatter.empty:
                                scatter_fig = px.scatter(
                                    sub_scatter, x=var_x, y=var_y, color=hue_var if hue_var != 'None' else None,
                                    facet_col=facet_col if facet_col != 'None' else None, facet_row=facet_row if facet_row != 'None' else None,
                                    color_discrete_sequence=PALETTE, category_orders=category_orders
                                )
                                
                                # Map clean density-based opacity per group and prevent duplicate legend entries
                                existing_legend_names = {trace.name for trace in fig.data}
                                
                                for trace in scatter_fig.data:
                                    if 'x' in trace and trace.x is not None and len(trace.x) > 2:
                                        x_vals = np.array(trace.x, dtype=float)
                                        y_vals = np.array(trace.y, dtype=float)
                                        xy_coords = np.vstack([x_vals, y_vals])
                                        try:
                                            pt_density = gaussian_kde(xy_coords)(xy_coords)
                                            pt_alpha = 0.15 + 0.7 * (pt_density - pt_density.min()) / (pt_density.max() - pt_density.min() + 1e-9)
                                        except Exception:
                                            pt_alpha = 0.5
                                        
                                        # Hide legend for scatter if a contour trace with the same name already exists
                                        show_leg = True
                                        if trace.name in existing_legend_names:
                                            show_leg = False
                                        else:
                                            existing_legend_names.add(trace.name)

                                        trace.update(
                                            marker=dict(size=4, opacity=pt_alpha),
                                            showlegend=show_leg
                                        )
                                    fig.add_trace(trace)

                    st.plotly_chart(fig, use_container_width=True)


            elif plot_type == 'scatter':
                var_x = global_var_x
                var_y = global_var_y
                hue = global_hue if global_hue else 'None'
                facet_col = global_facet_col if global_facet_col else 'None'
                facet_row = global_facet_row if global_facet_row else 'None'
                size = global_size if global_size else 'None'
                
                plot_data = data.copy()
                category_orders = {}

                with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                    style_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                    style = st.selectbox("Style (Symbol)", style_cols, index=0)
                    alpha = st.slider("Alpha (Opacity)", 0.0, 1.0, 0.8, 0.01)
                    size_max = st.slider("Max Marker Size", 5, 100, 10, 5)
                    enhance_size = st.selectbox("Enhance Size Differences", options=["None", "Min-Max Normalize"], index=0)

                    plot_3d = st.checkbox("3D Scatter", value=False, help="Add a Z variable and rotate the plot with the mouse.")
                    var_z, show_projections = None, False
                    if plot_3d:
                        z_options = [c for c in (data.columns.tolist() if risk_it_all else numeric_cols) if c not in (var_x, var_y)]
                        if z_options:
                            var_z = st.selectbox("Z Variable", z_options, index=0)
                            show_projections = st.checkbox("Show Projections on Walls", value=False,
                                                           help="Shadow of each point on the XY, XZ and YZ planes.")

                    def add_category_order(var_name, label):
                        if var_name != 'None' and var_name in plot_data.columns and plot_data[var_name].dtype.name in ['object', 'category']:
                            custom_order = st.multiselect(f"Custom Order for {label} ({var_name})", options=plot_data[var_name].dropna().unique().tolist(), default=sorted(plot_data[var_name].dropna().unique().tolist()))
                            if custom_order:
                                plot_data[var_name] = pd.Categorical(plot_data[var_name], categories=custom_order, ordered=True)
                                category_orders[var_name] = custom_order

                    add_category_order(hue, "Hue")
                    add_category_order(facet_col, "Facet Column")
                    add_category_order(facet_row, "Facet Row")

                size_param = None
                if size != 'None':
                    if not pd.api.types.is_numeric_dtype(plot_data[size]):
                        st.warning(f"Size column '{size}' is not numeric. Size parameter will be ignored.")
                    else:
                        if plot_data[size].isna().any():
                            plot_data = plot_data.dropna(subset=[size])
                        if (plot_data[size] < 0).any():
                            plot_data[size] = plot_data[size].abs()

                        if enhance_size == "Min-Max Normalize":
                            size_min = plot_data[size].min()
                            size_max_val = plot_data[size].max()
                            if size_max_val > size_min:
                                plot_data['size_for_plot'] = 10 + 40 * (plot_data[size] - size_min) / (size_max_val - size_min)
                            else:
                                plot_data['size_for_plot'] = 30
                            size_param = 'size_for_plot'
                        else:
                            size_param = size

                scatter_kwargs = dict(data_frame=plot_data, x=var_x, y=var_y, opacity=alpha, color_discrete_sequence=PALETTE, width=800, height=600, category_orders=category_orders)
                if hue != 'None': scatter_kwargs['color'] = hue
                if style != 'None': scatter_kwargs['symbol'] = style
                if facet_col != 'None': scatter_kwargs['facet_col'] = facet_col
                if facet_row != 'None': scatter_kwargs['facet_row'] = facet_row

                if size_param is not None:
                    scatter_kwargs['size'] = size_param
                    scatter_kwargs['size_max'] = size_max
                else:
                    scatter_kwargs['size'] = np.full(len(plot_data), 1)
                    scatter_kwargs['size_max'] = size_max

                if plot_3d:
                    if var_z is None:
                        st.warning("3D scatter needs a third numeric column for Z besides X and Y.")
                        return
                    ignored = [label for label, key in (("Facet Column", "facet_col"), ("Facet Row", "facet_row")) if scatter_kwargs.pop(key, None)]
                    if ignored:
                        st.info(f"{' and '.join(ignored)} not available in 3D; showing all data in one plot.")
                    scatter_kwargs.update(z=var_z, height=700)
                    fig = px.scatter_3d(**scatter_kwargs)
                    if show_projections:
                        fig.update_traces(projection=dict(x=dict(show=True, opacity=0.3), y=dict(show=True, opacity=0.3), z=dict(show=True, opacity=0.3)))
                    fig.update_layout(scene=dict(xaxis_title=var_x, yaxis_title=var_y, zaxis_title=var_z), margin=dict(l=0, r=0, t=40, b=0))
                else:
                    fig = px.scatter(**scatter_kwargs)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == 'catplot':
                var_x = global_var_x
                var_y = global_var_y
                hue = global_hue if global_hue else 'None'
                col = global_facet_col if global_facet_col else 'None'
                row = global_facet_row if global_facet_row else 'None'
                size_var = global_size if global_size else 'None'

                plot_data = data.copy()
                category_orders = {}

                with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                    kind = st.selectbox("Kind", ["strip", "swarm"])
                    global_point_size = st.slider("Global Marker Size", 5, 50, 10, 5)

                    for v_name, label in [(var_x, 'X Variable'), (var_y, 'Y Variable'), (hue, 'Hue'), (col, 'Facet Column'), (row, 'Facet Row')]:
                        if v_name != 'None' and v_name in plot_data.columns and plot_data[v_name].dtype.name in ['object', 'category']:
                            c_order = st.multiselect(f"Custom Order for {label} ({v_name})", options=plot_data[v_name].dropna().unique().tolist(), default=sorted(plot_data[v_name].dropna().unique().tolist()))
                            if c_order:
                                plot_data[v_name] = pd.Categorical(plot_data[v_name], categories=c_order, ordered=True)
                                category_orders[v_name] = c_order

                if col != 'None' or row != 'None':
                    col_vals = plot_data[col].dropna().unique() if col != 'None' else [None]
                    row_vals = plot_data[row].dropna().unique() if row != 'None' else [None]
                    n_cols, n_rows = max(len(col_vals), 1), max(len(row_vals), 1)

                    fig = make_subplots(
                        rows=n_rows, cols=n_cols,
                        subplot_titles=[f"{row}: {r} | {col}: {c}" if row != 'None' and col != 'None' else f"{col}: {c}" if col != 'None' else f"{row}: {r}" if row != 'None' else "" for r in row_vals for c in col_vals],
                        shared_yaxes=True, shared_xaxes=True, horizontal_spacing=0.1, vertical_spacing=0.12
                    )

                    hue_values = plot_data[hue].dropna().unique() if hue != 'None' else [None]
                    color_map = {hv: PALETTE[idx % len(PALETTE)] for idx, hv in enumerate(hue_values)}

                    for i_row, row_val in enumerate(row_vals):
                        for i_col, col_val in enumerate(col_vals):
                            row_idx, col_idx = i_row + 1, i_col + 1
                            sub_data = plot_data.copy()
                            if row != 'None': sub_data = sub_data[sub_data[row] == row_val]
                            if col != 'None': sub_data = sub_data[sub_data[col] == col_val]
                            if sub_data.empty: continue

                            for hue_val in hue_values:
                                subset = sub_data.copy()
                                if hue_val is not None: subset = subset[subset[hue] == hue_val]
                                if subset.empty: continue
                                marker_color = color_map[hue_val]

                                scatter = go.Scatter(
                                    x=subset[var_x], y=subset[var_y], mode='markers', name=str(hue_val) if hue_val else "Points",
                                    marker=dict(
                                        color=marker_color,
                                        size=global_point_size if size_var == 'None' or not pd.api.types.is_numeric_dtype(plot_data[size_var]) else subset[size_var],
                                        sizemode='diameter', sizeref=2.*subset[size_var].max()/(15.**2) if size_var != 'None' and pd.api.types.is_numeric_dtype(plot_data[size_var]) else None,
                                        sizemin=4 if size_var != 'None' and pd.api.types.is_numeric_dtype(plot_data[size_var]) else None
                                    ),
                                    showlegend=(row_idx == 1 and col_idx == 1), legendgroup=str(hue_val) if hue_val else None
                                )

                                if kind == "swarm":
                                    scatter.update(marker=dict(sizeref=2.*subset[size_var].max()/(15.**2) if size_var != 'None' else None))
                                    scatter.update(x=jittered_values(subset[var_x], 0.3))

                                fig.add_trace(scatter, row=row_idx, col=col_idx)

                    fig.update_layout(width=800, height=400 * n_rows, title="Catplot with Faceting", showlegend=True, legend_title=hue if hue != 'None' else None)

                else:
                    fig = px.strip(plot_data, x=var_x, y=var_y, color=hue if hue != 'None' else None, color_discrete_sequence=PALETTE, category_orders=category_orders, width=800, height=600)
                    if size_var != 'None' and pd.api.types.is_numeric_dtype(plot_data[size_var]):
                        sizes = plot_data[size_var]
                        fig.update_traces(marker=dict(size=sizes, sizemode='diameter', sizeref=2.*max(sizes)/(15.**2), sizemin=4))
                    else: fig.update_traces(marker=dict(size=global_point_size))
                    if kind == "swarm": fig.update_traces(jitter=0.3)

                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == 'regression':
                var_x = global_var_x
                var_y = global_var_y
                hue = global_hue if global_hue else 'None'

                with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                    methods = st.multiselect("Fit Methods", list(FIT_METHODS), default=["Polynomial"],
                                             help="Polynomial: OLS fit with confidence band. LOWESS: locally weighted smoothing (no band).")
                    order, ci_level, lowess_frac = 1, 95, 0.66
                    if "Polynomial" in methods:
                        order = st.selectbox("Order of Polynomial Fit", [1, 2, 3], index=0)
                        ci_level = st.selectbox("Confidence Interval (%)", [68, 90, 95, 99], index=2)
                    if "LOWESS" in methods:
                        lowess_frac = st.slider("LOWESS Fraction (smoothing span)", 0.05, 1.0, 0.66, 0.01,
                                                help="Share of points used for each local fit; larger is smoother.")
                    marginal_options = ["None", *MARGINAL_KINDS]
                    marginal_x = st.selectbox("Marginal Plot for X", marginal_options, index=0)
                    marginal_y = st.selectbox("Marginal Plot for Y", ["Same as X", *marginal_options], index=0,
                                              help="Defaults to the X choice so both axes get a marginal; pick another kind or None to change it.")
                    if marginal_y == "Same as X":
                        marginal_y = marginal_x
                    use_hue = st.checkbox("Use Hue for Fits?", value=True)

                try:
                    fig = regression_figure(
                        data, var_x, var_y,
                        hue=hue if use_hue and hue != 'None' else None,
                        methods=methods, order=order, ci_level=ci_level, lowess_frac=lowess_frac,
                        marginal_x=None if marginal_x == "None" else marginal_x,
                        marginal_y=None if marginal_y == "None" else marginal_y,
                        palette=PALETTE,
                    )
                except ValueError as exc:
                    st.warning(str(exc))
                    return
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == 'heatmap':
                cat_var = global_var_x if global_var_x != 'None' else None
                num_var = global_var_y
                plot_data = data.copy()

                with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                    colormap = st.selectbox("Select Colormap", ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Blues", "Reds", "Greens", "RdBu_r"], index=0)
                    if cat_var is not None and cat_var in plot_data.columns:
                        custom_order = st.multiselect(f"Custom Order for {cat_var}", options=plot_data[cat_var].dropna().unique().tolist(), default=sorted(plot_data[cat_var].dropna().unique().tolist()))
                        if custom_order: plot_data[cat_var] = pd.Categorical(plot_data[cat_var], categories=custom_order, ordered=True)
                    else:
                        heatmap_numeric = [c for c in numeric_cols if c in plot_data.columns]
                        heatmap_cols = st.multiselect("Numeric Columns", heatmap_numeric, default=heatmap_numeric,
                                                      help="Non-numeric columns are ignored.")
                        scaling = st.selectbox("Scale Each Column", list(HEATMAP_SCALINGS), index=1,
                                               help="Z-score / Min-max make columns with different units comparable. Hover always shows the original value.")
                        row_order = st.selectbox("Observation Order", list(HEATMAP_ROW_ORDERS), index=0,
                                                 help="Cluster groups similar observations together.")
                        sort_by, descending = None, False
                        if row_order == "Sort by column":
                            sort_by = st.selectbox("Sort Observations By", plot_data.columns.tolist())
                            descending = st.checkbox("Descending", value=False)
                        cluster_columns = st.checkbox("Cluster Columns", value=False, help="Place similar variables next to each other.")
                        variables_on = st.radio("Variables On", ["Y axis", "X axis"], horizontal=True)

                if cat_var is None:
                    try:
                        fig = numeric_heatmap_figure(
                            plot_data, columns=heatmap_cols, scaling=scaling, row_order=row_order,
                            sort_by=sort_by, descending=descending, cluster_columns=cluster_columns,
                            variables_on="y" if variables_on == "Y axis" else "x", colorscale=colormap,
                        )
                    except ValueError as exc:
                        st.warning(str(exc))
                        return
                    st.plotly_chart(fig, use_container_width=True)
                    return

                if num_var is None or not pd.api.types.is_numeric_dtype(plot_data[num_var]):
                    st.warning(f"Please select a valid numerical Y variable for Heatmap mode.")
                    return

                is_categorical = False
                if isinstance(plot_data[cat_var].dtype, pd.CategoricalDtype): is_categorical = True
                elif plot_data[cat_var].dtype.name == 'object': is_categorical = True
                if not is_categorical:
                    st.warning(f"Selected X variable '{cat_var}' must be categorical or object type.")
                    return

                plot_data['obs_id'] = [f"Obs {i}" for i in range(len(plot_data))]
                heatmap_data = plot_data.pivot(columns='obs_id', values=num_var, index=cat_var)
                x_labels = heatmap_data.columns.tolist()
                y_labels = heatmap_data.index.tolist()
                title = f"Heatmap Numerical: {cat_var} vs Observations"
                height = max(600, len(y_labels) * 20)

                fig = px.imshow(heatmap_data, labels=dict(x="Observations", y=cat_var, color="Value"), x=x_labels, y=y_labels, color_continuous_scale=colormap, title=title, width=800, height=height, aspect='auto')
                fig.update_layout(margin=dict(l=50, r=50, t=100, b=50), yaxis=dict(tickfont=dict(size=10)))
                st.plotly_chart(fig, use_container_width=True)

    if not data.empty:
        render_plot()
        st.markdown(
            """
            <hr style="margin-top:2em;margin-bottom:1em;">
            <div style="text-align: center; color: gray; font-size: 0.9em;">
            Explore. Understand. Inspire. 🚀 | Powered by Streamlit + Plotly.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("The dataset is empty after preprocessing. Please check your data or selections in the Home section.")

def jittered_values(series, jitter_amount=0.3):
    if pd.api.types.is_numeric_dtype(series):
        return series + np.random.uniform(-jitter_amount, jitter_amount, size=len(series))
    else:
        codes = pd.Categorical(series).codes
        return codes + np.random.uniform(-jitter_amount, jitter_amount, size=len(series))