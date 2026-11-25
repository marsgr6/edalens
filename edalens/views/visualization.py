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
import itertools
from scipy.stats import t

from edalens.utils.data_processing import parse_dates
from edalens.utils.ui import plotly_chart, start_charts
from edalens.utils.plot_utils import (
    ADDITIVE_AGGREGATIONS, BAR_AGGREGATIONS, BAR_ERRORS, BAR_SCALES, DATE_GROUPS,
    CANDLE_SOURCES, FIT_METHODS, HEATMAP_ROW_ORDERS, HEATMAP_SCALINGS, MARGINAL_KINDS,
    bar_figure, bar_summary, candlestick_figure, date_labels, date_values, group_dates, has_weekend_dates,
    LINE_AGGREGATIONS, LINE_INTERVALS, RAW_VALUES, _rgba, line_summary, line_widths,
    fill_steps, histogram_bars, step_histogram_frame, numeric_heatmap_figure, ohlc_frame, regression_figure,
    waterfall_figure, waterfall_frame,
)

warnings.filterwarnings('ignore')

LINEPLOT_ROW_GAP_PX = 110


def is_text_like(values) -> bool:
    """Text or categories: object, category, or pandas' string dtype ("str" in pandas 3)."""
    dtype = values.dtype
    return (
        pd.api.types.is_object_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(dtype)
    )


def text_columns(frame, include_bool=False) -> list:
    """Text-like columns (see is_text_like), plus boolean ones on request."""
    return [
        c for c in frame.columns
        if is_text_like(frame[c]) or (include_bool and pd.api.types.is_bool_dtype(frame[c].dtype))
    ]


_CALENDARS = [
    [m.lower() for m in ("January", "February", "March", "April", "May", "June", "July",
                         "August", "September", "October", "November", "December")],
    [d.lower() for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")],
]


def natural_sort(values: list) -> list:
    """Month and weekday names in calendar order (Jan, February, mon...); others sorted."""
    names = [str(v).strip().lower().rstrip(".") for v in values]
    for calendar in _CALENDARS:
        rank = {}
        for i, full in enumerate(calendar):
            rank[full] = rank[full[:3]] = i
            if full == "september":
                rank["sept"] = i
        if names and all(n in rank for n in names):
            return [v for _, v in sorted(zip((rank[n] for n in names), values), key=lambda pair: pair[0])]
    try:
        return sorted(values)
    except TypeError:  # Mixed types.
        return sorted(values, key=str)


def levels(values) -> list:
    """Distinct non-missing values: in category order for a categorical, else as they appear."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        present = set(values.dropna().unique())
        return [c for c in values.cat.categories if c in present]
    return list(values.dropna().unique())


def ordered_categories(values, order):
    """``values`` as an ordered categorical in ``order``; values left out of it become missing."""
    return pd.Categorical(values.where(values.isin(order)), categories=list(order), ordered=True)
# Plot types drawn with one panel per Facet Col / Facet Row value.
FACET_PLOTS = ('bars', 'boxes', 'lineplot', 'histogram', 'density 1', 'density 2', 'scatter', 'catplot')

def visualization():
    start_charts()
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
        'pairplot', 'regression', "heatmap", 'ridges', 'candlestick', 'waterfall'
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
            # Bars open as a count plot of the first categorical column (Y = None).
            "x": categorical_cols[0] if categorical_cols else (numeric_cols[0] if numeric_cols else None),
            "y": "None",
            "color": "None",
            "facet_col": "None",
            "facet_row": "None",
            "size": "None"
        }

    if "_plot_type_selector" not in st.session_state:
        st.session_state._plot_type_selector = st.session_state.viz_state["plot_type"]

    def role_options(plot, risk):
        """Columns allowed for X and Y by a plot type ('None' where the role is optional)."""
        if risk:
            x_opts, y_opts = data.columns.tolist(), data.columns.tolist()
        else:
            if plot in ['bars', 'boxes', 'catplot', 'lineplot', 'heatmap', 'waterfall', 'candlestick']:
                x_opts = categorical_cols.copy() if categorical_cols else data.columns.tolist()
                if plot in ('bars', 'lineplot'):  # Datetime columns too: dates can be grouped by month, year...
                    x_opts += [c for c in data.columns if pd.api.types.is_datetime64_any_dtype(data[c]) and c not in x_opts]
            else:  # scatter, histogram, density 1/2, regression, ridges
                x_opts = numeric_cols.copy() if numeric_cols else data.columns.tolist()
            if plot == 'ridges':
                y_opts = categorical_cols.copy() if categorical_cols else data.columns.tolist()
            else:  # Boxes need a numeric Y (the distribution); X is an optional grouping.
                y_opts = numeric_cols.copy() if numeric_cols else data.columns.tolist()
        x_opts = x_opts or data.columns.tolist()
        y_opts = y_opts or data.columns.tolist()
        if plot in ['boxes', 'heatmap']:
            x_opts = ['None'] + [c for c in x_opts if c != 'None']
        if plot == 'bars':  # Y None: count rows per X; a numeric Y: aggregate it.
            y_opts = ['None'] + [c for c in y_opts if c != 'None']
        if plot in ['histogram', 'density 1']:
            y_opts = ['None']
        return x_opts, y_opts

    def remember(role, value):
        """Recent X/Y choices, newest first, so a later plot type can bring them back."""
        memory = st.session_state.setdefault("viz_memory", {"x": [], "y": []})
        if value not in (None, "None"):
            memory[role] = [value] + [v for v in memory[role] if v != value][:9]

    def zone_changed(role, key):
        remember(role, st.session_state[key])  # An explicit pick in the X or Y dropdown.

    def handle_plot_change():
        new_plot = st.session_state.get("_plot_type_selector", st.session_state.viz_state.get("plot_type", "bars"))
        state = st.session_state.viz_state
        state["plot_type"] = new_plot
        # What the outgoing plot showed counts as a choice, unless the last switch filled it in.
        auto = st.session_state.get("viz_auto", {})
        for role in ("x", "y"):
            if state[role] != auto.get(role):
                remember(role, state[role])
        memory = st.session_state["viz_memory"]
        x_opts, y_opts = role_options(new_plot, st.session_state.get("risk_it_all", False))

        def pick(options, candidates, fallback):
            # Recent choices in this role, then in the other role (a numeric Y can become X).
            return next((c for c in candidates if c in options and c != "None"), fallback)

        x_fallback = "None" if new_plot == "heatmap" else next((c for c in x_opts if c != "None"), "None")
        state["x"] = pick(x_opts, [*memory["x"], *memory["y"]], x_fallback)
        if new_plot in ['histogram', 'density 1', 'bars']:
            # No Y (bars start as a count plot; pick a numeric Y to aggregate it).
            state["y"] = "None"
        else:
            # Two-variable plots need a Y other than X when there is one.
            y_candidates = [c for c in [*memory["y"], *memory["x"]] if c != state["x"]]
            y_fallback = next((c for c in y_opts if c not in ("None", state["x"])), y_opts[0])
            state["y"] = pick(y_opts, y_candidates, y_fallback)
        st.session_state["viz_auto"] = {"x": state["x"], "y": state["y"]}

        # The X/Y dropdowns hold their own value (see mapping_zone); move them to the choice.
        st.session_state["x_zone"] = state["x"]
        st.session_state["y_zone"] = state["y"]

    def order_select(label, var, values, sort=True):
        """A "Custom Order" multiselect whose order is kept per column for every plot.

        The order (and any categories removed from it) applies wherever the column
        appears again: another role, another plot type, or after other controls change.
        New categories in the data are appended.
        """
        version = st.session_state.get("data_version", 0)
        if st.session_state.get("category_orders_version") != version:
            st.session_state["category_orders"] = {}
            st.session_state["category_orders_version"] = version
        store = st.session_state["category_orders"]
        present = list(pd.unique(values.dropna()))
        if sort:
            present = natural_sort(present)
        saved = store.get(var)
        if saved:
            default = [v for v in saved["order"] if v in present]
            default += [v for v in present if v not in saved["known"]]
        else:
            default = present
        # A stable key keeps the widget (and an open dropdown) steady while it is edited;
        # a widget that was not shown last run starts again from the saved order.
        chosen = st.multiselect(label, options=present, default=default, key=f"category_order_{version}_{label}")
        if chosen != default:  # Only a user edit updates the shared order.
            known = set(saved["known"]) if saved else set()
            store[var] = {"order": list(chosen), "known": list(known | set(present))}
        return chosen

    def mapping_zone(label, options, key, state_value, role=None):
        # The widget keeps its value in session state; no `index`, which would come from
        # viz_state (one run behind the widget) and could reset a fresh selection.
        if st.session_state.get(key) not in options:
            st.session_state[key] = state_value if state_value in options else options[0]
        st.selectbox(label, options, key=key, label_visibility="collapsed",
                     on_change=zone_changed if role else None, args=(role, key) if role else None)

    def check_cardinality(var_name):
        if var_name and var_name != "None" and var_name in categorical_cols:
            n_unique = data[var_name].nunique()
            if n_unique > 50:
                st.warning(f"⚠️ High cardinality in `{var_name}` ({n_unique} unique values). Rendering may be slow or cluttered.")

    # --- SIDEBAR CONTROLS (Plot Type & Risk) ---
    with st.sidebar:
        st.markdown("### Controls")
        risk_it_all = st.checkbox("Risk It All (Unlock all variable mappings)", value=False, key="risk_it_all")

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
    
    x_opts, y_opts = role_options(current_plot, risk_it_all)

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
        'ridges': '🏔️ ridges',
        'candlestick': '🕯️ candlestick',
        'waterfall': '🌊 waterfall'
    }


    with z1:
        if st.session_state._plot_type_selector not in PLOT_TYPES:
            st.session_state._plot_type_selector = PLOT_TYPES[0]
        st.markdown("**Plot**")
        st.selectbox(
            "Plot", 
            PLOT_TYPES, 
            key="_plot_type_selector", 
            label_visibility="collapsed",
            on_change=handle_plot_change,
            format_func=lambda x: PLOT_ICONS.get(x, x)  # Shows icons without changing the actual value
        )
    with z2:
        st.markdown("**X**")
        mapping_zone("X", x_opts, "x_zone", s_x, role="x")
    with z3:
        st.markdown("**Y**")
        mapping_zone("Y", y_opts, "y_zone", s_y, role="y")
    with z4:
        st.markdown("**Color (Hue)**")
        mapping_zone("Color", cat_opts, "color_zone", s_color)
    with z5:
        st.markdown("**Facet Col**")
        mapping_zone("Facet Col", cat_opts, "facet_col_zone", s_facet_c)
    with z6:
        st.markdown("**Facet Row**")
        mapping_zone("Facet Row", cat_opts, "facet_row_zone", s_facet_r)
    with z7:
        st.markdown("**Size**")
        mapping_zone("Size", size_opts, "size_zone", s_size)

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
    # Candlestick and waterfall name every unused mapping themselves.
    if plot_type not in FACET_PLOTS + ("candlestick", "waterfall") and (global_facet_col or global_facet_row):
        st.info(f"ℹ️ The {plot_type} plot does not use Facet Col or Facet Row.")
    
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
            plotly_chart(fig)

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
            plotly_chart(fig)

        elif plot_type == "pairplot":
            numeric_cols_list = list(data.select_dtypes(include='number').columns)
            cat_cols = text_columns(data, include_bool=True)

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

            plotly_chart(fig)

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
                plotly_chart(fig)

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
                plotly_chart(fig)

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
                plotly_chart(fig)

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
                plotly_chart(fig)

        # Interactive plot types
        else:
            if plot_type == 'bars':
                # Y decides the plot: none -> rows counted per X (a count plot); a numeric
                # column -> that column aggregated per X (e.g. monthly kWh).
                var_x = global_var_x
                var_y = global_var_y
                hue = global_hue
                facet_col = global_facet_col
                facet_row = global_facet_row
                if var_y is not None and not pd.api.types.is_numeric_dtype(data[var_y]):
                    st.warning(f"Bars need a numeric Y to aggregate; '{var_y}' is not numeric. "
                               "Choose a numeric Y, or None to count rows.")
                    return

                with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                    tplot = st.selectbox("Plot Type", ["bars", "heatmap"])
                    if var_y is None:
                        st.caption("Y is None: bars count the rows in each X category. "
                                   "Pick a numeric Y to plot its values instead.")
                        aggregation = "Count"
                    else:
                        aggregation = st.selectbox(
                            f"Aggregate {var_y} per bar", BAR_AGGREGATIONS, index=0,
                            help="Sum suits totals (monthly kWh); with one row per X it is the value itself.")
                    scales = [s for s in BAR_SCALES if s != "Percent within color" or hue]
                    if aggregation in ADDITIVE_AGGREGATIONS:
                        scale = st.selectbox(
                            "Show as", scales, index=0,
                            format_func=lambda s: ("Counts" if var_y is None else "Values") if s == "Values" else s,
                            help="Percentages are taken within each facet panel.")
                    else:
                        scale = "Values"
                    error = st.selectbox("Error bars", BAR_ERRORS, index=0) if aggregation == "Mean" and tplot == "bars" else "None"
                    stacked = st.radio("Bar layout", ["Grouped", "Stacked"], horizontal=True) == "Stacked" if (hue and tplot == "bars") else False
                    show_labels = st.checkbox("Show values on bars", value=True) if tplot == "bars" else True

                    category_orders = {}
                    plot_data = data.copy()
                    if hue is not None and hue != var_x: plot_data[hue] = plot_data[hue].astype(str)
                    # A date X (datetime column, or text that is all dates) is plotted on a time
                    # axis in date order, optionally grouped by week, month, quarter or year.
                    x_dates = date_values(plot_data[var_x])
                    if x_dates is not None and not pd.api.types.is_datetime64_any_dtype(plot_data[var_x]):
                        if not st.checkbox("Day First when X holds dates (e.g., 31/12/2023)", value=True):
                            x_dates = date_values(plot_data[var_x], dayfirst=False)
                    date_group = None
                    if x_dates is not None:
                        date_group = st.selectbox(f"Group dates in {var_x} by", list(DATE_GROUPS), index=0,
                                                  help="Month on daily readings gives monthly totals (Sum) or averages (Mean).")
                        plot_data[var_x] = group_dates(x_dates, date_group)
                    for v_name, label in [(var_x, 'X'), (hue, 'Hue'), (facet_col, 'Col'), (facet_row, 'Row')]:
                        if v_name == var_x and x_dates is not None:
                            continue
                        if v_name and v_name in plot_data.columns and is_text_like(plot_data[v_name]) and v_name not in category_orders:
                            c_order = order_select(f"Custom Order for {label}: {v_name}", v_name, plot_data[v_name])
                            plot_data[v_name] = ordered_categories(plot_data[v_name], c_order)
                            category_orders[v_name] = c_order

                percent = scale != "Values"
                summary = bar_summary(plot_data, var_x, var_y, hue, (facet_col, facet_row), aggregation, scale, error)
                if summary.empty:
                    st.warning("No rows have values for the chosen X, color and facets.")
                    return
                if percent:
                    value_label = f"{scale} (%)"
                elif var_y is None:
                    value_label = "Count"
                else:
                    value_label = f"{aggregation} of {var_y}"
                what = "Rows" if var_y is None else f"{aggregation} of {var_y}"
                by_x = f"{var_x} ({date_group.lower()})" if date_group and date_group != "As is" else var_x
                title = f"{what} by {by_x}" + (f" and {hue}" if hue and hue != var_x else "") + (f" · {scale}" if percent else "")

                if tplot == "bars":
                    fig = bar_figure(summary, var_x, value_label, hue=hue, facet_col=facet_col, facet_row=facet_row,
                                     category_orders=category_orders, stacked=stacked, show_labels=show_labels,
                                     percent=percent, title=title, palette=PALETTE, date_group=date_group)
                else:
                    if hue is None or hue == var_x:
                        st.warning("Choose a Color (hue) other than X: the heatmap crosses X with it.")
                        return
                    if date_group:  # Readable row labels in date order.
                        summary = summary.sort_values(var_x)
                        labels = date_labels(summary[var_x], date_group)
                        category_orders[var_x] = list(dict.fromkeys(labels))
                        summary[var_x] = labels
                    table = summary.pivot_table(index=var_x, columns=hue, values="value", aggfunc="sum", observed=True)
                    for axis_name, var in (("index", var_x), ("columns", hue)):
                        if var in category_orders:
                            keep = [c for c in category_orders[var] if c in getattr(table, axis_name)]
                            table = table.reindex(**{axis_name: keep})
                    fig = px.imshow(table, text_auto='.1f' if percent else '.4~g', color_continuous_scale='Viridis',
                                    labels=dict(color=value_label), title=title, width=800, height=600, aspect="auto")
                    fig.update_xaxes(title=hue, type="category")
                    fig.update_yaxes(title=var_x, type="category")

                plotly_chart(fig)

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
                        if axis_var and axis_var in plot_data.columns and is_text_like(plot_data[axis_var]):
                            opts = plot_data[axis_var].dropna().unique().tolist()
                            order = order_select(f"Custom Order for {label}: {axis_var}", axis_var, plot_data[axis_var])
                            plot_data[axis_var] = ordered_categories(plot_data[axis_var], order)
                            category_orders[axis_var] = order
                    # Categories left out of a custom order are left out of the plot.
                    for axis_var in category_orders:
                        plot_data = plot_data[plot_data[axis_var].notna()]

                point_mode = 'all' if swarm_points else False
                jitter_val = 0.4
                point_kwargs = dict(jitter=jitter_val, pointpos=0, marker=dict(size=5, opacity=0.5, line=dict(width=0)))

                if hue_used:
                    groups = levels(plot_data[hue_used])
                    if hue_used in category_orders: groups = [g for g in category_orders[hue_used] if g in groups]
                elif cat_var:
                    groups = levels(plot_data[cat_var])
                    if cat_var in category_orders: groups = [g for g in category_orders[cat_var] if g in groups]
                else:
                    groups = [None]

                fig = go.Figure()

                for i, grp in enumerate(groups):
                    if hue_used:
                        mask = plot_data[hue_used] == grp
                        sub = plot_data[mask]
                        if cat_var:
                            cats = levels(sub[cat_var])
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
                    line_groups = levels(plot_data[hue_used]) if hue_used else [None]
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
                    hue_groups = levels(plot_data[hue_used]) if hue_used else [None]
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
                elif cat_var in category_orders:
                    # With a hue, each trace carries its own categories and Plotly would order the
                    # axis by first appearance; pin it to the custom order.
                    axis = dict(categoryorder="array", categoryarray=[c for c in category_orders[cat_var] if c in levels(plot_data[cat_var])])
                    if orientation == 'v': fig.update_xaxes(**axis)
                    else: fig.update_yaxes(**axis)

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

                plotly_chart(fig, width="content" if single_var else "stretch")

            elif plot_type == 'lineplot':
                var_x = global_var_x
                var_y = global_var_y
                hue_used = global_hue if global_hue != var_x else None
                facet_col_used = global_facet_col
                facet_row_used = global_facet_row

                with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                    style_cols = ['None'] + (data.columns.tolist() if risk_it_all else categorical_cols)
                    width_cols = ['None'] + (data.columns.tolist() if risk_it_all else numeric_cols)

                    aggregation = st.selectbox(
                        "Aggregation", LINE_AGGREGATIONS, index=0,
                        help="How the Y values at each X step become one point. "
                             f"'{RAW_VALUES}' plots every row, joined in X order (e.g., a series with one value per date).")
                    interval, interval_style = "None", "Band"
                    if aggregation == "Mean":
                        interval = st.selectbox("Spread around the mean", list(LINE_INTERVALS), index=0,
                                                help="A confidence interval of the mean, or ±1 standard deviation.")
                        if interval != "None":
                            interval_style = st.radio("Show spread as", ["Band", "Error bars"], horizontal=True)
                    show_markers = st.checkbox("Show Markers", value=True)
                    style = st.selectbox("Line Style (Categorical)", style_cols, index=0,
                                         help="Splits lines by this variable, drawn solid, dashed, dotted...")
                    width = st.selectbox("Line Width (Numeric)", width_cols, index=0,
                                         help="Each line's width follows its mean of this variable.")
                    style_used = None if style == 'None' else style
                    width_used = None if width == 'None' else width

                    plot_data = data.copy()
                    for name in (hue_used, style_used):
                        if name and name in plot_data.columns and not is_text_like(plot_data[name]):
                            plot_data[name] = plot_data[name].astype(str)

                    # A date X (datetime column, or text that is all dates) is plotted on a time
                    # axis in date order, optionally grouped by week, month, quarter or year.
                    x_dates = date_values(plot_data[var_x])
                    if x_dates is not None and not pd.api.types.is_datetime64_any_dtype(plot_data[var_x]):
                        if not st.checkbox("Day First when X holds dates (e.g., 31/12/2023)", value=True):
                            x_dates = date_values(plot_data[var_x], dayfirst=False)
                    date_group = None
                    if x_dates is not None:
                        date_group = st.selectbox(f"Group dates in {var_x} by", list(DATE_GROUPS), index=0,
                                                  help="Month on daily readings gives monthly totals (Sum) or averages (Mean).")
                        plot_data[var_x] = group_dates(x_dates, date_group)

                    category_orders = {}
                    for var_name, prefix in [(var_x, "x"), (hue_used, "hue"), (style_used, "style"),
                                             (facet_col_used, "col"), (facet_row_used, "row")]:
                        if var_name == var_x and x_dates is not None:
                            continue
                        if var_name and var_name in plot_data.columns and var_name not in category_orders \
                                and is_text_like(plot_data[var_name]):
                            c_order = order_select(f"Custom Order for {prefix}: {var_name}", var_name, plot_data[var_name])
                            plot_data[var_name] = ordered_categories(plot_data[var_name], c_order)
                            category_orders[var_name] = c_order

                if var_y in (None, 'None', var_x) or pd.to_numeric(plot_data[var_y], errors="coerce").isna().all():
                    st.warning(f"The lineplot needs a numeric Y other than X; '{var_y}' has no numbers to plot.")
                    return

                line_keys = [g for g in dict.fromkeys([hue_used, style_used]) if g]
                panel_keys = [g for g in dict.fromkeys([facet_row_used, facet_col_used]) if g]
                summary = line_summary(plot_data, var_x, var_y, line_keys + panel_keys, aggregation, interval)
                widths = line_widths(plot_data, line_keys, width_used) if width_used else {}

                facet_col_vals = levels(plot_data[facet_col_used]) if facet_col_used else [None]
                facet_row_vals = levels(plot_data[facet_row_used]) if facet_row_used else [None]
                n_cols = len(facet_col_vals) if facet_col_used else 1
                n_rows = len(facet_row_vals) if facet_row_used else 1

                panel_height = 400
                fig = make_subplots(
                    rows=n_rows, cols=n_cols, shared_yaxes=True, shared_xaxes=True, horizontal_spacing=0.1,
                    # A fixed pixel gap leaves room for each row's x-axis title above the next row's facet title.
                    vertical_spacing=min(LINEPLOT_ROW_GAP_PX / (panel_height * n_rows), 0.9 / max(1, n_rows - 1)),
                    subplot_titles=[
                        f"{facet_row_used}: {r} | {facet_col_used}: {c}" if facet_row_used and facet_col_used else
                        f"{facet_col_used}: {c}" if facet_col_used else f"{facet_row_used}: {r}" if facet_row_used else ""
                        for r in facet_row_vals for c in facet_col_vals
                    ]
                )

                # One line per combination of color and style values (one key when both are the same column).
                line_levels = [levels(plot_data[g]) for g in line_keys]
                color_map = {hv: PALETTE[idx % len(PALETTE)] for idx, hv in enumerate(levels(plot_data[hue_used]) if hue_used else [None])}
                dashes = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
                dash_map = {sv: dashes[idx % len(dashes)] for idx, sv in enumerate(levels(plot_data[style_used]) if style_used else [None])}

                seen = set()
                for i_row, row_val in enumerate(facet_row_vals):
                    for i_col, col_val in enumerate(facet_col_vals):
                        panel = summary
                        if facet_row_used: panel = panel[panel[facet_row_used] == row_val]
                        if facet_col_used: panel = panel[panel[facet_col_used] == col_val]
                        for combo in itertools.product(*line_levels):
                            values = dict(zip(line_keys, combo))
                            sub = panel
                            for g, v in values.items(): sub = sub[sub[g] == v]
                            if sub.empty: continue
                            color = color_map[values.get(hue_used)]
                            name = ", ".join(map(str, combo)) or var_y
                            line = dict(color=color, dash=dash_map[values.get(style_used)], width=widths.get(combo, 2))
                            has_spread = interval != "None" and sub["lower"].notna().any()
                            if has_spread and interval_style == "Band":
                                band = sub.dropna(subset=["lower", "upper"])
                                fig.add_trace(go.Scatter(
                                    x=pd.concat([band[var_x], band[var_x][::-1]]), y=pd.concat([band["upper"], band["lower"][::-1]]),
                                    fill="toself", fillcolor=_rgba(color, 0.2), line=dict(width=0), hoverinfo="skip",
                                    legendgroup=name, showlegend=False, name=f"{name} ({interval})",
                                ), row=i_row + 1, col=i_col + 1)
                            scatter_kwargs = dict(
                                x=sub[var_x], y=sub["value"], mode='lines' + ('+markers' if show_markers else ''),
                                name=name, legendgroup=name, showlegend=name not in seen,
                                marker=dict(size=6, color=color), line=line,
                            )
                            if has_spread and interval_style == "Error bars":
                                scatter_kwargs['error_y'] = dict(type='data', symmetric=False, array=sub["upper"] - sub["value"],
                                                                 arrayminus=sub["value"] - sub["lower"], thickness=1.5, width=5, color=color)
                            seen.add(name)
                            fig.add_trace(go.Scatter(**scatter_kwargs), row=i_row + 1, col=i_col + 1)

                raw = aggregation == RAW_VALUES
                y_title = var_y if raw else f"{aggregation} of {var_y}"
                x_title = f"{var_x} ({date_group.lower()})" if date_group and date_group != "As is" else var_x
                legend_title = ", ".join(dict.fromkeys(g for g in (hue_used, style_used) if g)) or None
                fig.update_layout(width=1000, height=panel_height * n_rows + 80, margin=dict(b=80),
                                  title=f"Lineplot: {y_title} by {x_title}", showlegend=bool(legend_title),
                                  legend_title=legend_title)
                fig.update_xaxes(title_text=x_title, showticklabels=True)
                fig.update_yaxes(title_text=y_title, showticklabels=True)
                if x_dates is not None:
                    fig.update_xaxes(type="date")
                    if date_group and DATE_GROUPS[date_group][1]:
                        fig.update_xaxes(tickformat=DATE_GROUPS[date_group][1])
                elif var_x in category_orders:
                    # All panels show the X steps in the chosen order, even where a line lacks some.
                    fig.update_xaxes(type="category", categoryorder="array", categoryarray=category_orders[var_x])
                plotly_chart(fig)

            elif plot_type == 'ridges':
                var_x = global_var_x
                var_y = global_var_y
                hue = global_hue

                cat_cols = text_columns(data, include_bool=True)
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

                                custom_order_y = order_select(f"Custom Order for Y: {var_y}", var_y, plot_data[var_y])
                                plot_data[var_y] = ordered_categories(plot_data[var_y], custom_order_y)
                                y_values_unique = custom_order_y

                                if hue_param != var_y and hue_param in plot_data.columns:
                                    hue_values_unique = plot_data[hue_param].dropna().unique()
                                    custom_order_hue = order_select(f"Custom Order for Hue: {hue_param}", hue_param, plot_data[hue_param])
                                    plot_data[hue_param] = ordered_categories(plot_data[hue_param], custom_order_hue)
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
                        hue_values = levels(subset[hue_param]) if (hue_param != var_y and hue_param in subset.columns) else [None]

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
                plotly_chart(fig)

            elif plot_type == 'histogram':
                var_x = global_var_x
                hue_var = global_hue if global_hue else 'None'
                facet_col = global_facet_col if global_facet_col else 'None'
                facet_row = global_facet_row if global_facet_row else 'None'

                with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                    multiple = st.selectbox("Multiple", ["layer", "dodge", "stack"])
                    stat = st.selectbox("Stat", ["count", "probability", "percent", "density"])
                    element = st.selectbox("Element", ["bars", "step"])
                    fill_alpha = st.slider("Fill opacity", 0.0, 1.0, 0.3, 0.05,
                                           help="Fill under each step outline; 0 draws the outline only.") if element == "step" else 0.0
                    bar_opacity = st.slider("Bar opacity", 0.1, 1.0, 0.6 if multiple == "layer" else 1.0, 0.05,
                                            help="Below 1, layered colors show through each other.") if element == "bars" else 1.0
                    common_norm = st.checkbox("Common Norm", value=False)
                    cumulative = st.checkbox("Cumulative", value=False)
                    
                    plot_data = data.copy()
                    if hue_var != "None" and hue_var in plot_data.columns: plot_data[hue_var] = plot_data[hue_var].astype(str)

                    category_orders = {}
                    for v_name, label in [(var_x, 'X'), (hue_var, 'Hue'), (facet_col, 'Col'), (facet_row, 'Row')]:
                        if v_name != 'None' and v_name in plot_data.columns and is_text_like(plot_data[v_name]):
                            c_order = order_select(f"Custom Order for {label}: {v_name}", v_name, plot_data[v_name])
                            plot_data[v_name] = ordered_categories(plot_data[v_name], c_order)
                            category_orders[v_name] = c_order

                if element == "step":
                    # Outlines on shared bins, in the app's palette and color order, with facets.
                    hue_used = hue_var if hue_var != "None" else None
                    steps = step_histogram_frame(
                        plot_data, var_x, hue_used,
                        (facet_col if facet_col != "None" else None, facet_row if facet_row != "None" else None),
                        stat=stat, cumulative=cumulative, stacked=multiple == "stack",
                    )
                    fig = px.line(
                        steps, x=var_x, y="value", color=hue_used, line_shape="hv",
                        facet_col=facet_col if facet_col != "None" else None, facet_row=facet_row if facet_row != "None" else None,
                        category_orders=category_orders, color_discrete_sequence=PALETTE, width=800, height=600,
                        labels={"value": stat.capitalize()}, title=f"Step histogram of {var_x}",
                    )
                    fig.update_traces(line_width=2)
                    fill_steps(fig, fill_alpha)
                    if multiple == "dodge":
                        st.caption("Step outlines are drawn on top of each other; dodge applies to bars.")
                else:
                    fig = px.histogram(
                        plot_data, x=var_x, color=hue_var if hue_var != "None" else None,
                        facet_col=facet_col if facet_col != "None" else None, facet_row=facet_row if facet_row != "None" else None,
                        histnorm=stat if stat != "count" else None, cumulative=cumulative,
                        category_orders=category_orders, color_discrete_sequence=PALETTE, width=800, height=600
                    )
                    histogram_bars(fig, multiple, bar_opacity)
                plotly_chart(fig)

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
                        if var_name and var_name != 'None' and is_text_like(plot_data[var_name]):
                            c_order = order_select(f"Custom Order for {prefix}: {var_name}", var_name, plot_data[var_name])
                            plot_data[var_name] = ordered_categories(plot_data[var_name], c_order)
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
                                        facet_combinations[facet] = ordered_categories(facet_combinations[facet], category_orders[facet])
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

                        hue_values = levels(group_df[hue_var]) if hue_var != 'None' else [None]
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
                    plotly_chart(fig)

                else:
                    if kind == "hist":
                        fig = px.density_heatmap(
                            plot_data, x=var_x, y=var_y, facet_col=facet_col if facet_col != 'None' else None, facet_row=facet_row if facet_row != 'None' else None,
                            # A list, not 'Viridis': px builds the rug's colour from the scale and fails on the name.
                            color_continuous_scale=px.colors.sequential.Viridis, histnorm='probability density' if not common_norm else None,
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

                    plotly_chart(fig)


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
                        if var_name != 'None' and var_name in plot_data.columns and is_text_like(plot_data[var_name]):
                            custom_order = order_select(f"Custom Order for {label} ({var_name})", var_name, plot_data[var_name])
                            if custom_order:
                                plot_data[var_name] = ordered_categories(plot_data[var_name], custom_order)
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
                plotly_chart(fig)

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
                        if v_name != 'None' and v_name in plot_data.columns and is_text_like(plot_data[v_name]):
                            c_order = order_select(f"Custom Order for {label} ({v_name})", v_name, plot_data[v_name])
                            if c_order:
                                plot_data[v_name] = ordered_categories(plot_data[v_name], c_order)
                                category_orders[v_name] = c_order
                    # Categories left out of a custom order are left out of the plot.
                    for v_name in category_orders:
                        plot_data = plot_data[plot_data[v_name].notna()]

                if col != 'None' or row != 'None':
                    col_vals = levels(plot_data[col]) if col != 'None' else [None]
                    row_vals = levels(plot_data[row]) if row != 'None' else [None]
                    n_cols, n_rows = max(len(col_vals), 1), max(len(row_vals), 1)

                    fig = make_subplots(
                        rows=n_rows, cols=n_cols,
                        subplot_titles=[f"{row}: {r} | {col}: {c}" if row != 'None' and col != 'None' else f"{col}: {c}" if col != 'None' else f"{row}: {r}" if row != 'None' else "" for r in row_vals for c in col_vals],
                        shared_yaxes=True, shared_xaxes=True, horizontal_spacing=0.1, vertical_spacing=0.12
                    )

                    hue_values = levels(plot_data[hue]) if hue != 'None' else [None]
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

                plotly_chart(fig)

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
                plotly_chart(fig)

            elif plot_type in ('candlestick', 'waterfall'):
                var_x = global_var_x
                var_y = global_var_y
                unused = [name for name, used in (("Color", global_hue), ("Facet Col", global_facet_col), ("Facet Row", global_facet_row), ("Size", global_size)) if used]
                if unused:
                    st.info(f"ℹ️ {plot_type.capitalize()} plots do not use {', '.join(unused)}.")
                plot_data = data.copy()

                with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                    dayfirst = st.checkbox("Day First when X holds dates (e.g., 31/12/2023)", value=True)
                    x_values = plot_data[var_x].dropna()
                    # Dates are plotted in date order; other text gets a custom order.
                    if is_text_like(x_values) and not parse_dates(x_values, dayfirst).notna().all():
                        c_order = order_select(f"Custom Order for x: {var_x}", var_x, plot_data[var_x], sort=False)
                        plot_data[var_x] = ordered_categories(plot_data[var_x], c_order)
                    if plot_type == 'candlestick':
                        source = st.radio("Candles from", CANDLE_SOURCES, index=0,
                                          help="One value column: each X step's first, highest, lowest and last value. "
                                               "Or pick existing Open/High/Low/Close columns.")
                        ohlc_cols = None
                        if source == CANDLE_SOURCES[1]:
                            ohlc_cols = []
                            for name in ("Open", "High", "Low", "Close"):
                                guess = next((c for c in numeric_cols if c.lower() == name.lower()), var_y)
                                ohlc_cols.append(st.selectbox(name, numeric_cols, index=numeric_cols.index(guess) if guess in numeric_cols else 0))
                        range_slider = st.checkbox("Range Slider", value=False)
                        weekends = has_weekend_dates(plot_data[var_x], dayfirst)
                        hide_weekends = st.checkbox(
                            "Hide weekend gaps", value=not weekends,
                            help="Removes Saturdays and Sundays from the date axis, so trading days sit side by side. "
                                 "On by default when no row falls on a weekend; it would hide weekend candles otherwise.",
                        ) if weekends is not None else False
                    else:
                        aggregation_method = st.selectbox("Change per X step", ["Sum", "Mean"], index=0)
                        show_total = st.checkbox("Show Total Bar", value=True)

                try:
                    if plot_type == 'candlestick':
                        ohlc = ohlc_frame(plot_data, var_x, value=None if ohlc_cols else var_y, ohlc=ohlc_cols, dayfirst=dayfirst)
                        y_label = var_y if not ohlc_cols else ", ".join(dict.fromkeys(ohlc_cols))
                        fig = candlestick_figure(ohlc, f"Candlestick of {y_label} by {var_x}", y_label, range_slider, hide_weekends)
                    else:
                        changes = waterfall_frame(plot_data, var_x, var_y, aggregation_method, dayfirst)
                        fig = waterfall_figure(changes, f"Waterfall: {aggregation_method} of {var_y} by {var_x}", var_y, show_total)
                except (KeyError, TypeError, ValueError) as exc:
                    st.warning(f"Could not build the {plot_type} plot: {exc}")
                    return
                if fig.data[0].x is None or not len(fig.data[0].x):
                    st.warning(f"No data to plot: {var_x} and {var_y} have no complete rows.")
                    return
                plotly_chart(fig)

            elif plot_type == 'heatmap':
                cat_var = global_var_x if global_var_x != 'None' else None
                num_var = global_var_y
                plot_data = data.copy()

                with st.sidebar.expander("⚙️ Plot Configuration", expanded=False):
                    colormap = st.selectbox("Select Colormap", ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Blues", "Reds", "Greens", "RdBu_r"], index=0)
                    if cat_var is not None and cat_var in plot_data.columns:
                        custom_order = order_select(f"Custom Order for {cat_var}", cat_var, plot_data[cat_var])
                        if custom_order: plot_data[cat_var] = ordered_categories(plot_data[cat_var], custom_order)
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
                    plotly_chart(fig)
                    return

                if num_var is None or not pd.api.types.is_numeric_dtype(plot_data[num_var]):
                    st.warning(f"Please select a valid numerical Y variable for Heatmap mode.")
                    return

                is_categorical = False
                if isinstance(plot_data[cat_var].dtype, pd.CategoricalDtype): is_categorical = True
                elif is_text_like(plot_data[cat_var]): is_categorical = True
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
                plotly_chart(fig)

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