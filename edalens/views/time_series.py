import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:  # In-browser mode (edalens.walk(mode="browser")): prophet cannot run in WebAssembly.
    PROPHET_AVAILABLE = False

    class Prophet:
        """Stand-in that explains why Prophet-based sections are unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Prophet is not available when EDALens runs inside the browser; "
                "use edalens.walk(df) or edalens.show() for forecasting."
            )
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import requests
import io
from edalens.utils.ui import sidebar_menu


def _frequency_aliases():
    """Resample aliases for this pandas: 2.2 renamed H/M/Q/Y to h/ME/QE/YE and pandas 3 rejects the old ones."""
    try:
        pd.tseries.frequencies.to_offset("ME")
    except ValueError:
        return {"Hourly": "H", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
    return {"Hourly": "h", "Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}


FREQ = _frequency_aliases()
CALENDAR_YEAR_PX = 180  # Height per year of the daily calendar heatmap.


def _require_prophet(section):
    """Explain and stop the page when Prophet cannot run (in-browser mode)."""
    if not PROPHET_AVAILABLE:
        st.info(
            f"**{section}** uses Prophet, which cannot run when EDALens runs inside the browser "
            "(`edalens.walk(df, mode=\"browser\")`). Use `edalens.walk(df)` or `edalens.show()` for this section."
        )
        st.stop()


def time_series_analysis():
    #st.subheader("📈 Time Series Analysis & Anomaly Detection")

    with st.sidebar:
        analysis_section = sidebar_menu(
            "Analysis Section",
            [
                "Load Data",
                "Time Series Visualization",
                "Decomposition & ACF/PACF",
                "Anomaly Detection",
                "Temporal Profiles",
                "Calendar Heatmap",
                "Heatmap Profiles",
                "Forecasting",
                "Changepoint Analysis",
                "Trend Analysis"
            ],
            icons=["upload", "bar-chart", "graph-up", "exclamation-triangle", "clock", "calendar", "heatmap", "predict", "pin-angle", "trending-up"],
            menu_icon="clock",
        )

    if analysis_section == "Load Data":
        na_values = ['', 'NA', 'N/A', 'na', 'n/a', 'NaN', 'nan', 'NULL', 'null', 'missing', '-', '--']
        df = None

        source_type = st.radio("Select data source", ["Upload CSV", "GitHub Link"])

        if source_type == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file, na_values=na_values)
                    st.success("✅ File successfully uploaded!")
                except Exception as e:
                    st.error(f"❌ Failed to load CSV: {e}")
                    st.stop()

        else:
            github_url = st.text_input("Paste GitHub CSV URL")
            if github_url:
                try:
                    github_url = github_url.strip()
                    if "github.com" in github_url and "raw.githubusercontent.com" not in github_url:
                        github_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                    if not github_url.lower().split("?")[0].endswith('.csv'):
                        st.error("❌ The provided link does not point to a CSV file.")
                        st.stop()
                    response = requests.get(github_url, timeout=10)
                    response.raise_for_status()
                    csv_content = response.content.decode('utf-8')
                    df = pd.read_csv(io.StringIO(csv_content), na_values=na_values)
                    st.success("✅ File successfully loaded from GitHub!")
                except Exception as e:
                    st.error(f"❌ Failed to load CSV: {e}")
                    st.stop()

        if df is not None:
            date_col = st.selectbox("Select the date column", df.columns, index=0)
            date_format = st.radio(
                "Select date format",
                options=["DD/MM/YYYY (Day First)", "MM/DD/YYYY (Month First)"],
                index=0
            )
            dayfirst = date_format == "DD/MM/YYYY (Day First)"

            try:
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors='coerce')
                invalid_dates = df[date_col].isna().sum()
                if invalid_dates > 0:
                    st.warning(f"⚠️ {invalid_dates} rows could not be parsed as dates and will be removed.")
                df = df.dropna(subset=[date_col])
                if df.empty:
                    st.error("❌ No valid dates remain after parsing.")
                    st.stop()
                df = df.sort_values(by=date_col).set_index(date_col)
                st.session_state['df'] = df
                st.success(f"✅ Data loaded successfully! {len(df)} rows processed.")
                st.write("**Preview of Loaded Data (First 5 Rows):**")
                st.write(df.head())
            except Exception as e:
                st.error(f"❌ Error parsing dates: {e}")
                st.stop()
        st.stop()

    if 'df' not in st.session_state:
        st.warning("Please load data first using the 'Load Data' section.")
        return

    df = st.session_state['df']

    st.sidebar.header("Options")

    numerical_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]

    use_multi = st.sidebar.checkbox("🔢 Use multiple variables", value=False,
                                    help="Enables multi-var support (Time Series Viz, Anomaly, Forecasting, Changepoint, Trend)")

    if use_multi:
        selected_vars = st.sidebar.multiselect(
            "Selected Variables (click ❌ to remove • order = legend order)",
            options=numerical_cols,
            default=numerical_cols[:1],
            key="multi_var_select"
        )
        if not selected_vars:
            st.error("Select at least one variable")
            st.stop()
    else:
        selected_col = st.sidebar.selectbox(
            "Select a variable to analyze",
            numerical_cols
        )
        selected_vars = [selected_col]

    selected_col = selected_vars[0]

    time_granularity = st.sidebar.selectbox(
        "Select summarization frequency",
        ["None", "Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
        index=0
    )
    aggregation_method = st.sidebar.selectbox(
        "Aggregation Method",
        ["Mean", "Sum"],
        index=0
    ) if time_granularity != "None" else "Raw"
    window = st.sidebar.number_input("Rolling Window (set 0 for no smoothing)", min_value=0, max_value=60, value=0)

    resample_map = {
        "None": None, "Hourly": FREQ["Hourly"], "Daily": "D", "Weekly": "W",
        "Monthly": FREQ["Monthly"], "Quarterly": FREQ["Quarterly"], "Yearly": FREQ["Yearly"]
    }
    resample_freq = resample_map[time_granularity]

    df_selected = df[selected_vars].copy()

    if resample_freq:
        agg_dict = {col: aggregation_method.lower() for col in selected_vars}
        resampled = df_selected.resample(resample_freq).agg(agg_dict)
        if window > 0:
            resampled = resampled.rolling(window).mean()
        df_summary = resampled
    else:
        df_summary = df_selected.copy()
        if window > 0:
            df_summary = df_summary.rolling(window).mean()

    # ====================== ANALYSIS SECTIONS ======================

    if analysis_section == "Time Series Visualization":
        st.subheader("📈 Time Series Visualization")

        facet_option = False
        same_y_axis = False
        facet_cols = 3

        if use_multi and len(selected_vars) > 1:
            facet_option = st.checkbox("📊 Use facetted plots (one subplot per variable)", value=False)
            if facet_option:
                same_y_axis = st.checkbox("🔗 Same Y-axis scale for all facets", value=True)
                facet_cols = st.slider("📏 Number of facet columns", 1, 3, 3)

        if not facet_option or len(selected_vars) == 1:
            # Overlaid plot with correct bands
            fig = go.Figure()

            for col in selected_vars:
                # Main line (smoothed/aggregated)
                fig.add_trace(go.Scatter(
                    x=df_summary.index, 
                    y=df_summary[col], 
                    mode='lines', 
                    name=col
                ))

                if len(selected_vars) == 1:
                    # Correct bands: use raw data variability, not smoothed
                    raw_col = df[[col]].resample('D').agg(aggregation_method.lower() if time_granularity != "None" else 'mean')
                    std_series = raw_col[col].rolling(window if window > 0 else 7).std().reindex(df_summary.index, method='nearest')

                    fig.add_trace(go.Scatter(
                        x=df_summary.index, 
                        y=df_summary[col] + std_series, 
                        mode='lines', 
                        name='Upper Band', 
                        line=dict(dash='dot', color='rgba(0,0,0,0.3)')
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_summary.index, 
                        y=df_summary[col] - std_series, 
                        mode='lines', 
                        name='Lower Band', 
                        line=dict(dash='dot', color='rgba(0,0,0,0.3)'),
                        fill='tonexty',
                        fillcolor='rgba(0,100,80,0.15)'
                    ))

            fig.update_layout(
                title=f"Time Series ({aggregation_method if resample_freq else 'Raw'})",
                xaxis_title="Date",
                yaxis_title="Value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
        else:
            # Faceted
            df_long = df_summary.reset_index().melt(id_vars=df_summary.index.name or 'date',
                                                    var_name='Variable', value_name='Value')
            df_long = df_long.rename(columns={df_long.columns[0]: 'Date'})
            fig = px.line(
                df_long, x='Date', y='Value', color='Variable',
                facet_col='Variable', facet_col_wrap=facet_cols,
                title=f"Time Series – Facetted ({aggregation_method if resample_freq else 'Raw'})"
            )
            if same_y_axis:
                fig.update_yaxes(matches='y')
            else:
                fig.update_yaxes(matches=None)

        st.plotly_chart(fig, use_container_width=True)

    elif analysis_section == "Decomposition & ACF/PACF":
        st.subheader("📉 Seasonal Decomposition")
        if use_multi and len(selected_vars) > 1:
            st.info(f"🔢 Decomposition is univariate → using **{selected_col}**")

        decomposition_method = st.selectbox(
            "Select Decomposition Method", ["Statsmodels"] + (["Prophet"] if PROPHET_AVAILABLE else []), index=0
        )

        if decomposition_method == "Statsmodels":
            window_dec = st.slider("Seasonal window for decomposition", 2, 60, 12)
            try:
                result = seasonal_decompose(df[selected_col], model='additive', period=window_dec)
                fig = result.plot()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Decomposition failed: {e}")
        else:
            df_prophet = df[[selected_col]].reset_index().rename(columns={df.index.name or 'date': 'ds', selected_col: 'y'})
            df_prophet['y'] = df_prophet['y'].ffill()
            try:
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
                model.fit(df_prophet)
                forecast = model.predict(df_prophet[['ds']])

                fig = make_subplots(rows=4, cols=1, subplot_titles=("Trend", "Yearly", "Weekly", "Daily"))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines'), row=1, col=1)
                if 'yearly' in forecast: fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yearly'], mode='lines'), row=2, col=1)
                if 'weekly' in forecast: fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['weekly'], mode='lines'), row=3, col=1)
                if 'daily' in forecast: fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['daily'], mode='lines'), row=4, col=1)
                fig.update_layout(height=800, title_text=f"Prophet Decomposition of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Prophet failed: {e}")

        st.subheader("🔁 ACF and PACF")
        fig_acf, ax = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(df[selected_col].dropna(), ax=ax[0])
        plot_pacf(df[selected_col].dropna(), ax=ax[1])
        st.pyplot(fig_acf)

    elif analysis_section == "Anomaly Detection":
        st.subheader("🚨 Anomaly Detection (Isolation Forest)")
        contamination = st.slider("Contamination", 0.01, 0.5, 0.05, 0.01)

        if use_multi and len(selected_vars) > 1:
            facet_option = st.checkbox("📊 Facetted anomaly plots", value=True)
            same_y = st.checkbox("Same Y-axis", value=True) if facet_option else False
            rows = len(selected_vars)
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles=selected_vars)
            for i, col in enumerate(selected_vars, 1):
                df_anom = df_summary[[col]].dropna().rename(columns={col: 'value'})
                df_anom['anomaly'] = IsolationForest(contamination=contamination, random_state=42).fit_predict(df_anom[['value']])
                df_anom['anomaly'] = df_anom['anomaly'].map({1: 0, -1: 1})
                fig.add_trace(go.Scatter(x=df_anom.index, y=df_anom['value'], mode='lines'), row=i, col=1)
                anom = df_anom[df_anom['anomaly'] == 1]
                if not anom.empty:
                    fig.add_trace(go.Scatter(x=anom.index, y=anom['value'], mode='markers', marker=dict(color='red', size=8)), row=i, col=1)
            if same_y: fig.update_yaxes(matches='y')
            fig.update_layout(height=300*rows, title="Anomaly Detection (Multi-Variable)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            df_anom = df_summary[[selected_col]].dropna().rename(columns={selected_col: 'value'})
            df_anom['anomaly'] = IsolationForest(contamination=contamination, random_state=42).fit_predict(df_anom[['value']])
            df_anom['anomaly'] = df_anom['anomaly'].map({1: 0, -1: 1})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_anom.index, y=df_anom['value'], mode='lines'))
            fig.add_trace(go.Scatter(x=df_anom[df_anom['anomaly']==1].index, y=df_anom[df_anom['anomaly']==1]['value'],
                                     mode='markers', marker=dict(color='red', size=8)))
            fig.update_layout(title=f"Anomaly Detection - {selected_col}")
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_section == "Temporal Profiles":
        # Unchanged (single-var)
        st.subheader("🕒 Temporal Profiles")
        profile_unit = st.selectbox("Profile by", ["Hour of Day", "Day of Week", "Month", "Quarter", "Year"])
        df_profile = df[[selected_col]].copy()

        if profile_unit == "Hour of Day":
            df_profile['profile'] = df_profile.index.hour
        elif profile_unit == "Day of Week":
            df_profile['profile'] = df_profile.index.dayofweek
        elif profile_unit == "Month":
            df_profile['profile'] = df_profile.index.month
        elif profile_unit == "Quarter":
            df_profile['profile'] = df_profile.index.quarter
        else:
            df_profile['profile'] = df_profile.index.year

        profile_agg_method = st.selectbox(
            "Aggregation Method",
            ["Mean", "Median", "Sum", "Min", "Max", "Std"],
            key="profile_agg_method"
        )
        plot_type = st.radio("Plot type", ["Boxplot", "Lineplot"], horizontal=True)

        if plot_type == "Boxplot":
            fig = px.box(df_profile, x='profile', y=selected_col, title=f"Boxplot by {profile_unit}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            agg_func = profile_agg_method.lower()
            profile_summary = df_profile.groupby('profile')[selected_col].agg([agg_func, 'std'])
            profile_summary.columns = ['value', 'std']

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=profile_summary.index, y=profile_summary['value'],
                                     mode='lines+markers', name=profile_agg_method))
            fig.add_trace(go.Scatter(x=profile_summary.index, y=profile_summary['value'] + profile_summary['std'],
                                     mode='lines', name='Upper', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=profile_summary.index, y=profile_summary['value'] - profile_summary['std'],
                                     mode='lines', name='Lower', line=dict(dash='dot')))
            fig.update_layout(
                title=f"Profile by {profile_unit} ({profile_agg_method})",
                xaxis_title=profile_unit,
                yaxis_title=selected_col
            )
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_section == "Calendar Heatmap":

        st.subheader("📆 Calendar Heatmap")

        # === Colorscale selector (works for both Daily and Hourly) ===
        available_colorscales = [
            "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
            "Blues", "Greens", "Reds", "YlOrRd", "RdYlBu",
            "Turbo", "Rainbow", "Jet", "Hot", "Gray"
        ]

        # Toggle between Daily and Hourly
        heatmap_type = st.radio(
            "Select heatmap type",
            options=["Daily", "Hourly"],
            horizontal=True,
            help="Daily: full years visible. Hourly: detailed month grid with hours"
        )

        agg_method = aggregation_method.lower() if aggregation_method != "Raw" else "mean"

        try:
            from plotly_calheatmap import calheatmap, hourly_calheatmap
        except ImportError:
            st.error("❌ Please install plotly-calheatmap:\n`pip install plotly-calheatmap`")
            st.stop()

        if heatmap_type == "Daily":
            # Daily view — all years visible
            daily_df = df[[selected_col]].resample('D').agg(agg_method).reset_index()
            daily_df.columns = ['date', selected_col]

            greens = ["#161b22", "#0e4429", "#006d32", "#26a641", "#39d353"]
            n_years = max(1, daily_df["date"].dt.year.nunique())

            fig = calheatmap(
                data=daily_df,
                x="date",
                y=selected_col,
                colors=greens,      # ← uses the selected colorscale
                scale_type="linear",
                zero_color="#161b22",
                nan_color="aliceblue",
                replace_nans_with_zeros=False,
                annotations=False,
                month_lines=True,
                week_start="monday",
                vertical=False,
                navigation=False,
                skip_empty_years=True,
                # One row per year: Plotly needs the gap below 1 / (rows - 1).
                total_height=CALENDAR_YEAR_PX * n_years,
                space_between_plots=min(0.08, 0.5 / max(1, n_years - 1)),
                gap=2,
                title=f"Daily Calendar Heatmap – {selected_col} ({aggregation_method})"
            )

        else:
            selected_colorscale = st.selectbox(
                        "Choose Colorscale",
                        options=available_colorscales,
                        index=0,  # default Viridis
                        help="Applies to both Daily and Hourly heatmaps"
                    )
            # Hourly view — month grid like your screenshot
            hourly_df = df[[selected_col]].resample(FREQ["Hourly"]).agg(agg_method).reset_index()
            hourly_df.columns = ['datetime', selected_col]
            hourly_df[selected_col] = hourly_df[selected_col].ffill().fillna(0)

            fig = hourly_calheatmap(
                data=hourly_df,
                x="datetime",
                y=selected_col,
                colorscale=selected_colorscale,   # ← uses the selected colorscale
                title=f"Hourly {selected_col} Heatmap ({aggregation_method})"
            )

            fig.update_layout(height=850)

        st.plotly_chart(fig, use_container_width=True)

    elif analysis_section == "Heatmap Profiles":
        # Unchanged (single-var)
        st.subheader("📅 Time Profiles Heatmap")
        df_time = df[[selected_col]].copy()

        df_time['year'] = df_time.index.year
        df_time['month'] = df_time.index.month
        df_time['hour'] = df_time.index.hour
        df_time['weekday'] = df_time.index.dayofweek
        df_time['week'] = df_time.index.isocalendar().week

        profile_type = st.selectbox(
            "Select Profile Type",
            ["Monthly", "Hourly", "Yearly", "Weekday", "Week of Year"],
            key="heatmap_profile_type"
        )

        heatmap_agg_method = st.selectbox(
            "Aggregation Method",
            ["Mean", "Median", "Sum", "Min", "Max", "Std"],
            key="heatmap_agg_method"
        )
        agg_func = heatmap_agg_method.lower()

        if profile_type == "Monthly":
            st.subheader("📅 Monthly Heatmap")
            heatmap_data = df_time.groupby(['year', 'month'])[selected_col].agg(agg_func).unstack()
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Month", y="Year", color=selected_col),
                x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                y=heatmap_data.index,
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                title=f"Monthly Heatmap ({heatmap_agg_method})",
                xaxis_title="Month",
                yaxis_title="Year",
                coloraxis_colorbar=dict(title=selected_col, thickness=20, len=0.8, x=1., y=0.5, yanchor="middle", tickformat=".2f", ticks="outside")
            )

        elif profile_type == "Hourly":
            st.subheader("⏰ Hourly Heatmap")
            heatmap_data = df_time.groupby(['year', 'hour'])[selected_col].agg(agg_func).unstack()
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Hour", y="Year", color=selected_col),
                x=[f"{h:02d}:00" for h in range(24)],
                y=heatmap_data.index,
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                title=f"Hourly Heatmap ({heatmap_agg_method})",
                xaxis_title="Hour of Day",
                yaxis_title="Year",
                coloraxis_colorbar=dict(title=selected_col, thickness=20, len=0.8, x=1., y=0.5, yanchor="middle", tickformat=".2f", ticks="outside")
            )

        elif profile_type == "Yearly":
            st.subheader("📅 Yearly Heatmap")
            heatmap_data = df_time.groupby(['year'])[selected_col].agg(agg_func).to_frame().T
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Year", y="", color=selected_col),
                x=heatmap_data.columns,
                y=["Value"],
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                title=f"Yearly Heatmap ({heatmap_agg_method})",
                xaxis_title="Year",
                yaxis_title="",
                coloraxis_colorbar=dict(title=selected_col, thickness=20, len=0.8, x=1., y=0.5, yanchor="middle", tickformat=".2f", ticks="outside")
            )

        elif profile_type == "Weekday":
            st.subheader("📅 Weekday Heatmap")
            heatmap_data = df_time.groupby(['year', 'weekday'])[selected_col].agg(agg_func).unstack()
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Day of Week", y="Year", color=selected_col),
                x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                y=heatmap_data.index,
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                title=f"Weekday Heatmap ({heatmap_agg_method})",
                xaxis_title="Day of Week",
                yaxis_title="Year",
                coloraxis_colorbar=dict(title=selected_col, thickness=20, len=0.8, x=1., y=0.5, yanchor="middle", tickformat=".2f", ticks="outside")
            )

        elif profile_type == "Week of Year":
            st.subheader("📅 Week of Year Heatmap")
            heatmap_data = df_time.groupby(['year', 'week'])[selected_col].agg(agg_func).unstack()
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Week of Year", y="Year", color=selected_col),
                x=list(range(1, 54)),
                y=heatmap_data.index,
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                title=f"Week of Year Heatmap ({heatmap_agg_method})",
                xaxis_title="Week of Year",
                yaxis_title="Year",
                coloraxis_colorbar=dict(title=selected_col, thickness=20, len=0.8, x=1., y=0.5, yanchor="middle", tickformat=".2f", ticks="outside")
            )

        st.plotly_chart(fig, use_container_width=True)

    elif analysis_section == "Forecasting":
        _require_prophet("Forecasting")
        st.subheader(f"📈 Forecasting")

        if use_multi and len(selected_vars) > 1:
            st.info("🔢 Forecasting uses Prophet (univariate) → one tab per variable")
            tabs = st.tabs(selected_vars)
            for idx, col in enumerate(selected_vars):
                with tabs[idx]:
                    # Original Prophet forecasting code adapted for this variable
                    df_prophet = df[[col]].reset_index()
                    df_prophet = df_prophet.rename(columns={df_prophet.columns[0]: 'ds', col: 'y'})
                    if time_granularity != "None":
                        df_prophet = df_prophet.set_index('ds').resample(resample_map[time_granularity]).agg({'y': aggregation_method.lower()}).reset_index()
                    df_prophet = df_prophet[['ds', 'y']].copy()
                    df_prophet['y'] = df_prophet['y'].ffill()

                    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
                    model.fit(df_prophet)

                    unit_map = {"Hourly": "hours", "Daily": "days", "Weekly": "weeks", "Monthly": "months",
                                "Quarterly": "quarters", "Yearly": "years", "None": "days"}
                    unit = unit_map.get(time_granularity, "days")

                    if time_granularity == "None":
                        forecast_period = st.slider(f"Select forecast period ({unit})", 1, 365, 30, format="%d", key=f"fp_{col}")
                    elif time_granularity == "Hourly":
                        forecast_period = st.slider(f"Select forecast period ({unit})", 24, 8760, 720, format="%d", key=f"fp_{col}")
                    elif time_granularity == "Daily":
                        forecast_period = st.slider(f"Select forecast period ({unit})", 7, 365, 30, format="%d", key=f"fp_{col}")
                    elif time_granularity == "Weekly":
                        forecast_period = st.slider(f"Select forecast period ({unit})", 1, 52, 4, format="%d", key=f"fp_{col}")
                    elif time_granularity == "Monthly":
                        forecast_period = st.slider(f"Select forecast period ({unit})", 1, 12, 3, format="%d", key=f"fp_{col}")
                    elif time_granularity == "Quarterly":
                        forecast_period = st.slider(f"Select forecast period ({unit})", 1, 4, 1, format="%d", key=f"fp_{col}")
                    elif time_granularity == "Yearly":
                        forecast_period = st.slider(f"Select forecast period ({unit})", 1, 5, 1, format="%d", key=f"fp_{col}")

                    future = model.make_future_dataframe(periods=forecast_period, freq=resample_map.get(time_granularity, 'D'))
                    forecast = model.predict(future)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Historical Data', mode='lines'))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', mode='lines'))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', mode='lines', line=dict(color='rgba(0,0,0,0)')))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', mode='lines', line=dict(color='rgba(0,0,0,0)')))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)', name='Confidence Interval'))

                    fig.update_layout(
                        title=f"Forecast for {col} ({aggregation_method if time_granularity != 'None' else 'Raw'})",
                        xaxis_title="Date",
                        yaxis_title=col,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            # Original single-var forecasting code
            df_prophet = df[[selected_col]].reset_index()
            df_prophet = df_prophet.rename(columns={df_prophet.columns[0]: 'ds', selected_col: 'y'})
            if time_granularity != "None":
                df_prophet = df_prophet.set_index('ds').resample(resample_map[time_granularity]).agg({'y': aggregation_method.lower()}).reset_index()
            df_prophet = df_prophet[['ds', 'y']].copy()
            df_prophet['y'] = df_prophet['y'].ffill()

            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
            model.fit(df_prophet)

            unit_map = {"Hourly": "hours", "Daily": "days", "Weekly": "weeks", "Monthly": "months",
                        "Quarterly": "quarters", "Yearly": "years", "None": "days"}
            unit = unit_map.get(time_granularity, "days")

            if time_granularity == "None":
                forecast_period = st.slider(f"Select forecast period ({unit})", 1, 365, 30, format="%d")
            elif time_granularity == "Hourly":
                forecast_period = st.slider(f"Select forecast period ({unit})", 24, 8760, 720, format="%d")
            elif time_granularity == "Daily":
                forecast_period = st.slider(f"Select forecast period ({unit})", 7, 365, 30, format="%d")
            elif time_granularity == "Weekly":
                forecast_period = st.slider(f"Select forecast period ({unit})", 1, 52, 4, format="%d")
            elif time_granularity == "Monthly":
                forecast_period = st.slider(f"Select forecast period ({unit})", 1, 12, 3, format="%d")
            elif time_granularity == "Quarterly":
                forecast_period = st.slider(f"Select forecast period ({unit})", 1, 4, 1, format="%d")
            elif time_granularity == "Yearly":
                forecast_period = st.slider(f"Select forecast period ({unit})", 1, 5, 1, format="%d")

            future = model.make_future_dataframe(periods=forecast_period, freq=resample_map.get(time_granularity, 'D'))
            forecast = model.predict(future)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Historical Data', mode='lines'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', mode='lines'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', mode='lines', line=dict(color='rgba(0,0,0,0)')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', mode='lines', line=dict(color='rgba(0,0,0,0)')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)', name='Confidence Interval'))

            fig.update_layout(
                title=f"Forecast for {selected_col} ({aggregation_method if time_granularity != 'None' else 'Raw'})",
                xaxis_title="Date",
                yaxis_title=selected_col,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # CHANGEPOINT ANALYSIS WITH PROPHET
    # ============================================================
    elif analysis_section == "Changepoint Analysis":
        _require_prophet("Changepoint Analysis")

        st.subheader("🔍 Changepoint Analysis (Prophet)")


        # --------------------------------------------------------
        # Helper function
        # --------------------------------------------------------

        def run_prophet_changepoints(
            df_prophet,
            changepoint_prior,
            sensitivity,
            forced_n,
            use_yearly
        ):


            # -----------------------------
            # Adaptive number of candidates
            # -----------------------------

            n_points = len(df_prophet)


            n_candidates = min(
                50,
                max(5, n_points // 10)
            )


            # -----------------------------
            # Prophet model
            # -----------------------------

            model = Prophet(
                yearly_seasonality=use_yearly,
                weekly_seasonality=False,
                daily_seasonality=False,
                n_changepoints=n_candidates,
                changepoint_range=0.95,
                changepoint_prior_scale=changepoint_prior,
                seasonality_mode="additive"
            )


            model.fit(df_prophet)


            forecast = model.predict(df_prophet)



            # -----------------------------
            # Extract Prophet deltas
            # -----------------------------

            deltas = (
                np.array(model.params["delta"])
                .mean(axis=0)
            )


            abs_deltas = np.abs(deltas)



            # ====================================================
            # NEW: two selection modes
            # ====================================================

            if forced_n is None:


                # ----- original behavior -----

                threshold = np.percentile(
                    abs_deltas,
                    sensitivity
                )


                selected_idx = np.where(
                    abs_deltas >= threshold
                )[0]


            else:


                # ----- fixed number mode -----

                selected_idx = (
                    np.argsort(abs_deltas)[::-1]
                    [:forced_n]
                )



            # chronological order for plotting

            selected_idx = np.sort(selected_idx)



            significant_cp = (
                model.changepoints
                .iloc[selected_idx]
            )



            # correct Date <-> Delta mapping

            cp_table = pd.DataFrame(
                {
                    "Date":
                        significant_cp.values,

                    "Delta":
                        deltas[selected_idx],

                    "Abs Delta":
                        abs_deltas[selected_idx]
                }
            )



            if len(cp_table):

                cp_table = (
                    cp_table
                    .sort_values(
                        "Abs Delta",
                        ascending=False
                    )
                )


            return (
                model,
                forecast,
                significant_cp,
                cp_table
            )



        # --------------------------------------------------------
        # Plot helper
        # --------------------------------------------------------

        def plot_changepoints(
            df_prophet,
            forecast,
            changepoints,
            title
        ):


            fig = go.Figure()


            fig.add_trace(
                go.Scatter(
                    x=df_prophet["ds"],
                    y=df_prophet["y"],
                    name="Observed",
                    mode="lines",
                    line=dict(width=2)
                )
            )


            fig.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat"],
                    name="Forecast"
                )
            )


            fig.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["trend"],
                    name="Trend",
                    line=dict(
                        dash="dash"
                    )
                )
            )



            for cp in changepoints:

                fig.add_vline(
                    x=pd.to_datetime(cp),
                    line=dict(
                        color="red",
                        dash="dot",
                        width=2
                    )
                )


            fig.update_layout(
                title=title,
                xaxis_title="Date",
                height=650
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )



        # --------------------------------------------------------
        # Controls
        # --------------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)



        with col1:

            changepoint_prior = st.slider(
                "Trend flexibility",
                0.01,
                0.5,
                0.15,
                0.01,
                help="Higher values allow more trend changes"
            )



        with col2:


            selection_mode = st.radio(
                "Selection mode",
                [
                    "Auto",
                    "Fixed number"
                ],
                horizontal=True
            )



        with col3:


            if selection_mode == "Auto":


                sensitivity = st.slider(
                    "Change importance percentile",
                    50,
                    99,
                    90,
                    step=1
                )


                forced_n = None



            else:


                forced_n = st.slider(
                    "Number of changepoints",
                    1,
                    20,
                    5,
                    step=1
                )


                sensitivity = None



        with col4:


            use_yearly = st.checkbox(
                "Yearly seasonality",
                False
            )



        # ========================================================
        # MULTIPLE VARIABLES
        # ========================================================

        if use_multi and len(selected_vars) > 1:


            st.info(
                "Prophet is univariate → one model per variable"
            )


            tabs = st.tabs(selected_vars)



            for idx, col in enumerate(selected_vars):


                with tabs[idx]:


                    df_prophet = (
                        df[[col]]
                        .reset_index()
                        .rename(
                            columns={
                                df.index.name or "index": "ds",
                                col: "y"
                            }
                        )
                    )



                    if time_granularity != "None":

                        df_prophet = (
                            df_prophet
                            .set_index("ds")
                            .resample(
                                resample_map[
                                    time_granularity
                                ]
                            )
                            .agg(
                                {
                                    "y":
                                    aggregation_method.lower()
                                }
                            )
                            .reset_index()
                        )



                    df_prophet["y"] = (
                        df_prophet["y"]
                        .ffill()
                        .bfill()
                    )



                    (
                        model,
                        forecast,
                        changepoints,
                        cp_table

                    ) = run_prophet_changeppoints(
                        df_prophet,
                        changepoint_prior,
                        sensitivity,
                        forced_n,
                        use_yearly
                    )



                    plot_changepoints(
                        df_prophet,
                        forecast,
                        changepoints,
                        f"Changepoints - {col}"
                    )


                    st.dataframe(
                        cp_table,
                        use_container_width=True
                    )



        # ========================================================
        # SINGLE VARIABLE
        # ========================================================

        else:


            df_prophet = (
                df[[selected_col]]
                .reset_index()
                .rename(
                    columns={
                        df.index.name or "index": "ds",
                        selected_col: "y"
                    }
                )
            )



            if time_granularity != "None":

                df_prophet = (
                    df_prophet
                    .set_index("ds")
                    .resample(
                        resample_map[
                            time_granularity
                        ]
                    )
                    .agg(
                        {
                            "y":
                            aggregation_method.lower()
                        }
                    )
                    .reset_index()
                )



            df_prophet["y"] = (
                df_prophet["y"]
                .ffill()
                .bfill()
            )



            (
                model,
                forecast,
                changepoints,
                cp_table

            ) = run_prophet_changepoints(
                df_prophet,
                changepoint_prior,
                sensitivity,
                forced_n,
                use_yearly
            )



            plot_changepoints(
                df_prophet,
                forecast,
                changepoints,
                "Changepoint Detection with Prophet"
            )



            if len(cp_table):

                st.write(
                    "### Detected Changepoints"
                )


                st.dataframe(
                    cp_table,
                    use_container_width=True
                )


            else:

                st.info(
                    "No significant changepoints detected"
                )

    

    elif analysis_section == "Trend Analysis":
        st.subheader("📈 Trend Analysis")

        if use_multi and len(selected_vars) > 1:
            facet_option = st.checkbox("📊 Use facetted trend plots", value=True)
            same_y_axis = st.checkbox("🔗 Same Y-axis scale for all facets", value=True) if facet_option else False

            rows = len(selected_vars)
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles=selected_vars, vertical_spacing=0.05)

            for i, col in enumerate(selected_vars, 1):
                y = df_summary[col].dropna()
                x_numeric = np.arange(len(y)).reshape(-1, 1)
                y_numeric = y.values.reshape(-1, 1)

                from sklearn.linear_model import LinearRegression
                from sklearn.preprocessing import PolynomialFeatures
                from statsmodels.tsa.seasonal import STL

                model_linear = LinearRegression().fit(x_numeric, y_numeric)
                trend_linear = model_linear.predict(x_numeric).flatten()

                poly = PolynomialFeatures(degree=2)
                x_poly = poly.fit_transform(x_numeric)
                model_poly = LinearRegression().fit(x_poly, y_numeric)
                trend_poly = model_poly.predict(x_poly).flatten()

                fig.add_trace(go.Scatter(x=y.index, y=y, mode='lines', name='Observed'), row=i, col=1)
                fig.add_trace(go.Scatter(x=y.index, y=trend_linear, mode='lines', name='Linear', line=dict(dash='dash')), row=i, col=1)
                fig.add_trace(go.Scatter(x=y.index, y=trend_poly, mode='lines', name='Polynomial', line=dict(dash='dot')), row=i, col=1)

            if same_y_axis:
                fig.update_yaxes(matches='y')
            fig.update_layout(height=300 * rows,
                              title_text=f"Trend Analysis – Multi Variable ({aggregation_method if resample_freq else 'Raw'})")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Original single-var trend code
            df_trend = df_summary[[selected_col]].copy()
            y = df_trend[selected_col]
            x_numeric = np.arange(len(y)).reshape(-1, 1)
            y_numeric = y.values.reshape(-1, 1)

            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import PolynomialFeatures
            from statsmodels.tsa.seasonal import STL

            model_linear = LinearRegression().fit(x_numeric, y_numeric)
            trend_linear = model_linear.predict(x_numeric).flatten()
            slope = model_linear.coef_[0][0]
            intercept = model_linear.intercept_[0]

            poly = PolynomialFeatures(degree=2)
            x_poly = poly.fit_transform(x_numeric)
            model_poly = LinearRegression().fit(x_poly, y_numeric)
            trend_poly = model_poly.predict(x_poly).flatten()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_trend.index, y=y, mode='lines', name=f'Observed ({aggregation_method})'))
            fig.add_trace(go.Scatter(x=df_trend.index, y=trend_linear, mode='lines', name='Linear Trend', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=df_trend.index, y=trend_poly, mode='lines', name='Polynomial Trend', line=dict(dash='dot')))

            fig.update_layout(
                title=f"Trend Analysis for {selected_col} ({aggregation_method} with {time_granularity} granularity)",
                xaxis_title="Date",
                yaxis_title=selected_col,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

    # (All other sections you had are fully preserved above)
