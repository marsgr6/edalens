import streamlit as st

from edalens.utils.data_processing import load_preset_data
from edalens.utils.ui import export_settings, sidebar_menu

def main():
    st.set_page_config(layout="wide", page_title="EDALens App", page_icon="🚀")
    published = st.query_params.get("report")
    if published:  # A link from "Publish link" on the Report page.
        from edalens.views.report import show_published
        show_published(published)
        return
    load_preset_data(st.session_state)
    pages = ["Load data", "Missing Data Imputation", "Statistics", "Visualization", "Time Series", "Report", "About"]
    with st.sidebar:
        selected = sidebar_menu(
            "EDALens", pages, icons=["house", "gear", "calculator", "bar-chart", "clock", "journal-richtext", "info-circle"], menu_icon="cast"
        )
        if selected in ("Statistics", "Visualization", "Time Series", "Report"):
            export_settings()
        from edalens.views.session import session_panel
        session_panel()
    
    if selected == "Load data":
        from edalens.views.home import home
        home()
    elif selected == "Missing Data Imputation":
        from edalens.views.missing_data import missing_data_imputation
        missing_data_imputation()
    elif selected == "Statistics":
        from edalens.views.statistics import statistics
        statistics()
    elif selected == "Visualization":
        from edalens.views.visualization import visualization
        visualization()
    elif selected == "Time Series":
        from edalens.views.time_series import time_series_analysis
        time_series_analysis()
    elif selected == "Report":
        from edalens.views.report import report_page
        report_page()
    elif selected == "About":
        from edalens.views.about import about
        about()

if __name__ == "__main__":
    main()
