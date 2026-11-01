import streamlit as st

from edalens.utils.data_processing import load_preset_data
from edalens.utils.ui import sidebar_menu

def main():
    st.set_page_config(layout="wide", page_title="EDALens App", page_icon="🚀")
    load_preset_data(st.session_state)
    pages = ["Load data", "Missing Data Imputation", "Visualization", "Time Series", "About"]
    with st.sidebar:
        selected = sidebar_menu(
            "EDALens", pages, icons=["house", "gear", "bar-chart", "clock", "info-circle"], menu_icon="cast"
        )
    
    if selected == "Load data":
        from edalens.views.home import home
        home()
    elif selected == "Missing Data Imputation":
        from edalens.views.missing_data import missing_data_imputation
        missing_data_imputation()
    elif selected == "Visualization":
        from edalens.views.visualization import visualization
        visualization()
    elif selected == "Time Series":
        from edalens.views.time_series import time_series_analysis
        time_series_analysis()
    elif selected == "About":
        from edalens.views.about import about
        about()

if __name__ == "__main__":
    main()
