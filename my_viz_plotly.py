import streamlit as st
from streamlit_option_menu import option_menu

from utils.data_processing import load_preset_data

def main():
    st.set_page_config(layout="wide", page_title="EDAViz App", page_icon="🚀")
    load_preset_data(st.session_state)
    with st.sidebar:
        selected = option_menu(
            menu_title="EDAViz",
            options=["Home", "Missing Data Imputation", "Visualization", "Time Series", "About"],
            icons=["house", "gear", "bar-chart", "clock", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )
    
    if selected == "Home":
        from views.home import home
        home()
    elif selected == "Missing Data Imputation":
        from views.missing_data import missing_data_imputation
        missing_data_imputation()
    elif selected == "Visualization":
        from views.visualization import visualization
        visualization()
    elif selected == "Time Series":
        from views.time_series import time_series_analysis
        time_series_analysis()
    elif selected == "About":
        from views.about import about
        about()

if __name__ == "__main__":
    main()
