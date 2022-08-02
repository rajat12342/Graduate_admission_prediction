import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page

page = st.sidebar.selectbox("Explore data or Prediction tool", ('Explore data', 'Prediction tool'))

if page == 'Prediction tool':
    show_predict_page()
else:
    show_explore_page()