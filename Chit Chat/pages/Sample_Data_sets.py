import streamlit as st
import pandas as pd
st.header("Download sample data to chit chat")
df = pd.read_csv("Datasets/100 Sales Records.csv")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='sample_data.csv',
    mime='text/csv'
)