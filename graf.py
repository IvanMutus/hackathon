import csv
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import itertools
import base64
from io import BytesIO
from sklearn.metrics import confusion_matrix, classification_report
# import shap
# import streamlit.components.v1 as components


st.set_page_config(layout="wide")

@st.cache
def load_data():
    return shap.datasets.boston()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# st.title("SHAP in Streamlit")


def plot_params(df, track_id, code):
    
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(15,5))

    to_plot = df[(df.track_id == track_id)&(df.code==code)]
    
    i = 0
    cols = ['height', 'longitude', 'latitude']
    for ax in axes.flatten():
        ax.plot(to_plot.drop_duplicates('seconds')
                        .set_index('seconds')
                        .reindex(np.arange(min(to_plot.seconds), max(to_plot.seconds)),fill_value=0)[cols[i]], label=cols[i])
        i+=1
        ax.legend()
    return fig

def get_table_download_link(df_res):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df_res.to_csv(header=None, index=None, sep=' ', mode='a')
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/txt;base64,{b64}" download="result.txt">Download txt file</a>'
    return href

st.title('Вечность пахнет нефтью')

col1, col2 = st.beta_columns([4,20])
with col1:
    st.image('logo.jpg', width=100)
with col2:
    st.write('### Кейс ГосНИИАС "Анализ трековых данных воздушных судов"')


feature_imp = pd.read_csv('feature_importance.csv')

df_res = pd.read_csv('result.csv')
y_true = df_res.groupby(['name','track_id']).target.max()
y_pred = df_res.groupby(['name','track_id']).result.max()
cnf_matrix = confusion_matrix(y_true, y_pred)


st.write('### Скачать результаты:')
st.markdown(get_table_download_link(df_res), unsafe_allow_html=True)


col1, col2 = st.beta_columns([10,10])
with col1:
    c = alt.Chart(feature_imp.sort_values(by="Value", ascending=False)[0:30]).mark_bar().encode(
            x='Value',
            y=alt.Y("Feature", sort='-x'),
            color=alt.Color('Value', legend=alt.Legend(orient="bottom"))
        ).properties(height=500, width=500)

    st.altair_chart(c)
    
with col2:
    st.write('Confusion matrix: ', cnf_matrix)
    st.write('Метрики классификации: ') 
    st.write(classification_report(y_true, y_pred))


st.write('#### SHAP значения:')
st.image('summary_plot.jpg', width=800)
# fig = plot_params(df, track_id, code)
