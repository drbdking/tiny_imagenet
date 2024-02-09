import json
import os


import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_data():
    data = []
    for filename in os.listdir("output/"):
        if filename.endswith(".json"): 
            print(filename)
            # with open(f"output/{filename}", "r") as fp:
            #     rec = json.load(fp)
            # data.append(pd.DataFrame(data=rec))
        else:
            continue
    # df = pd.concat(data)
    return df

def make_bubble_plot(df, method="max"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['mean'],
        ))

    fig = px.scatter(
        df,
        x="depth",
        y="width",
        color="Specie",
        size="num_params",
    ).update_traces(mode="markers")

    st.plotly_chart(fig)


if __name__ == '__main__':
    # Load output
    df = load_data()
    # Select box, choose max or quantile
    # Make bubble plot
    pass
    
