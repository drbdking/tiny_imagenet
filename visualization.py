import json
import os


import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_data(type=0):
    data = []
    for filename in os.listdir(f"output/type{type}"):
        if filename.endswith(".json"): 
            with open(f"output/type{type}/{filename}", "r") as fp:
                rec = json.load(fp)
            data.append(pd.DataFrame(data=rec))
        else:
            continue
    df = pd.concat(data)
    return df

def make_bubble_plot(df, method="max"):
    df = df.groupby(['depth', 'width', 'num_params'], as_index=False).quantile(0.75)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=df['depth'],
            y=df['width'],
            marker_size=(df['num_params'] / 10000) ** 0.5,
            marker_color=df['val_acc'],
            hovertext=df['num_params'],
            mode="markers",
            marker_colorscale="Viridis",
            marker_colorbar=dict(
                title="Colorbar"
            ),
        ))
    fig.update_yaxes(type="log")
    fig.update_layout(
    yaxis={
        "tickmode": "array",
        "tickvals": [16, 32, 64, 128, 256, 512, 1024, 2048]
    }
)
    st.plotly_chart(fig)

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #         x=df['num_params'],
    #         y=df['val_acc'],
    #         mode="markers",
    #         marker_color=df['depth'],
    #         marker_colorscale="Viridis",
    #         marker_colorbar=dict(
    #             title="Colorbar"
    #         ),
    #     ))
    # fig.update_xaxes(type="log")
#     fig.update_layout(
#     xaxis={
#         "tickmode": "array",
#         "tickvals": np.linspace(1, 20, 20) * 1000000
#     }
# )
    fig = px.line(df, x="num_params", y="val_acc", color='depth', color_discrete_sequence= px.colors.qualitative.Prism)
    st.plotly_chart(fig)



if __name__ == '__main__':
    # Load output
    df = load_data(type=0)
    make_bubble_plot(df)
    df1 = load_data(type=1)
    make_bubble_plot(df1)
    # Select box, choose max or quantile
    # Make bubble plot

    
