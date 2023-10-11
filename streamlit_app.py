import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Generate data and PCA
df = pd.read_csv("streamlit-data/nuclear_pca.csv")
df_z = pd.read_csv("streamlit-data/z_pca.csv")
df_n = pd.read_csv("streamlit-data/n_pca.csv")


st.title("Select PCA components to plot")

# Z PCA
st.write("## Proton (Z) Embeddings")
x_pc = st.slider("Select Z embeddings PC for X-axis:", 1, 10, 1)
y_pc = st.slider("Select Z embeddings PC for Y-axis:", 1, 10, 2)
color = st.slider("Select Z embeddings PC for color:", 1, 10, 3)

fig = px.scatter(
    df_z,
    x=f"PC{x_pc}",
    y=f"PC{y_pc}",
    hover_data=["z"],
    labels={"x": f"PC{x_pc}", "y": f"PC{y_pc}"},
    color=f"PC{color}",
    color_continuous_scale="rdbu",
)
st.plotly_chart(fig)

z_pc = st.slider("Select Z embeddings PC for Z-axis:", 1, 10, 3)

fig = px.scatter_3d(
    df_z,
    x=f"PC{x_pc}",
    y=f"PC{y_pc}",
    z=f"PC{z_pc}",
    hover_data=["z"],
    color=f"PC{color}",
    color_continuous_scale="rdbu",
)
st.plotly_chart(fig)

# N PCA
st.write("## Neutron (N) Embeddings")
x_pc = st.slider("Select N embeddings PC for X-axis:", 1, 10, 1)
y_pc = st.slider("Select N embeddings PC for Y-axis:", 1, 10, 2)
color = st.slider("Select N embeddings PC for color:", 1, 10, 3)

fig = px.scatter(
    df_n,
    x=f"PC{x_pc}",
    y=f"PC{y_pc}",
    hover_data=["n"],
    labels={"x": f"PC{x_pc}", "y": f"PC{y_pc}"},
    color=f"PC{color}",
    color_continuous_scale="rdbu",
)
st.plotly_chart(fig)

z_pc = st.slider("Select N embeddings PC for Z-axis:", 1, 10, 3)

fig = px.scatter_3d(
    df_n,
    x=f"PC{x_pc}",
    y=f"PC{y_pc}",
    z=f"PC{z_pc}",
    hover_data=["n"],
    color=f"PC{color}",
    color_continuous_scale="rdbu",
)
st.plotly_chart(fig)

# NUCLEAR PCA
st.write("## Nuclear Embeddings")
st.write("Output of first layer representation (concatenated Z, N, and Task)")
x_pc = st.slider("Select PC for X-axis:", 1, 10, 1)
y_pc = st.slider("Select PC for Y-axis:", 1, 10, 2)
color = st.slider("Select PC for color:", 1, 10, 3)

fig = px.scatter(
    df,
    x=f"PC{x_pc}",
    y=f"PC{y_pc}",
    hover_data=["z", "n"],
    labels={"x": f"PC{x_pc}", "y": f"PC{y_pc}"},
    color=f"PC{color}",
    color_continuous_scale="rdbu",
)
st.plotly_chart(fig)

z_pc = st.slider("Select PC for Z-axis:", 1, 10, 3)

fig = px.scatter_3d(
    df,
    x=f"PC{x_pc}",
    y=f"PC{y_pc}",
    z=f"PC{z_pc}",
    hover_data=["z", "n"],
    color=f"PC{color}",
    color_continuous_scale="rdbu",
)
st.plotly_chart(fig)
