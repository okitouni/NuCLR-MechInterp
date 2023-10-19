"""Module to generate streamlit app to visualize NuCLR embeddings."""
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="nuclr playground", page_icon="⚛")
st.markdown(
    "<h1 style='text-align: center;'>Mechanistic Interpretability of NuCLR</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h5 style='text-align: center;'>Nuclear Co-Learned Representations</h5>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Generate data and PCA
df = pd.read_csv("streamlit-data/nuclear_pca.csv")
df_z = pd.read_csv("streamlit-data/z_pca.csv")
df_n = pd.read_csv("streamlit-data/n_pca.csv")

df_t_sne = pd.read_csv("streamlit-data/nuclear_t_sne.csv")


# Dropdown to select visualization type
vis_type = st.selectbox(
    "Select a visualization",
    [
        "Proton (Z) Embeddings",
        "Neutron (N) Embeddings",
        "Nuclear Embeddings (PCA)",
        "Nuclear Embeddings (t-SNE)",
    ],
)

if vis_type == "Proton (Z) Embeddings":
    # Create a sidebar for user control inputs
    st.sidebar.header("Control Panel")
    st.sidebar.write("-----")

    # Z PCA Controls
    st.sidebar.write("### Proton (Z) Embeddings")

    # Create sliders for Z and N range
    z_min = st.sidebar.slider(
        "Minimum Z:", int(min(df_z["z"])), int(max(df_z["z"])), int(min(df_z["z"]))
    )
    z_max = st.sidebar.slider(
        "Maximum Z:", int(min(df_z["z"])), int(max(df_z["z"])), int(max(df_z["z"]))
    )
    # Filter DataFrame based on slider values
    filtered_df = df_z[(df_z["z"] >= z_min) & (df_z["z"] <= z_max)]

    st.sidebar.write("-----")

    # Slider Controls for 2D and 3D plots
    x_pc = st.sidebar.slider("X-axis PC:", 1, 10, 1)
    y_pc = st.sidebar.slider("Y-axis PC:", 1, 10, 2)
    z_pc = st.sidebar.slider("Z-axis PC for 3D plot:", 1, 10, 3)
    color = st.sidebar.slider("Color Scale PC:", 1, 10, 3)

    # Display the 2D scatter plot
    fig = px.scatter(
        filtered_df,
        x=f"PC{x_pc}",
        y=f"PC{y_pc}",
        hover_data=["z"],
        labels={"x": f"PC{x_pc}", "y": f"PC{y_pc}"},
        color=f"PC{color}",
        color_continuous_scale="rdbu",
        title="2D Scatter Plot of Proton (Z) Embeddings",
    )
    st.plotly_chart(fig)

    # Display the 3D scatter plot
    fig = px.scatter_3d(
        filtered_df,
        x=f"PC{x_pc}",
        y=f"PC{y_pc}",
        z=f"PC{z_pc}",
        hover_data=["z"],
        color=f"PC{color}",
        color_continuous_scale="rdbu",
        title="3D Scatter Plot of Proton (Z) Embeddings",
    )
    st.plotly_chart(fig)

elif vis_type == "Neutron (N) Embeddings":
    st.sidebar.header("Control Panel")
    st.sidebar.write("-----")

    st.sidebar.write("### Neutron (N) Embeddings")

    # Create sliders for Z and N range
    n_min = st.sidebar.slider(
        "Minimum N:", int(min(df_n["n"])), int(max(df_n["n"])), int(min(df_n["n"]))
    )
    n_max = st.sidebar.slider(
        "Maximum N:", int(min(df_n["n"])), int(max(df_n["n"])), int(max(df_n["n"]))
    )
    # Filter DataFrame based on slider values
    filtered_df = df_n[(df_n["n"] >= n_min) & (df_n["n"] <= n_max)]

    st.sidebar.write("-----")

    # Slider Controls for 2D and 3D plots for Neutron
    x_pc_n = st.sidebar.slider("X-axis PC:", 1, 10, 1)
    y_pc_n = st.sidebar.slider("Y-axis PC:", 1, 10, 2)
    z_pc_n = st.sidebar.slider("Z-axis PC for 3D plot:", 1, 10, 3)
    color_n = st.sidebar.slider("Color Scale PC:", 1, 10, 3)

    fig_n = px.scatter(
        filtered_df,
        x=f"PC{x_pc_n}",
        y=f"PC{y_pc_n}",
        hover_data=["n"],
        labels={"x": f"PC{x_pc_n}", "y": f"PC{y_pc_n}"},
        color=f"PC{color_n}",
        color_continuous_scale="rdbu",
        title="2D Scatter Plot of Neutron (N) Embeddings",
    )
    st.plotly_chart(fig_n)

    fig_n_3d = px.scatter_3d(
        filtered_df,
        x=f"PC{x_pc_n}",
        y=f"PC{y_pc_n}",
        z=f"PC{z_pc_n}",
        hover_data=["n"],
        color=f"PC{color_n}",
        color_continuous_scale="rdbu",
        title="3D Scatter Plot of Neutron (N) Embeddings",
    )
    st.plotly_chart(fig_n_3d)

elif vis_type == "Nuclear Embeddings (PCA)":
    # NUCLEAR PCA
    st.sidebar.header("Control Panel")

    st.sidebar.write("-----")
    st.sidebar.write("### Nuclear Embeddings")
    st.sidebar.write(
        "Output of first layer representation (concatenated Z, N, and Task)"
    )

    # Create sliders for Z and N range
    z_min = st.sidebar.slider(
        "Minimum Z:", int(min(df["z"])), int(max(df["z"])), int(min(df["z"]))
    )
    z_max = st.sidebar.slider(
        "Maximum Z:", int(min(df["z"])), int(max(df["z"])), int(max(df["z"]))
    )
    n_min = st.sidebar.slider(
        "Minimum N:", int(min(df["n"])), int(max(df["n"])), int(min(df["n"]))
    )
    n_max = st.sidebar.slider(
        "Maximum N:", int(min(df["n"])), int(max(df["n"])), int(max(df["n"]))
    )
    # Filter DataFrame based on slider values
    filtered_df = df[
        (df["z"] >= z_min)
        & (df["z"] <= z_max)
        & (df["n"] >= n_min)
        & (df["n"] <= n_max)
    ]

    st.sidebar.write("-----")
    # Sidebar Controls for Nuclear Embeddings in 2D and 3D
    x_pc_nuclear = st.sidebar.slider("X-axis PC:", 1, 10, 1)
    y_pc_nuclear = st.sidebar.slider("Y-axis PC:", 1, 10, 2)
    z_pc_nuclear = st.sidebar.slider("Z-axis PC for 3D plot:", 1, 10, 3)
    color_nuclear = st.sidebar.slider("Color Scale PC:", 1, 10, 3)

    # 2D scatter plot for Nuclear Embeddings
    fig_nuclear = px.scatter(
        filtered_df,
        x=f"PC{x_pc_nuclear}",
        y=f"PC{y_pc_nuclear}",
        hover_data=["z", "n"],
        labels={"x": f"PC{x_pc_nuclear}", "y": f"PC{y_pc_nuclear}"},
        color=f"PC{color_nuclear}",
        color_continuous_scale="rdbu",
        title="2D Scatter Plot of Nuclear Embeddings (PCA)",
    )
    st.plotly_chart(fig_nuclear)

    # 3D scatter plot for Nuclear Embeddings
    fig_nuclear_3d = px.scatter_3d(
        filtered_df,
        x=f"PC{x_pc_nuclear}",
        y=f"PC{y_pc_nuclear}",
        z=f"PC{z_pc_nuclear}",
        hover_data=["z", "n"],
        color=f"PC{color_nuclear}",
        color_continuous_scale="rdbu",
        title="3D Scatter Plot of Nuclear Embeddings (PCA)",
    )
    st.plotly_chart(fig_nuclear_3d)

elif vis_type == "Nuclear Embeddings (t-SNE)":
    mass_numbers = df_t_sne["z"] + df_t_sne["n"]
    df_t_sne["mass_number"] = mass_numbers
    # NUCLEAR PCA
    st.sidebar.header("Control Panel")

    st.sidebar.write("-----")
    st.sidebar.write("### Nuclear Embeddings")
    st.sidebar.write(
        "Output of first layer representation (concatenated Z, N, and Task)"
    )

    # Create sliders for Z and N range
    z_min = st.sidebar.slider(
        "Minimum Z:",
        int(min(df_t_sne["z"])),
        int(max(df_t_sne["z"])),
        int(min(df_t_sne["z"])),
    )
    z_max = st.sidebar.slider(
        "Maximum Z:",
        int(min(df_t_sne["z"])),
        int(max(df_t_sne["z"])),
        int(max(df_t_sne["z"])),
    )
    n_min = st.sidebar.slider(
        "Minimum N:",
        int(min(df_t_sne["n"])),
        int(max(df_t_sne["n"])),
        int(min(df_t_sne["n"])),
    )
    n_max = st.sidebar.slider(
        "Maximum N:",
        int(min(df_t_sne["n"])),
        int(max(df_t_sne["n"])),
        int(max(df_t_sne["n"])),
    )
    # Filter DataFrame based on slider values
    filtered_df = df_t_sne[
        (df_t_sne["z"] >= z_min)
        & (df_t_sne["z"] <= z_max)
        & (df_t_sne["n"] >= n_min)
        & (df_t_sne["n"] <= n_max)
    ]

    st.sidebar.write("-----")

    # 2D scatter plot for Nuclear Embeddings
    fig_nuclear = px.scatter(
        filtered_df,
        x="C1",
        y="C2",
        hover_data=["z", "n"],
        labels={"x": "C1", "y": "C2"},
        color="mass_number",
        color_continuous_scale="rdbu",
        title="2D Scatter Plot of Nuclear Embeddings (t-SNE)",
    )
    fig_nuclear.update_traces(marker_size=5)
    st.plotly_chart(fig_nuclear)
