"""
To run the app
$ streamlit run app.py
"""

from cv2 import COLOR_RGB2BGR
from pandas.core.frame import DataFrame
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from graph_reader import GraphReader

import base64


def get_table_download_link(df: DataFrame) -> str:
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False, sep=";")
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = (
        f'<a href="data:file/csv;base64,{b64}" download="graph_datapoints.csv">Datapoints</a>        '
        "     (right-click and save as &lt;some_name&gt;.csv)"
    )
    return href

def display_result(graph_reader, graph):
    result_df = pd.DataFrame()
    for i, (x, y) in enumerate(zip(graph_reader.x_points, graph_reader.y_points)):
        result_df = pd.concat([result_df, pd.DataFrame({f"x_{i+1}": x}), pd.DataFrame({f"y_{i+1}": y})], axis=1)
    # result_x = {f"x_{i+1}": y for i, y in enumerate(graph_reader.x_points)}
    # result_y = {f"y_{i+1}": y for i, y in enumerate(graph_reader.y_points)}
    # result_df = pd.DataFrame({**result_x, **result_y})
    st.markdown(get_table_download_link(result_df), unsafe_allow_html=True)
    st.image(graph)
    result_image = graph_reader.get_graph_image()
    st.image(result_image)


st.title("Graph Reader")
graph_reader = GraphReader()
automatic_detection = st.radio("Use Automatic detection or give min/max?", ("Automatic", "Manual"))
if automatic_detection == "Manual":
    with st.form(key="manual_axis"):
        cols = st.columns(2,)
        inputs = {}
        inputs["x_min_value"] = st.number_input(label="X_min - first tick", key=10, value=-3)
        inputs["x_max_value"] = st.number_input(label="X_max - last tick", key=11, value=3)
        inputs["y_min_value"] = st.number_input(label="Y_min - lowest tick", key=12, value=-1)
        inputs["y_max_value"] = st.number_input(label="Y_max - highest tick", key=13, value=1)

        submit_button = st.form_submit_button(label="Read chart")

type_chart = st.radio("Type of graph?", ("Matplotlib", "Other(not supported yet)"))
nb_sample_points = int(st.number_input(label="How many sample points?", min_value=10, max_value=100_000, step=1, value=100))

# import graph
graph_file = st.file_uploader("Import your graph (jpg, jpeg or png)")

if graph_file and automatic_detection == "Automatic":
    graph = np.array(Image.open(graph_file))
    # Convert to BGR as the graph reader expect OpenCV BGR format
    graph = cv2.cvtColor(graph, COLOR_RGB2BGR)
    graph_reader.from_image(graph, nb_points=nb_sample_points, automatic_scale=True)
    display_result(graph_reader, graph)



if automatic_detection == "Manual" and submit_button:
    graph = np.array(Image.open(graph_file))
    graph_reader.from_image(
        graph,
        nb_points=nb_sample_points,
        automatic_scale=False,
        x_min_value=inputs["x_min_value"],
        x_max_value=inputs["x_max_value"],
        y_min_value=inputs["y_min_value"],
        y_max_value=inputs["y_max_value"],
    )
    display_result(graph_reader, graph)

# else:
#     st.warning("Please upload the chart first")
