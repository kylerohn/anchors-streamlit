import numpy as np
import pandas as pd
import streamlit as st
import dill
from anchor import anchor_tabular as atx
        

# must be a processed csv with training data and column headers
datafile = st.file_uploader("Select Data (csv)", type="csv")

if datafile:
# load data and extract df keys
    data = pd.read_csv(datafile)
    keys = data.keys().to_list()
    
    st.markdown("## Target Feature Selection")
    
    target = st.selectbox("Target Feature: ",
                options=keys)
    
    if target:
        class_values = np.unique(data[target])
        class_names = st.text_input("Enter class value names, delimited by commas").split(",")
        
        if len(class_values) != len(class_names):
            st.write("Incorrect number of class names")
        else:
            st.write(np.array(class_names))
            st.session_state["class_names"] = class_names
    
    keys.remove(target)
    
    st.markdown("## Input Feature Selection")
    is_all = st.segmented_control(label=" ", options=["All", "Select"], default="All")
    
    if is_all == "All":
        feature_names = keys
    else:
        feature_names = st.multiselect("Input Features: ",
                                        options=keys)
    if feature_names:
        st.write("Selected Features", np.array(feature_names))
        st.session_state["feature_names"] = feature_names
        
    st.session_state["data"] = data[feature_names].to_numpy()
    
    categorical_feature_names = st.multiselect("Select Categorical Features. The rest will be discretized", options=keys)
    
    st.session_state["categorical_features"] = []
    
    for i, key in enumerate(keys):
        for name in categorical_feature_names:
            if key == name:
                st.session_state["categorical_features"].append(i)


modelfile = st.file_uploader("Select Model")
if modelfile:
    with open("./temp/temp_model.modelfile", "wb") as f:
        f.write(modelfile.getvalue())
    with open("./temp/temp_model.modelfile", "rb") as f:
        model = dill.load(f)

    r, c = st.session_state["data"].shape
            
    st.session_state["model"] = model

if "class_names" in st.session_state and "feature_names" in st.session_state and "data" in st.session_state and "categorical_features" in st.session_state and "model" in st.session_state:
    st.write("Analyze Model with Anchors")
    
    explainer = atx.AnchorTabularExplainer(
        st.session_state["class_names"],
        st.session_state["feature_names"],
        st.session_state["data"],
        categorical_features=st.session_state["categorical_features"])
    
    idx = st.selectbox("Data index", options=[i for i in range(r)])
    
    X_test = st.session_state["data"][idx].reshape(1, -1)
    
    st.write('Prediction: ', explainer.class_names[model.predict(X_test)[0]])
    exp = explainer.explain_instance(X_test, model.predict, threshold=0.95)
    st.markdown("---")
    st.write("Anchor: ")
    for name in exp.names():
        st.write(name)
    st.markdown("---")
    st.write(f"Precision: {exp.precision():.2f}")
    st.write(f"Coverage: {exp.coverage():.2f}")



