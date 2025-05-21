import numpy as np
import pandas as pd
import streamlit as st
import dill
import threading
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
        st.session_state["y_true"] = data[target]
        class_values = np.unique(data[target])
        class_values_str = "\n".join([str(i) for i in class_values])
        res = st.text_area("Enter desired class value names on each line", value = class_values_str)
        
        class_names = res.split("\n")
        
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
    st.session_state["categorical_names"] = {}
    
    
    for i, key in enumerate(keys):
        for name in categorical_feature_names:
            if key == name:
                st.session_state["categorical_features"].append(i)
    
    if st.segmented_control("Assign Categorical Feature Class Values:", options=["Automatic", "Manual"], default="Automatic") == "Manual":
        st.write("For each categorical feature listed below, insert the class values associated with the numerical values. The current numerical values are listed. If no change is needed, keep them as is. Otherwise, replace each number with the respective value names")
        for idx, name in enumerate(categorical_feature_names):
            unique = np.unique(data[name])
            n_unique = len(unique)
            numerical_list = "\n".join([str(i) for i in unique])
            res = st.text_area(name, value = numerical_list)
            st.session_state["categorical_names"][st.session_state["categorical_features"][idx]] = res.split("\n")
    
    
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
        categorical_names=st.session_state["categorical_names"],
        categorical_features=st.session_state["categorical_features"])
    
    import warnings
    warnings.filterwarnings("ignore")
    
    X_test = st.session_state["data"]
    y_pred = st.session_state["model"].predict(X_test)
    
    sample_options = np.array([f"{i} - Prediction: {st.session_state["class_names"][y_pred[i]]}" for i in range(st.session_state["data"].shape[0])])
    option_idx = st.selectbox("Predicted Samples", options=sample_options)
    idx = np.where(sample_options == option_idx)[0]
    
    X_row = st.session_state["data"][idx].reshape(1, -1)
    
    
    # st.write('Prediction: ', explainer.class_names[model.predict(X_row)[0]])
    exp = explainer.explain_instance(X_row, model.predict, threshold=0.95)
    st.markdown("---")
    st.write("Anchor: ")
    for name in exp.names():
        st.write(name)
    st.markdown("---")
    st.write(f"Precision: {exp.precision():.2f}")
    st.write(f"Coverage: {exp.coverage():.2f}")

    st.markdown("---")
    
    def find_all_anchors():
        anchors = {
            "y_true": [],
            "y_pred": [],
            "precision": [],
            "coverage": []
        }
        
        # st.session_state["progress"] = st.progress(value = len(anchors["y_pred"]) / len(y_pred), text = "Finding Anchors...")
        # st.session_state["count"] = st.empty()

        
        for X_t, y_t, y_p in zip(X_test, st.session_state["y_true"], y_pred):
            
            
            exp = explainer.explain_instance(X_t, model.predict, threshold=0.95)
            anchors["y_true"].append(y_t)
            anchors["y_pred"].append(y_p)
            anchors["precision"].append(exp.precision())
            anchors["coverage"].append(exp.coverage())
            names = exp.names()
            
            iters = (len(anchors) - 4) if (len(anchors) - 4) > len(names) else len(names)
            for count in range(iters):
                if f"a{count}" not in anchors:
                    anchors[f"a{count}"] = [None for i in range(len(anchors["y_pred"])-1)]
                if len(names) > 0:
                    anchors[f"a{count}"].append(names[0])
                    names.pop(0)
                else:
                    anchors[f"a{count}"].append(None)
                # print(len(anchors[f"a{count}"]))
            
            print(f"{len(anchors['y_pred'])} / {len(y_pred)}")
        st.session_state["final_anchors"] = pd.DataFrame(anchors)
    
      
    
    st.button("Anchor Everything", on_click = find_all_anchors)
    placeholder = st.empty()  
    if "final_anchors" in st.session_state:
        placeholder.dataframe(st.session_state["final_anchors"])
