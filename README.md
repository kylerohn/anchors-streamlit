# Anchors-Streamlit

## Overview

A lightweight, interactive Streamlit application for generating Anchors explanations for tabular models using the anchors-env library. Designed for interpretability of black-box classifiers through high-precision, human-readable rules. 

More information about anchors here:  
- https://christophm.github.io/interpretable-ml-book/anchors.html
- https://homes.cs.washington.edu/~marcotcr/aaai18.pdf 



## Requirements/Features

Python package requirements are found in the requirements.txt file. There is a quickstart script, `run.sh` which will work on linux systems, otherwise manual installation and setup is required.

This application has two input requirements, a **csv file** and a **binary model file**. Here are the specifications for each:

### CSV File
- Headers are required
- Numeric data formatted consistently with how the model was trained
- Can be train data, test data, or a combined set
- Loaded via `pandas`
- Must include *at minimum* all of the independent and dependent variables used during model training

### Binary Model File
- Supervised Classification Model
- This application has only been tested with `scikit-learn` classification models, but it *may* work with similar libraries 
- The model *is not* saved with the built-in serialization methods, but rather with the [dill](https://pypi.org/project/dill/) python library. Instructions can be found [here](#saving-your-model)

Once you have these items prepared, you are able to utilize this application to streamline the process of finding anchors for your model.

This application is built solely for classification models on tabular datasets. 


### Saving Your Model

Assuming you have a trained 