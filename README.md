# Anchors-Streamlit

## Overview

A lightweight, interactive Streamlit application for generating Anchors explanations for tabular models using the anchors-env library. Designed for interpretability of black-box classifiers through high-precision, human-readable rules. 

More information about anchors here:  
- https://christophm.github.io/interpretable-ml-book/anchors.html
- https://homes.cs.washington.edu/~marcotcr/aaai18.pdf 



## Requirements/Features

Python package dependencies are listed in the `requirements.txt` file.

For Linux users, a quickstart script (`run.sh`) is provided to streamline setup. Users on other systems will need to install dependencies manually.

This application requires two inputs:

### CSV File
- Headers are required
- Numeric data formatted consistently with how the model was trained
- Can be train data, test data, or a combined set
- Loaded via `pandas`
- Must include *at minimum* all of the independent and dependent variables used during model training

### Binary Model File
- Must be a supervised classification model
- Currently tested only with `scikit-learn` models; other libraries *may* work but are unverified 
- Must be serialized using the [dill](https://pypi.org/project/dill/) library, not standard Python pickle or joblib. 
  - For help saving your model correctly, see [Saving Your Model](#saving-your-model)

Once you have both files prepared, you can use the app to quickly generate anchor-based explanations for your model’s predictions.

## Getting Started

First, clone this repository to your local machine. If you're unfamiliar with this process, see GitHub’s guide: [Cloning a Repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

### Quickstart (Linux/WSL)

If you are using Linux or Windows Subsystem for Linux, simply run the following command from the root of the repository

```bash
./run.sh
```

On first run, this script will set up a python virtual environment, install dependencies from `requirements.txt`, apply necessary modifications to the local `anchors-exp` library, and launch the Streamlit app. On subsequent runs, only the final step (launching the app) will be executed. 

### Manual Setup

#### 1. Create and run a Virtual Environment

Create and run a Python Virtual Environment from the root of the repository

##### macOS / Linux

```bash
python3 -m venv .
source bin/activate
```

##### macOS / Linux

```bat
python -m venv .
.\Scripts\activate
```

#### 2. Install Dependencies

Once the virtual environment is active, install all dependencies:

```bash
pip install -r requirements.txt
```

#### 3. Manually edit `anchor_tabular.py`

The `anchors-exp` library requires a small adjustment to use. I know this is kinda hacky, but just go with it for now.

- Mac/Linux path: `lib/python3.*/site-packages/anchor/anchor_tabular.py`
- Windows path `Lib\site-packages\anchor\anchor_tabular.py`

Open the file, and you will **manually modify the constructor of AnchorTabularExplainer**. The first is on the input parameters. Locate line 31, it should look something like this:

```python
...
31 def __init__(self, class_names, feature_names, train_data,
32               categorical_names={}, discretizer='quartile', encoder_fn=None):
33      self.min = {}
34      self.max = {}
...     
```

Add the parameter `categorical_features = []` so that it becomes:

```python
...
31 def __init__(self, class_names, feature_names, train_data,
32               categorical_names={}, categorical_features = [], discretizer='quartile', encoder_fn=None):
33      self.min = {}
34      self.max = {}
...     
```

Next, locate line 40. It should look something like this:

```python
...
40 self.categorical_features = []
41 self.feature_names = feature_names
42 self.train = train_data
...
```

Change `self.categorical_features = []` to `self.categorical_features = categorical_features` so that it looks like

```python
...
40 self.categorical_features = categorical_features 
41 self.feature_names = feature_names
42 self.train = train_data
...
```

And you're done!

#### 4. Run the app

Run the following command to start the app:
```bash
streamlit run src/app.py
```

Now you can start explaining your model.

## Usage

### CSV Settings

1. Upload CSV dataset with formatted data
2. Select the Target Feature, or the Dependent Variable
   - Optionally, manually set labels for each possible target value
3.  

### Saving Your Model

To save a model, it is required to use the [dill](https://pypi.org/project/dill/) library:

```python
import dill
dill.settings['recurse'] = True

# The file name/extension can be virtually anything
with open('path/model.modelfile', 'rb') as file:
    dill.dump(model, file)
```