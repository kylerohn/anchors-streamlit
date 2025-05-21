#!/bin/bash
if [ ! -f ./pyvenv.cfg ]; then
    python3 -m venv .
fi

source ./bin/activate
if [ $? -neq 0 ]; then
    echo "Virtual environment could not be activated."
    exit
fi

expected=$(cat requirements.txt)
actual=$(pip freeze)

if [ "$expected" != "$actual" ]; then
    pip install -r requirements.txt
fi

anchor_tabular_file=./lib/python3.*/site-packages/anchor/anchor_tabular.py

# add input parameter to AnchorTabularExplainer constructor
if [ -f $anchor_tabular_file ]; then
    if ! grep -q "categorical_features = \[\]," $anchor_tabular_file; then
        sed -i 's/categorical_names={},/categorical_names={}, categorical_features = [],/' $anchor_tabular_file
    fi

    if grep -q "self.categorical_features = \[\]" $anchor_tabular_file; then
        sed -i 's/self\.categorical_features = []/self\.categorical_features = categorical_features/' $anchor_tabular_file
    fi
else
    echo "Missing File: $anchor_tabular_file"
    exit 1
fi


streamlit run src/app.py