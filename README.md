# Spam Detection Using Natural Language Processing

Building a spam detection algorithm by utilizing Natural Language Processing to extract features associated with spam emails. Deep Learning methods as well as word-to-vector transformation are used to create a spam email classifier.

## Usage Instructions

Potential run.py arguments:
* data: downloads and populates data folder from source.
* build: feature extraction and bi-LSTM model building, requires data to be run.
* predict: runs script to predict phrases whether it is a spam or not, requires data and build to be run.

## Project Contents

```
ROOT FOLDER
├── .gitignore
├── .gitmodules
├── AutoPhrase (forked submodule repository)
├── run.py
├── README.md
├── data (populated by calling data argument of run.py)
├── models (populated by calling build argument of run.py)
│   ├── model
│   │   └── ...
│   └── tokenizer.pickle
├── notebooks
│   └── report.ipynb
└── src
    └── generate_dataset.py
    └── process_build.py
    └── spam_or_not.py
```

### `src`

* `generate_dataset.py`: Code that pulls dataset used for training from data source, and combines data into a dataframe.
* `process_build.py`: Code that extracts features from data, processes data to be used for training deep learning models. Trains Bidirectional Long Short-Term Memory model.
* `spam_or_not.py`: Code that loads model from process_build and runs a script to predict input text's probability of being a spam message.


### `notebooks`

* Jupyter notebooks for Reports and line-by-line executed code.