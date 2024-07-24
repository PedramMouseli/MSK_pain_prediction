[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PedramMouseli/MSK_pain_prediction/blob/main/notebooks/MSK_pain_prediction.ipynb)
# A Predictive Model of Experimentally Evoked Acute Musculoskeletal (MSK) Facial Pain Based on Functional Muscle Recordings

## Overview

This repository contains the code and models developed for predicting post-MVC pain using features extracted from EMG and NIRS signals recorded from the masseter muscle during a repetitive clenching task. The project aims to provide a reliable predictive model of acute MSK facial pain based on functional muscle recordings.
![Study_design](data/figures/Figure1.png)

## Features

- **Feature Extraction**: Code to extract relevant features from EMG and NIRS signals.
- **Model Training**: Scripts and models to predict pain levels using the extracted features.
- **Jupyter Notebook**: An interactive notebook for running the feature extraction and model training. 

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Jupyter Notebook
- Required Python packages (listed in `requirements.txt`)

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/PedramMouseli/MSK_pain_prediction.git
cd MSK_pain_prediction
```
### Usage

#### Feature Extraction and Model Training

You can use the provided Jupyter notebook to extract features and train the models. The notebook is designed to be run locally or on Google Colab.

To run the notebook on Google Colab, click the link below:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PedramMouseli/MSK_pain_prediction/blob/main/notebooks/MSK_pain_prediction.ipynb)

#### Running Locally

To run the Jupyter notebook locally:

```bash
jupyter notebook
```
Open the notebooks/MSK_pain_prediction.ipynb notebook and follow the instructions inside.

### License
This project is licensed under the GNU General Public License v3.0 License - see the LICENSE file for details.
