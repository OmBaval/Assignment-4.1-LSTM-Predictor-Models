# LSTM Regression for IoT Time-Series Prediction

## Overview
This project implements a **Long Short-Term Memory (LSTM)** neural network to perform **regression on multivariate IoT time-series data**. The objective is to predict future values of household electricity consumption using historical sensor readings.

The notebook demonstrates how deep learning models can effectively capture temporal dependencies that traditional regression techniques fail to model.

---

## Problem Statement
Time-series data from IoT systems exhibit strong temporal dependencies. Classical regression models treat observations as independent and therefore underperform. This project uses an **LSTM-based Recurrent Neural Network (RNN)** to model sequential patterns and predict a continuous-valued target variable.

---

## Dataset
- **Type**: Multivariate time-series
- **Domain**: Household electricity consumption
- **Granularity**: Fixed time intervals
- **Features**: Power consumption and sensor-related variables
- **Target**: Future electricity consumption value

> The dataset is assumed to be pre-cleaned and stored as a CSV file before modeling.

---

## Methodology

### 1. Data Preparation
- Load the cleaned dataset
- Normalize features using Min-Max scaling
- Transform the time-series into supervised learning format
- Create sliding window sequences:
  - **Input**: Previous `n` timesteps
  - **Output**: Next timestep value

Assertions are used to ensure correct tensor shapes and prevent silent preprocessing errors.

---

### 2. Model Architecture
- **Model Type**: LSTM (Many-to-One)
- **Layers**:
  - LSTM layer(s)
  - Dense output layer
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam

The architecture is designed specifically for **regression tasks**, not classification.

---

### 3. Training
- Train–validation split
- Epoch-based training
- Monitoring of training and validation loss
- Visualization of loss curves to detect overfitting

---

### 4. Evaluation
- Compare predicted vs actual values
- Regression error analysis
- Visual inspection of prediction performance over time

---

## Results & Observations
- LSTM effectively captures temporal dependencies in the data
- Training loss converges with sufficient epochs
- Model performance is sensitive to:
  - Sequence length
  - Feature scaling
  - Overfitting without regularization

---

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - NumPy
  - Pandas
  - Matplotlib
  - Scikit-learn
  - TensorFlow / Keras


---

## Key Learning Outcomes
- Converting time-series data into supervised learning format
- Implementing LSTM models for regression
- Understanding temporal sequence modeling
- Evaluating deep learning models for continuous outputs
- Applying defensive checks in ML pipelines

---

## Limitations
- Single-step forecasting only
- No hyperparameter tuning
- Assumes stationary and clean data
- No deployment or real-time inference

---

## Future Improvements
- Multi-step time-series forecasting
- GRU and Transformer-based comparisons
- Attention mechanisms
- Hyperparameter optimization
- Model deployment as an API service

---

## Author
**Om Baval**  
LSTM Regression Assignment  
Applied Artificial Intelligence
