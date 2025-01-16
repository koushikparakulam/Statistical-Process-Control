# Statistical Process Control (SPC)

## Overview
This repository implements **Statistical Process Control (SPC) for detecting distributional shifts** in time-series data using machine learning techniques, including **LSTM models** and **probabilistic distance measures**. The project leverages **distributional analysis, data preprocessing, and deep learning models** to identify anomalies in streaming data. This was compared against more common CUSUM methods to analyze and compare probabilistic vs. predictive modeling.

## Features
- **Data Preprocessing**: Implements data batch makers for structured input.
- **Distributional Shift Detection**: Uses probabilistic methods (KL divergence, Wasserstein, KS test).
- **Deep Learning Model**: An **LSTM-based framework** for sequence-based anomaly detection.
- **Self-Supervised Learning**: Training pipeline based on simulated shifts.
- **Statistical Validation**: Compares models with traditional SPC techniques.
- **Automated Workflow**: Uses Pickle for intermediate data storage.

## Installation
To set up the project locally:
```bash
git clone https://github.com/koushikparakulam/Statistical-Process-Control.git
cd Statistical-Process-Control
pip install -r requirements.txt
```

## Usage
Run the main script to execute the pipeline:
```bash
python main.py
```
For training the LSTM model:
```bash
python Train_LSTM.py
```
For statistical analysis:
```bash
python Calculate_Statistics.py
```

## Project Structure
```
📂 Statistical-Process-Control
├── 📂 Pickled_Variables            # Serialized data for reuse
├── 📂 Project_Notes                # Documentation and workflow notes
├── 📜 All_Batch_Makers.py          # Batch maker for data preprocessing
├── 📜 Calculate_Statistics.py      # Statistical computation utilities
├── 📜 Create_Data_Stream.py        # Data stream pipeline implementation
├── 📜 Create_Train_Test_Validate.py# Data splitting for model training
├── 📜 Data_Module.py               # Data module for handling I/O
├── 📜 Data_Pickler.py              # Optimization for data serialization
├── 📜 Distributional_Shifts.py     # Detecting distributional shifts
├── 📜 Find_ROC.py                  # ROC curve computation
├── 📜 LSTM.py                      # LSTM model architecture
├── 📜 Norm_Shifts_Overlap_Based.py # Normalization-based shift detection
├── 📜 Test_LSTM.py                 # Unit tests for LSTM
├── 📜 Train_LSTM.py                # Training pipeline for LSTM
├── 📜 Truncate_Tensors.py          # Tensor truncation functions
├── 📜 main.py                      # Main execution script
└── 📜 README.md                    # Project documentation
```

## Key Components
### **Distributional Shift Detection**
- Uses **Gaussian KDE, Kolmogorov-Smirnov, Hellinger Distance, Wasserstein Distance** to quantify distributional changes.
- Compares pre-shift vs post-shift data using **simulated permutations**.

### **Deep Learning with LSTM**
- Implements a **sequence-based anomaly detection model** using **PyTorch**.
- Trained using **custom batch creation** and **adaptive learning rates**.

### **Statistical Analysis**
- Calculates various **probabilistic distances** to validate anomalies.
- Uses **ROC Curves** to compare model predictions with statistical baselines.


