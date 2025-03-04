# Air Quality Prediction with Deep Learning

https://doi.org/10.5281/zenodo.14960648

This repository contains code for predicting ground ozone levels using various deep learning models such as LSTM, Bi-LSTM, GRU, Bi-GRU, and an ensemble model. The project leverages time-series data to forecast air quality and evaluate multiple models' performance.

## Overview

The models in this repository are designed to predict ozone levels using data collected from environmental monitoring stations. The primary models implemented are:

- **LSTM (Long Short-Term Memory)**
- **Bi-LSTM (Bidirectional Long Short-Term Memory)**
- **GRU (Gated Recurrent Unit)**
- **Bi-GRU (Bidirectional Gated Recurrent Unit)**
- **Ensemble Model** (combination of the above models)

The evaluation includes multiple performance metrics, including RMSE, MAE, MAPE, and R-squared. The results are saved as CSV files for further analysis.

## Requirements

To run the code, you need to install the following dependencies:

- `TensorFlow`
- `NumPy`
- `Pandas`
- `Matplotlib`
- `Scikit-learn`

You can install them by running:

```bash
pip install -r requirements.txt
```

## Functionality

### 1. Model Evaluation: `evaluate_all_models()`
This function evaluates several pre-trained models (LSTM, Bi-LSTM, GRU, Bi-GRU) on test data (`X_test`, `y_test`) and stores performance metrics and predictions. The metrics include:

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **MAPE (Mean Absolute Percentage Error)**
- **R-squared** 

It also saves the prediction results in CSV files and generates plots comparing predicted vs true values.

### 2. Training the Ensemble Model: `train_ensemble_model()`
This function trains an ensemble model by combining the predictions of the individual models (LSTM, Bi-LSTM, GRU, Bi-GRU). It utilizes a custom training loop with callbacks for early stopping and learning rate adjustments.

### 3. Ensemble Model Evaluation: `evaluate_ensemble()`
After training the ensemble model, this function evaluates its performance using the same metrics as the individual models. It saves the predictions and evaluation metrics to CSV files.

### 4. Decile-Based Classification: `get_bin_numbers()`
This function discretizes the predicted and actual ozone values into bins (deciles). It calculates classification metrics (Accuracy, Precision, Recall, F1-Score) based on the decile assignments, making it easier to assess how well the models perform in terms of classifying different ozone levels.

## Usage

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-repo/ozone-prediction.git
cd ozone-prediction
```

### Step 2: Install Dependencies

Make sure to install the required Python libraries:

```bash
pip install -r requirements.txt
```

### Step 3: Training and Evaluation

Run the model training and evaluation script. For example:

```python
output_dir = Path(dataset_folder) / Path(model_dataset) / Path('results')
evaluate_all_models(models_folder, X_test, y_test, X_test_meta, output_dir)
```

### Step 4: Ensemble Model Training and Evaluation

To train and evaluate the ensemble model, use:

```python
model_fns = [lstm_model, bi_lstm_model, gru_model, bi_gru_model]
history = train_ensemble_model(X_train, y_train, X_test, y_test, sequence_length, models_folder, model_fns, epochs=500)
evaluate_ensemble(models_folder, X_test, y_test, X_test_meta, output_dir, model_name="ensemble")
```

### Step 5: Classification Metrics

Once the deciles have been calculated for both true and predicted values, classification metrics (Accuracy, Precision, Recall, F1-Score) are computed and saved as CSV files.

```python
# Apply deciles and compute classification metrics
df['Predicted_Value_Decile'] = df['Predicted_Value'].apply(get_bin_numbers)
df['True_Value_Decile'] = df['True_Value'].apply(get_bin_numbers)
```

## Results

The results of the evaluations are saved in the following structure:

```
results/
    ├── lstm/
    │   ├── lstm_metrics.csv
    │   ├── lstm_predictions.csv
    │   └── lstm_predictions.pdf
    ├── bi_lstm/
    │   ├── bi_lstm_metrics.csv
    │   ├── bi_lstm_predictions.csv
    │   └── bi_lstm_predictions.pdf
    ├── ensemble/
    │   ├── ensemble_metrics.csv
    │   ├── ensemble_predictions.csv
    │   └── ensemble_predictions.pdf
    └── ...
```

### Metrics File: `model_name_metrics.csv`
Contains the evaluation metrics for each model (RMSE, MAE, MAPE, R-squared).

### Predictions File: `model_name_predictions.csv`
Contains the predicted ozone values alongside true values for comparison.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
