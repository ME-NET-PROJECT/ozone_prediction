# Ozone Prediction with Deep Learning

This repository contains code for predicting ground ozone levels using various deep learning models such as LSTM, Bi-LSTM, GRU, Bi-GRU, and an ensemble model. The project uses time-series data to forecast air quality and evaluate multiple models' performance.

<table>
  <tr>
    <td style="text-align: center; font-weight: bold;">
      <img src="https://github.com/user-attachments/assets/5c6e6cf8-77e0-4db8-bb29-56e9046e5de6" alt="Actual Ozone on 2024-01-09" style="width: 100%;" />
      Actual Ozone (2024-01-09)
    </td>
    <td style="text-align: center; font-weight: bold;">
      <img src="https://github.com/user-attachments/assets/233e89dd-b7c9-4f8d-8112-46ee7f9b120a" alt="Predicted Ozone on 2024-01-09" style="width: 100%;" />
      Predicted Ozone (2024-01-09)
    </td>
  </tr>
  
  <tr>
    <td style="text-align: center; font-weight: bold;">
      <img src="https://github.com/user-attachments/assets/a4090b00-5251-4d62-a1d8-2a1617e6ac6d" alt="Actual Ozone on 2024-01-05" style="width: 100%;" />
      Actual Ozone (2024-01-05)
    </td>
    <td style="text-align: center; font-weight: bold;">
      <img src="https://github.com/user-attachments/assets/7a27d6ba-b8a5-4f53-83f2-bcf19a0bf669" alt="Predicted Ozone on 2024-01-05" style="width: 100%;" />
      Predicted Ozone (2024-01-05)
    </td>
  </tr>
</table>


## Overview

The models in this repository are designed to predict ozone levels using data collected from environmental monitoring stations. The primary models implemented are:

- **LSTM (Long Short-Term Memory)**
- **Bi-LSTM (Bidirectional Long Short-Term Memory)**
- **GRU (Gated Recurrent Unit)**
- **Bi-GRU (Bidirectional Gated Recurrent Unit)**
- **Ensemble Model** (combination of the above models)

The evaluation includes multiple performance metrics, including RMSE, MAE and  MAPE. The results are saved as CSV files for further analysis.

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
The results are based on both regression and classification metrics. In addition to predicting a specific ozone value, we categorise the ozone levels into four classes: low, medium, high, and very high, and present the corresponding results using classification metrics.

<table class="tg"><thead>
  <tr>
    <th class="tg-7btt" colspan="4">Regression Results</th>
    <th class="tg-7btt" colspan="4">Classifcation Results   <br>     (Low, Medium, High, Very High)</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-fymr">Model</td>
    <td class="tg-fymr">RMSE</td>
    <td class="tg-fymr">MAE</td>
    <td class="tg-fymr">MAPE</td>
    <td class="tg-fymr">Accuracy</td>
    <td class="tg-fymr">Precision</td>
    <td class="tg-fymr">Recall</td>
    <td class="tg-fymr">F1-Score</td>
  </tr>
  <tr>
    <td class="tg-0pky">bi_lstm</td>
    <td class="tg-c3ow">11.65</td>
    <td class="tg-c3ow">8.58</td>
    <td class="tg-c3ow">11.85</td>
    <td class="tg-c3ow">0.83</td>
    <td class="tg-c3ow">0.81</td>
    <td class="tg-c3ow">0.83</td>
    <td class="tg-c3ow">0.82</td>
  </tr>
  <tr>
    <td class="tg-0pky">bi_gru</td>
    <td class="tg-c3ow">11.89</td>
    <td class="tg-c3ow">8.62</td>
    <td class="tg-c3ow">12.06</td>
    <td class="tg-c3ow">0.83</td>
    <td class="tg-c3ow">0.82</td>
    <td class="tg-c3ow">0.83</td>
    <td class="tg-c3ow">0.82</td>
  </tr>
  <tr>
    <td class="tg-0pky">ensemble</td>
    <td class="tg-c3ow">11.78</td>
    <td class="tg-c3ow">8.65</td>
    <td class="tg-c3ow">12.07</td>
    <td class="tg-c3ow">0.84</td>
    <td class="tg-c3ow">0.82</td>
    <td class="tg-c3ow">0.84</td>
    <td class="tg-c3ow">0.83</td>
  </tr>
  <tr>
    <td class="tg-0pky">gru</td>
    <td class="tg-c3ow">11.91</td>
    <td class="tg-c3ow">8.77</td>
    <td class="tg-c3ow">12.14</td>
    <td class="tg-c3ow">0.83</td>
    <td class="tg-c3ow">0.80</td>
    <td class="tg-c3ow">0.83</td>
    <td class="tg-c3ow">0.79</td>
  </tr>
  <tr>
    <td class="tg-0pky">lstm</td>
    <td class="tg-c3ow">11.81</td>
    <td class="tg-c3ow">8.65</td>
    <td class="tg-c3ow">12.11</td>
    <td class="tg-c3ow">0.83</td>
    <td class="tg-c3ow">0.81</td>
    <td class="tg-c3ow">0.83</td>
    <td class="tg-c3ow">0.82</td>
  </tr>
</tbody></table>

Below is a visualisation of a 7 days prediction from 05/01/2024 to 11/01/2024.

![7 days_Prediction](https://github.com/user-attachments/assets/7be44e03-8fee-401c-a301-c54cd78dbbe4)

The results of the evaluations are saved in the following structure:

```
results/
    ├── lstm/
    │   ├── lstm_metrics.csv
    │   ├── lstm_predictions.csv
    │   ├── visualise/
    ├── bi_lstm/
    │   ├── bi_lstm_metrics.csv
    │   ├── bi_lstm_predictions.csv
    │   ├── visualise/
    ├── ensemble/
    │   ├── ensemble_metrics.csv
    │   ├── ensemble_predictions.csv
    │   ├── visualise/
    └── ...
```

### Metrics File: `model_name_metrics.csv`
Contains the evaluation metrics for each model (RMSE, MAE, MAPE).

### Predictions File: `model_name_predictions.csv`
Contains the predicted ozone values alongside true values for comparison.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
