# Boston House Price Prediction

This project trains regression models to predict house prices using housing attributes such as number of rooms, crime rate, tax rate, and other related features.

## Objective

Design and implement a regression model to predict Boston house prices accurately.

The project includes:

- Data loading from CSV
- Missing value handling
- Outlier removal using the IQR method
- Feature preprocessing
- Model training
- Model evaluation using MSE, RMSE, MAE, and R2 score
- Saving the best trained model
- Predicting prices for new input data

## Project Structure

```text
boston-house-price-prediction/
  README.md
  SUBMISSION.md
  requirements.txt
  .gitignore
  src/
    train.py
    predict.py
  data/
    README.md
  models/
    README.md
```

## Dataset Format

Place your dataset at:

```text
data/BostonHousing.csv
```

The CSV must contain a target column named one of:

```text
MEDV, medv, Price, price, target, Target
```

If no dataset is provided, the training script uses a small synthetic demo dataset so the workflow can still run.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Model

With your CSV:

```powershell
python src/train.py --csv data/BostonHousing.csv
```

Without a CSV, using demo data:

```powershell
python src/train.py
```

The trained model is saved to:

```text
models/boston_house_price_model.pkl
```

## Predict a Price

Example:

```powershell
python src/predict.py --features "{\"RM\": 6.5, \"CRIM\": 0.2, \"TAX\": 300, \"PTRATIO\": 15, \"DIS\": 4.0}"
```

## Evaluation Metrics

The project reports:

- Mean Squared Error
- Root Mean Squared Error
- Mean Absolute Error
- R2 Score

