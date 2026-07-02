# Boston House Price Prediction

## Objective

The objective of this project is to build a regression model that predicts Boston house prices using attributes such as number of rooms, crime rate, tax value, pupil-teacher ratio, and distance-related features.

## Data Preprocessing

The dataset is loaded from a CSV file. Duplicate rows are removed. Missing values are handled using:

- Median for numerical columns
- Most frequent value for categorical columns

Outliers in the target column are removed using the IQR technique.

## Model Selection

The following regression models are trained and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Random Forest is often effective for this task because it can learn non-linear relationships between input features and house prices.

## Training

The dataset is split into training and testing sets. Each model is trained using the training data and tested using unseen test data.

Training command:

```powershell
python src/train.py --csv data/BostonHousing.csv
```

## Evaluation

The models are evaluated using:

- Mean Squared Error
- Root Mean Squared Error
- Mean Absolute Error
- R2 Score

The model with the highest R2 score is selected as the best model and saved for prediction.

## Fine-Tuning

Basic hyperparameters are used in the Decision Tree and Random Forest models. Performance can be improved by tuning:

- Number of estimators
- Maximum tree depth
- Minimum samples split
- Minimum samples leaf

## Conclusion

This project demonstrates a complete machine learning regression workflow, including preprocessing, model selection, training, evaluation, and prediction. The final model can help estimate house prices from housing-related features.

