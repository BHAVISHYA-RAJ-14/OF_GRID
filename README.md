# Real Estate Price Prediction (High Accuracy)

## Project Overview
This project leverages a Random Forest Regressor with advanced feature engineering to predict house prices. It achieves high accuracy by using log-transformations, interaction terms, and target encoding for location data.

## Key Features
- **Feature Engineering:** 18+ custom features (House Age, Renovations, Ratios).
- **Advanced Encoding:** Target Encoding for Zip Codes and Cities.
- **Outlier Handling:** Top 1% of expensive homes removed to stabilize predictions.
- **Model:** Random Forest with Hyperparameter Tuning.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
