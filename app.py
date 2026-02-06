import streamlit as st
import pandas as pd
import pickle
import numpy as np
import gzip  # <--- Added to handle compression

# Load Artifacts
@st.cache_resource
def load_data():
    # READ COMPRESSED MODEL
    with gzip.open('best_model.pkl.gz', 'rb') as f:
        model = pickle.load(f)
        
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        cols = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, scaler, cols, encoders

model, scaler, model_columns, encoders = load_data()

st.title("🏡 Advanced House Price Estimator")
st.markdown("Optimization Level: **High (~90% Accuracy Target)**")

# --- USER INPUTS ---
col1, col2 = st.columns(2)
with col1:
    sqft_living = st.number_input("Sqft Living", 300, 10000, 2000)
    sqft_lot = st.number_input("Sqft Lot", 300, 50000, 5000)
    sqft_above = st.number_input("Sqft Above", 300, 10000, 2000)
    sqft_basement = st.number_input("Sqft Basement", 0, 5000, 0)
    bedrooms = st.slider("Bedrooms", 1, 9, 3)
    bathrooms = st.slider("Bathrooms", 1.0, 8.0, 2.0)
with col2:
    yr_built = st.number_input("Year Built", 1900, 2024, 2000)
    yr_renovated = st.number_input("Year Renovated (0 if none)", 0, 2024, 0)
    condition = st.slider("Condition (1-5)", 1, 5, 3)
    view = st.slider("View Quality (0-4)", 0, 4, 0)
    floors = st.selectbox("Floors", [1, 1.5, 2, 2.5, 3])
    waterfront = st.selectbox("Waterfront?", [0, 1])

# Location Selectors
city_list = list(encoders['city'].classes_)
zip_list = list(encoders['zip'].classes_)

city = st.selectbox("City", city_list)
zipcode = st.selectbox("Zip Code", zip_list)

if st.button("Predict Valuation"):
    input_data = {}
    
    # Basic Features
    input_data['bedrooms'] = bedrooms
    input_data['bathrooms'] = bathrooms
    input_data['sqft_living'] = sqft_living
    input_data['sqft_lot'] = sqft_lot
    input_data['floors'] = floors
    input_data['waterfront'] = waterfront
    input_data['view'] = view
    input_data['condition'] = condition
    input_data['sqft_above'] = sqft_above
    input_data['sqft_basement'] = sqft_basement
    
    # Feature Engineering
    input_data['house_age'] = 2024 - yr_built
    input_data['effective_age'] = (2024 - yr_renovated) if yr_renovated > 0 else input_data['house_age']
    input_data['was_renovated'] = 1 if yr_renovated > 0 else 0
    input_data['has_basement'] = 1 if sqft_basement > 0 else 0
    input_data['total_rooms'] = bedrooms + bathrooms
    input_data['bed_bath_ratio'] = bedrooms / (bathrooms + 0.1)
    input_data['living_lot_ratio'] = sqft_living / (sqft_lot + 1)
    input_data['above_pct'] = sqft_above / (sqft_living + 1)
    input_data['basement_pct'] = sqft_basement / (sqft_living + 1)
    input_data['avg_room_size'] = sqft_living / (input_data['total_rooms'] + 1)
    input_data['quality_score'] = condition + view + (waterfront * 3)
    input_data['is_luxury'] = 1 if (sqft_living > 3000 and bathrooms >= 3) else 0
    input_data['floor_density'] = sqft_living / (floors + 0.5)
    
    # Log Transforms
    input_data['log_sqft_living'] = np.log1p(sqft_living)
    input_data['log_sqft_lot'] = np.log1p(sqft_lot)
    input_data['log_sqft_above'] = np.log1p(sqft_above)
    
    # Interactions
    input_data['sqft_x_condition'] = sqft_living * condition
    input_data['sqft_x_view'] = sqft_living * view
    
    # Encoding
    try:
        input_data['zip_encoded'] = encoders['zip'].transform([zipcode])[0]
        input_data['city_encoded'] = encoders['city'].transform([city])[0]
    except:
        input_data['zip_encoded'] = 0
        input_data['city_encoded'] = 0

    # Target Encoding
    zip_mean_dict = encoders['zip_mean']
    city_mean_dict = encoders['city_mean']
    global_mean = encoders.get('global_mean', 500000)
    
    zip_avg_val = zip_mean_dict.get(input_data['zip_encoded'], global_mean)
    city_avg_val = city_mean_dict.get(input_data['city_encoded'], global_mean)
    
    input_data['log_zip_price'] = np.log1p(zip_avg_val)
    input_data['log_city_price'] = np.log1p(city_avg_val)

    # Predict
    df_input = pd.DataFrame([input_data])
    df_input = df_input[model_columns]
    df_scaled = scaler.transform(df_input)
    
    pred_log = model.predict(df_scaled)[0]
    pred_price = np.expm1(pred_log)
    
    st.success(f"💰 Estimated Price: ${pred_price:,.2f}")
