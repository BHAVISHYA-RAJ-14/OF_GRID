import streamlit as st
import pandas as pd
import pickle
import numpy as np
import gzip

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Real Estate AI Valuator",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .stButton>button {
        background-color: #2e8b57;
        color: white;
        font-size: 20px;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #3cb371;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2e8b57;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_resource
def load_data():
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

# --- HEADER ---
st.title("🏡 AI Real Estate Valuator")
st.markdown("### Professional Property Price Estimation Engine")
st.markdown("---")

# --- MAIN LAYOUT (2 Columns) ---
col_left, col_right = st.columns([1, 1.5], gap="large")

with col_left:
    st.subheader("📝 Property Details")
    
    # Use Tabs to organize inputs cleanly
    tab1, tab2, tab3 = st.tabs(["📏 Size & Rooms", "✨ Features", "📍 Location"])
    
    with tab1:
        sqft_living = st.number_input("Living Area (sqft)", 300, 10000, 2000, step=50)
        sqft_lot = st.number_input("Lot Size (sqft)", 300, 50000, 5000, step=100)
        sqft_above = st.number_input("Above Ground (sqft)", 300, 10000, 2000)
        sqft_basement = st.number_input("Basement (sqft)", 0, 5000, 0)
        
        c1, c2 = st.columns(2)
        with c1:
            bedrooms = st.slider("Bedrooms", 1, 9, 3)
        with c2:
            bathrooms = st.slider("Bathrooms", 1.0, 8.0, 2.0)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            floors = st.selectbox("Floors", [1, 1.5, 2, 2.5, 3])
            waterfront = st.selectbox("Waterfront Access", ["No", "Yes"])
        with c2:
            view = st.slider("View Quality (0-4)", 0, 4, 0)
            condition = st.slider("Condition (1-5)", 1, 5, 3)
            
        yr_built = st.number_input("Year Built", 1900, 2024, 2000)
        yr_renovated = st.number_input("Year Renovated (0 if none)", 0, 2024, 0)

    with tab3:
        city_list = list(encoders['city'].classes_)
        zip_list = list(encoders['zip'].classes_)
        
        city = st.selectbox("City", city_list)
        zipcode = st.selectbox("Zip Code", zip_list)

    # Convert textual inputs
    is_waterfront = 1 if waterfront == "Yes" else 0

    # PREDICT BUTTON
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Calculate Valuation")

with col_right:
    # --- LOGIC & PREDICTION DISPLAY ---
    if predict_btn:
        with st.spinner("Analyzing market data..."):
            # 1. Initialize Input Dict
            input_data = {}
            
            # 2. Basic Mapping
            input_data['bedrooms'] = bedrooms
            input_data['bathrooms'] = bathrooms
            input_data['sqft_living'] = sqft_living
            input_data['sqft_lot'] = sqft_lot
            input_data['floors'] = floors
            input_data['waterfront'] = is_waterfront
            input_data['view'] = view
            input_data['condition'] = condition
            input_data['sqft_above'] = sqft_above
            input_data['sqft_basement'] = sqft_basement
            
            # 3. Advanced Feature Engineering (Exact Match to Training)
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
            input_data['quality_score'] = condition + view + (is_waterfront * 3)
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

            # Target Encoding Lookup
            zip_mean_dict = encoders['zip_mean']
            city_mean_dict = encoders['city_mean']
            global_mean = encoders.get('global_mean', 500000)
            
            zip_avg_val = zip_mean_dict.get(input_data['zip_encoded'], global_mean)
            city_avg_val = city_mean_dict.get(input_data['city_encoded'], global_mean)
            
            input_data['log_zip_price'] = np.log1p(zip_avg_val)
            input_data['log_city_price'] = np.log1p(city_avg_val)

            # 4. Prepare DataFrame & Predict
            df_input = pd.DataFrame([input_data])
            df_input = df_input[model_columns] # Sort columns
            df_scaled = scaler.transform(df_input)
            
            pred_log = model.predict(df_scaled)[0]
            pred_price = np.expm1(pred_log)
            
            # --- DISPLAY RESULTS ---
            st.subheader("📊 Valuation Result")
            
            # Styled container for the price
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color:#555;">Estimated Market Value</h3>
                <h1 style="margin:0; color:#2e8b57; font-size: 3em;">${pred_price:,.2f}</h1>
                <p style="margin:0; color:#888;">&plusmn; 5% Confidence Range: <b>${pred_price*0.95:,.0f} - ${pred_price*1.05:,.0f}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparison Metrics
            st.markdown("### 💡 Key Insights")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Price per SqFt", f"${pred_price/sqft_living:.0f}")
            with m2:
                st.metric("Effective Age", f"{input_data['effective_age']} years")
            with m3:
                st.metric("Location Premium", "High" if input_data['zip_encoded'] > 50 else "Standard")
                
    else:
        # Default State
        st.info("👈 Adjust property details in the sidebar or tabs to generate a valuation.")
        st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1000&q=80", use_column_width=True, caption="AI-Powered Real Estate Analytics")
