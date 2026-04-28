import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from supabase import create_client
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Vehicle Valuation Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimalistisches, professionelles CSS-Styling (versteckt Streamlit-Brandings)
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        h1, h2, h3 {font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #1f2937;}
        .stButton>button {width: 100%; background-color: #0f172a; color: white; border-radius: 4px;}
        .stButton>button:hover {background-color: #334155; color: white;}
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. INITIALIZATION & DATA LOADING
# ==========================================
@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


supabase = init_supabase()
geolocator = Nominatim(user_agent="vehicle_valuation_app")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


@st.cache_resource
def load_models():
    models = {}
    # Den genauen Pfad zu dem Ordner ermitteln, in dem die app.py liegt
    base_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # DE Models
        with open(os.path.join(base_dir, "car_price_xgboost.pkl"), "rb") as f:
            models["de_model"] = pickle.load(f)
        with open(os.path.join(base_dir, "categorical_encoder.pkl"), "rb") as f:
            models["de_encoder"] = pickle.load(f)
        with open(os.path.join(base_dir, "numeric_columns.pkl"), "rb") as f:
            models["de_num_cols"] = pickle.load(f)

        # US Models
        with open(os.path.join(base_dir, "car_price_xgboost_us.pkl"), "rb") as f:
            models["us_model"] = pickle.load(f)
        with open(os.path.join(base_dir, "categorical_encoder_us.pkl"), "rb") as f:
            models["us_encoder"] = pickle.load(f)
        with open(os.path.join(base_dir, "numeric_columns_us.pkl"), "rb") as f:
            models["us_num_cols"] = pickle.load(f)
        return models
    except Exception as e:
        st.error(f"System Error: Model initialization failed. Details: {e}")
        st.stop()


models = load_models()


@st.cache_data
def get_coordinates(location_string):
    """Konvertiert String-Adressen in Koordinaten für die Karte"""
    if not location_string or location_string == "unbekannt":
        return None, None
    try:
        loc = geocode(location_string)
        if loc:
            return loc.latitude, loc.longitude
    except:
        pass
    return None, None


# ==========================================
# 3. HELPER FUNCTIONS FOR PREDICTION
# ==========================================
def predict_price(market, input_data):
    if market == "DE":
        model = models["de_model"]
        encoder = models["de_encoder"]
        num_cols = models["de_num_cols"]
        cat_cols = ["brand", "model", "transmission", "fuel"]
    else:
        model = models["us_model"]
        encoder = models["us_encoder"]
        num_cols = models["us_num_cols"]
        cat_cols = ["brand", "model", "trim", "drivetrain", "fuel", "transmission", "body_style", "engine",
                    "exterior_color", "interior_color", "usage_type"]

    df_input = pd.DataFrame([input_data])

    # Sicherstellen, dass alle Features vorhanden sind (mit 0 füllen, falls nicht)
    for col in num_cols:
        if col not in df_input.columns:
            df_input[col] = 0.0

    # Kategoriale Daten encodieren
    encoded_cats = encoder.transform(df_input[cat_cols])
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    df_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols, index=df_input.index)

    # Zusammenfügen
    X_final = pd.concat([df_input[num_cols], df_encoded], axis=1)

    # Vorhersage
    prediction = model.predict(X_final)[0]

    # SHAP Erklärung berechnen
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_final)

    return prediction, shap_values


# ==========================================
# 4. UI LAYOUT
# ==========================================
st.title("Vehicle Valuation Intelligence")
st.markdown("Professional market analysis and price prediction powered by Machine Learning.")

# Sidebar: Market Selection
st.sidebar.header("Market Selection")
market = st.sidebar.radio("Select Target Market:", ["Germany (DE)", "United States (US)"])
market_code = "DE" if "Germany" in market else "US"
currency = "€" if market_code == "DE" else "$"

tab_seller, tab_buyer = st.tabs(["Seller Intelligence (Valuation)", "Buyer Intelligence (Market Search)"])

# ------------------------------------------
# TAB 1: SELLER (Valuation & SHAP)
# ------------------------------------------
with tab_seller:
    st.subheader("Asset Valuation")
    st.markdown(
        "Input vehicle specifications to receive an AI-driven market value estimation and feature impact analysis.")

    with st.form("valuation_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            brand = st.text_input("Brand", value="mercedes-benz" if market_code == "DE" else "ford")
            model_name = st.text_input("Model", value="c-klasse" if market_code == "DE" else "f-150")
            car_age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=50, value=5)

        with col2:
            mileage = st.number_input("Mileage", min_value=0, value=50000)
            transmission = st.selectbox("Transmission", ["automatic", "manual", "unbekannt"])
            fuel = st.selectbox("Fuel Type", ["benzin", "diesel", "elektro", "hybrid", "unbekannt", "gas"])

        with col3:
            if market_code == "DE":
                power_ps = st.number_input("Power (PS)", min_value=0, value=150)
                owners = st.number_input("Previous Owners", min_value=1, value=1)
                st.markdown("**Key Features**")
                has_pano = st.checkbox("Panoramic Roof")
                has_amg = st.checkbox("AMG Line")
            else:
                cylinders = st.number_input("Cylinders", min_value=0, value=6)
                has_accidents = st.checkbox("Accident History")
                is_cpo = st.checkbox("Certified Pre-Owned (CPO)")

        submit_valuation = st.form_submit_button("Calculate Market Value")

    if submit_valuation:
        with st.spinner("Analyzing market data..."):
            input_data = {
                "brand": brand.lower().strip(),
                "model": model_name.lower().strip(),
                "car_age": float(car_age),
                "mileage": float(mileage),
                "transmission": transmission.lower(),
                "fuel": fuel.lower()
            }

            if market_code == "DE":
                input_data.update({
                    "power_ps": float(power_ps),
                    "owners": float(owners),
                    "ausstattung_pano": 1.0 if has_pano else 0.0,
                    "ausstattung_amg_line": 1.0 if has_amg else 0.0
                })
            else:
                input_data.update({
                    "cylinders": float(cylinders),
                    "has_accidents": 1.0 if has_accidents else 0.0,
                    "is_cpo": 1.0 if is_cpo else 0.0,
                    "doors": 4.0, "seats": 5.0,  # Defaults
                    "trim": "unbekannt", "drivetrain": "unbekannt", "body_style": "unbekannt",
                    "engine": "unbekannt", "exterior_color": "unbekannt", "interior_color": "unbekannt",
                    "usage_type": "unbekannt"
                })

            predicted_price, shap_vals = predict_price(market_code, input_data)

            st.divider()
            st.metric(label="Estimated Market Value", value=f"{predicted_price:,.2f} {currency}")

            st.markdown("### Feature Impact Analysis")
            st.markdown("Understanding the key drivers behind this valuation:")

            # Professionelles Matplotlib SHAP Chart
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_vals[0], show=False)
            plt.gcf().set_size_inches(8, 5)
            plt.tight_layout()
            st.pyplot(fig)

# ------------------------------------------
# TAB 2: BUYER (Search & Map)
# ------------------------------------------
with tab_buyer:
    st.subheader("Market Inventory Search")
    st.markdown("Discover available listings matching your criteria and visualize their locations.")

    with st.form("search_form"):
        col1, col2 = st.columns(2)
        with col1:
            search_brand = st.text_input("Target Brand", value="bmw")
            search_model = st.text_input("Target Model", value="")
        with col2:
            max_price = st.number_input("Maximum Price", min_value=0, value=50000, step=1000)
            max_mileage = st.number_input("Maximum Mileage", min_value=0, value=100000, step=5000)

        submit_search = st.form_submit_button("Search Market")

    if submit_search:
        with st.spinner("Querying database & resolving geographic data..."):
            # Datenbankabfrage via Supabase
            query = supabase.table("listings").select("id, brand, model, price, mileage, location").eq("market",
                                                                                                       market_code).ilike(
                "brand", f"%{search_brand}%")
            if search_model:
                query = query.ilike("model", f"%{search_model}%")

            # API Limitierung zum Schutz der Performance
            res = query.lte("price", max_price).lte("mileage", max_mileage).limit(50).execute()

            df_results = pd.DataFrame(res.data)

            if df_results.empty:
                st.warning("No listings found matching your criteria.")
            else:
                st.success(f"Found {len(df_results)} matching listings.")

                # Karte generieren
                locations_for_map = []
                for _, row in df_results.iterrows():
                    lat, lon = get_coordinates(row.get("location"))
                    if lat and lon:
                        locations_for_map.append({"lat": lat, "lon": lon})

                if locations_for_map:
                    df_map = pd.DataFrame(locations_for_map)
                    st.map(df_map)
                else:
                    st.info("Location data not resolved for the map view. Ensure valid ZIP codes/Cities are present.")

                # Daten-Tabelle anzeigen (aufgeräumt)
                st.dataframe(
                    df_results[["brand", "model", "price", "mileage", "location"]].style.format({"price": "{:,.2f}"}),
                    use_container_width=True
                )