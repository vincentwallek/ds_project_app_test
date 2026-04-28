import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from supabase import create_client
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import google.generativeai as genai
import os

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="VVI | Vehicle Valuation Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}
        h1, h2, h3 {font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #111827; font-weight: 600;}
        .stButton>button {width: 100%; background-color: #0f172a; color: white; border-radius: 6px; padding: 0.6rem; border: none; font-weight: 500;}
        .stButton>button:hover {background-color: #334155; color: white;}
        .card {background-color: #ffffff; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #e5e7eb; margin-bottom: 1rem;}
        .logo-text {font-size: 24px; font-weight: 700; color: #0f172a; letter-spacing: -0.5px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. STATE MANAGEMENT & ROUTING
# ==========================================
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'role' not in st.session_state: st.session_state.role = None
if 'market' not in st.session_state: st.session_state.market = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []


def navigate(page, role=None, market=None):
    st.session_state.page = page
    if role: st.session_state.role = role
    if market: st.session_state.market = market
    st.rerun()


# ==========================================
# 3. INITIALIZATION & HELPERS
# ==========================================
@st.cache_resource
def init_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


supabase = init_supabase()
geolocator = Nominatim(user_agent="vvi_app_v2")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


@st.cache_resource
def load_models():
    models = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(base_dir, "car_price_xgboost.pkl"), "rb") as f:
            models["de_model"] = pickle.load(f)
        with open(os.path.join(base_dir, "categorical_encoder.pkl"), "rb") as f:
            models["de_encoder"] = pickle.load(f)
        with open(os.path.join(base_dir, "numeric_columns.pkl"), "rb") as f:
            models["de_num_cols"] = pickle.load(f)
        with open(os.path.join(base_dir, "car_price_xgboost_us.pkl"), "rb") as f:
            models["us_model"] = pickle.load(f)
        with open(os.path.join(base_dir, "categorical_encoder_us.pkl"), "rb") as f:
            models["us_encoder"] = pickle.load(f)
        with open(os.path.join(base_dir, "numeric_columns_us.pkl"), "rb") as f:
            models["us_num_cols"] = pickle.load(f)
        return models
    except Exception as e:
        st.error(f"System Error: Model path failed. Details: {e}")
        st.stop()


models = load_models()


@st.cache_data
def get_unique_vehicles(market_code):
    """Holt Marken, Modelle und Titel live aus Supabase"""
    try:
        res = supabase.table("listings").select("brand, model, title").eq("market", market_code).execute()
        df = pd.DataFrame(res.data)
        hierarchy = {}
        for b in df['brand'].unique():
            models_for_brand = df[df['brand'] == b]['model'].unique().tolist()
            hierarchy[b.lower()] = [m.lower() for m in models_for_brand]
        return hierarchy, df
    except:
        return {}, pd.DataFrame()


@st.cache_data
def get_coordinates(location_string):
    if not location_string or location_string == "unbekannt": return None, None
    try:
        loc = geocode(location_string)
        if loc: return loc.latitude, loc.longitude
    except:
        pass
    return None, None


def predict_price(market, input_data):
    if market == "DE":
        model, encoder, num_cols = models["de_model"], models["de_encoder"], models["de_num_cols"]
        cat_cols = ["brand", "model", "transmission", "fuel"]
    else:
        model, encoder, num_cols = models["us_model"], models["us_encoder"], models["us_num_cols"]
        cat_cols = ["brand", "model", "trim", "drivetrain", "fuel", "transmission", "body_style", "engine",
                    "exterior_color", "interior_color", "usage_type"]

    df_input = pd.DataFrame([input_data])
    for col in num_cols:
        if col not in df_input.columns: df_input[col] = 0.0

    encoded_cats = encoder.transform(df_input[cat_cols])
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    df_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols, index=df_input.index)
    X_final = pd.concat([df_input[num_cols], df_encoded], axis=1)

    prediction = model.predict(X_final)[0]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_final)

    return prediction, shap_values


# ==========================================
# 4. UI VIEWS
# ==========================================
def view_header():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown('<div class="logo-text">VVI</div>', unsafe_allow_html=True)
    with col2:
        if st.session_state.page != 'home':
            st.button("Back to Dashboard", on_click=navigate, args=('home',))
    st.divider()


def view_home():
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>Intelligence Dashboard</h1>",
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.h3("Seller Intelligence")
        st.markdown("Asset valuation and feature impact analysis for professionals.")
        if st.button("Evaluate Vehicle (DE)"): navigate('app', 'seller', 'DE')
        if st.button("Evaluate Vehicle (US)"): navigate('app', 'seller', 'US')
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.h3("Buyer Intelligence")
        st.markdown("Market search, fair-price estimation and inventory tracking.")
        if st.button("Search Inventory (DE)"): navigate('app', 'buyer', 'DE')
        if st.button("Search Inventory (US)"): navigate('app', 'buyer', 'US')
        st.markdown('</div>', unsafe_allow_html=True)


def render_vehicle_form(role):
    market = st.session_state.market
    currency = "€" if market == "DE" else "$"

    # Live Daten laden
    hierarchy, raw_db_data = get_unique_vehicles(market)

    with st.form(f"{role}_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            brand_options = list(hierarchy.keys())
            brand = st.selectbox("Brand", options=brand_options if brand_options else ["mercedes-benz"])

            model_options = hierarchy.get(brand, [])
            model_name = st.selectbox("Model", options=model_options if model_options else ["c-klasse"])

            # Spezifische Version (Title) Auswahl
            variant_options = \
            raw_db_data[(raw_db_data['brand'].str.lower() == brand) & (raw_db_data['model'].str.lower() == model_name)][
                'title'].unique().tolist()
            selected_variant = st.selectbox("Specific Version",
                                            options=variant_options if variant_options else ["Standard"])

            car_age = st.number_input("Age (Years)", min_value=0, max_value=30, value=3)
        with col2:
            mileage = st.number_input("Mileage", min_value=0, value=50000, step=5000)
            transmission = st.selectbox("Transmission", ["automatic", "manual"])
            fuel = st.selectbox("Fuel Type", ["benzin", "diesel", "elektro", "hybrid"])
        with col3:
            if market == "DE":
                power_ps = st.number_input("Power (PS)", min_value=0, value=150)
                has_pano = st.checkbox("Panoramic Roof")
                has_amg = st.checkbox("AMG Line")
            else:
                cylinders = st.selectbox("Cylinders", [4, 6, 8])
                has_accidents = st.checkbox("Accident History")
                is_cpo = st.checkbox("Certified (CPO)")

        submit_btn = st.form_submit_button("Start Analysis")

    if submit_btn:
        with st.spinner("Analyzing market drivers..."):
            input_data = {"brand": brand, "model": model_name, "car_age": float(car_age), "mileage": float(mileage),
                          "transmission": transmission, "fuel": fuel}
            if market == "DE":
                input_data.update(
                    {"power_ps": float(power_ps), "owners": 1.0, "ausstattung_pano": 1.0 if has_pano else 0.0,
                     "ausstattung_amg_line": 1.0 if has_amg else 0.0})
            else:
                # Defaults für US falls nicht im Formular
                input_data.update({"cylinders": float(cylinders), "has_accidents": 1.0 if has_accidents else 0.0,
                                   "is_cpo": 1.0 if is_cpo else 0.0, "doors": 4.0, "seats": 5.0, "trim": "unbekannt",
                                   "drivetrain": "unbekannt", "body_style": "unbekannt", "engine": "unbekannt",
                                   "exterior_color": "unbekannt", "interior_color": "unbekannt",
                                   "usage_type": "unbekannt"})

            predicted_price, shap_vals = predict_price(market, input_data)
            st.divider()
            st.metric(label=f"Estimated Market Value ({selected_variant})", value=f"{predicted_price:,.2f} {currency}")

            if role == "seller":
                st.markdown("### Feature Impact Analysis")
                fig = plt.figure(figsize=(10, 6))
                shap.plots.waterfall(shap_vals[0], show=False)
                plt.subplots_adjust(left=0.35, right=0.9, top=0.9)
                st.pyplot(fig, clear_figure=True)

            if role == "buyer":
                st.markdown("### Matching Inventory")
                query = supabase.table("listings").select("brand, model, price, mileage, location, url").eq("market",
                                                                                                            market).eq(
                    "brand", brand).eq("model", model_name)
                res = query.lte("price", predicted_price * 1.2).limit(15).execute()
                df_results = pd.DataFrame(res.data)

                if not df_results.empty:
                    st.dataframe(df_results, column_config={
                        "price": st.column_config.NumberColumn("Price", format="%d " + currency),
                        "url": st.column_config.LinkColumn("Listing", display_text="Open Link")},
                                 use_container_width=True, hide_index=True)
                    lats = [{"lat": lat, "lon": lon} for lat, lon in
                            [get_coordinates(loc) for loc in df_results["location"]] if lat]
                    if lats: st.map(pd.DataFrame(lats))


def view_app():
    title = "Seller Intelligence" if st.session_state.role == "seller" else "Buyer Intelligence"
    st.subheader(f"{title} | {st.session_state.market} Market")
    t1, t2 = st.tabs(["Analysis Engine", "VVI Chat Assistant"])

    with t1:
        render_vehicle_form(st.session_state.role)

    with t2:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-1.5-flash',
                                          system_instruction=f"You are a professional automotive market expert for {st.session_state.market}. Tone: Serious, professional, no emojis.")

            for m in st.session_state.chat_history:
                with st.chat_message(m["role"]): st.markdown(m["content"])

            if p := st.chat_input("Ask about market trends..."):
                with st.chat_message("user"): st.markdown(p)
                st.session_state.chat_history.append({"role": "user", "content": p})

                with st.spinner("VVI is analyzing..."):
                    h = [{"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]} for msg in
                         st.session_state.chat_history[:-1]]
                    resp = model.start_chat(history=h).send_message(p)
                    with st.chat_message("assistant"): st.markdown(resp.text)
                    st.session_state.chat_history.append({"role": "assistant", "content": resp.text})
        else:
            st.error("Chat Assistant unavailable: GEMINI_API_KEY missing.")


view_header()
if st.session_state.page == 'home':
    view_home()
else:
    view_app()