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
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AutoValue | Vehicle Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. THEME ENGINE
# ==========================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# --- Sidebar: Theme Toggle & Navigation ---
with st.sidebar:
    st.markdown("### Settings")
    theme_label = "Switch to Light Mode" if st.session_state.theme == "dark" else "Switch to Dark Mode"
    if st.button(theme_label, key="theme_toggle"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

    st.markdown("---")
    st.markdown(
        f"<p style='font-size:0.78rem; color: {'#94a3b8' if st.session_state.theme == 'dark' else '#64748b'};'>"
        "AutoValue v3.0 &middot; Vehicle Intelligence Platform</p>",
        unsafe_allow_html=True,
    )

# --- Theme Palette ---
if st.session_state.theme == "dark":
    T = {
        "bg": "#0f172a",
        "bg_secondary": "#1e293b",
        "card_bg": "#1e293b",
        "card_border": "#334155",
        "text_primary": "#f1f5f9",
        "text_secondary": "#94a3b8",
        "text_heading": "#f8fafc",
        "accent": "#3b82f6",
        "accent_hover": "#60a5fa",
        "btn_bg": "#3b82f6",
        "btn_text": "#ffffff",
        "btn_hover_bg": "#2563eb",
        "divider": "#334155",
        "input_bg": "#1e293b",
        "input_border": "#475569",
        "input_text": "#f1f5f9",
        "shadow": "rgba(0,0,0,0.4)",
        "tab_active": "#3b82f6",
        "tab_inactive": "#94a3b8",
        "metric_bg": "#1e293b",
        "sidebar_bg": "#0f172a",
    }
else:
    T = {
        "bg": "#f8fafc",
        "bg_secondary": "#ffffff",
        "card_bg": "#ffffff",
        "card_border": "#e2e8f0",
        "text_primary": "#1e293b",
        "text_secondary": "#64748b",
        "text_heading": "#0f172a",
        "accent": "#2563eb",
        "accent_hover": "#1d4ed8",
        "btn_bg": "#2563eb",
        "btn_text": "#ffffff",
        "btn_hover_bg": "#1d4ed8",
        "divider": "#e2e8f0",
        "input_bg": "#f1f5f9",
        "input_border": "#cbd5e1",
        "input_text": "#1e293b",
        "shadow": "rgba(0,0,0,0.08)",
        "tab_active": "#2563eb",
        "tab_inactive": "#64748b",
        "metric_bg": "#f1f5f9",
        "sidebar_bg": "#ffffff",
    }

# ==========================================
# 3. GLOBAL STYLESHEET
# ==========================================
st.markdown(f"""
    <style>
        /* --- Google Font --- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* --- Root Reset --- */
        html, body, .stApp {{
            background-color: {T['bg']} !important;
            font-family: 'Inter', sans-serif !important;
        }}

        /* --- Sidebar --- */
        section[data-testid="stSidebar"] {{
            background-color: {T['bg_secondary']} !important;
            border-right: 1px solid {T['divider']} !important;
        }}
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {{
            color: {T['text_primary']} !important;
        }}

        /* --- Typography --- */
        .stMarkdown, p, span, label, li {{
            color: {T['text_primary']} !important;
            font-family: 'Inter', sans-serif !important;
        }}
        h1, h2, h3, h4 {{
            color: {T['text_heading']} !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
        }}

        /* --- Cards --- */
        .av-card {{
            background-color: {T['card_bg']};
            padding: 2rem;
            border-radius: 14px;
            box-shadow: 0 4px 24px {T['shadow']};
            border: 1px solid {T['card_border']};
            margin-bottom: 1.5rem;
            min-height: 240px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .av-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 32px {T['shadow']};
        }}
        .av-card h3 {{
            color: {T['text_heading']} !important;
            margin-bottom: 0.5rem;
            font-size: 1.25rem;
        }}
        .av-card p {{
            color: {T['text_secondary']} !important;
            font-size: 0.9rem;
            line-height: 1.6;
        }}
        .av-card .card-tag {{
            display: inline-block;
            background-color: {T['accent']};
            color: #ffffff;
            font-size: 0.7rem;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 20px;
            margin-bottom: 0.75rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}

        /* --- Buttons --- */
        .stButton > button {{
            width: 100%;
            background-color: {T['btn_bg']} !important;
            color: {T['btn_text']} !important;
            border-radius: 10px;
            padding: 0.75rem 1.25rem;
            border: none !important;
            font-weight: 600;
            font-size: 0.9rem;
            font-family: 'Inter', sans-serif !important;
            cursor: pointer;
            transition: background-color 0.2s ease, transform 0.15s ease;
            letter-spacing: 0.2px;
        }}
        .stButton > button:hover {{
            background-color: {T['btn_hover_bg']} !important;
            color: {T['btn_text']} !important;
            transform: translateY(-1px);
        }}
        .stButton > button:active {{
            transform: translateY(0);
        }}
        /* Fix: force button text color on all states */
        .stButton > button p,
        .stButton > button span,
        .stButton > button div {{
            color: {T['btn_text']} !important;
        }}

        /* --- Form submit button --- */
        .stFormSubmitButton > button {{
            background-color: {T['accent']} !important;
            color: #ffffff !important;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.75rem;
            border: none !important;
            transition: background-color 0.2s ease;
        }}
        .stFormSubmitButton > button:hover {{
            background-color: {T['accent_hover']} !important;
            color: #ffffff !important;
        }}
        .stFormSubmitButton > button p,
        .stFormSubmitButton > button span {{
            color: #ffffff !important;
        }}

        /* --- Tabs --- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0;
            border-bottom: 2px solid {T['divider']};
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 48px;
            padding: 0 1.5rem;
            background-color: transparent !important;
            color: {T['tab_inactive']} !important;
            border: none;
            font-weight: 500;
            font-family: 'Inter', sans-serif !important;
            transition: color 0.2s ease;
        }}
        .stTabs [aria-selected="true"] {{
            color: {T['tab_active']} !important;
            border-bottom: 2px solid {T['tab_active']} !important;
            font-weight: 600;
        }}

        /* --- Input fields --- */
        .stSelectbox > div > div,
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {{
            background-color: {T['input_bg']} !important;
            color: {T['input_text']} !important;
            border: 1px solid {T['input_border']} !important;
            border-radius: 8px !important;
        }}
        .stSelectbox label,
        .stNumberInput label,
        .stTextInput label,
        .stCheckbox label {{
            color: {T['text_secondary']} !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
        }}
        .stSelectbox svg {{
            fill: {T['text_secondary']} !important;
        }}

        /* --- Checkbox --- */
        .stCheckbox span {{
            color: {T['text_primary']} !important;
        }}

        /* --- Metrics --- */
        [data-testid="stMetricValue"] {{
            color: {T['accent']} !important;
            font-weight: 700 !important;
            font-size: 2rem !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {T['text_secondary']} !important;
        }}
        [data-testid="metric-container"] {{
            background-color: {T['metric_bg']};
            border: 1px solid {T['card_border']};
            padding: 1rem 1.5rem;
            border-radius: 12px;
        }}

        /* --- Divider --- */
        hr {{
            border-color: {T['divider']} !important;
        }}

        /* --- Dataframe --- */
        .stDataFrame {{
            border-radius: 10px;
            overflow: hidden;
        }}

        /* --- Chat --- */
        .stChatMessage {{
            background-color: {T['card_bg']} !important;
            border: 1px solid {T['card_border']} !important;
            border-radius: 12px !important;
        }}

        /* --- Spinner --- */
        .stSpinner > div {{
            border-top-color: {T['accent']} !important;
        }}

        /* --- Hide Streamlit branding --- */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        /* --- Logo container --- */
        .av-logo-wrap {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .av-logo-wrap img {{
            border-radius: 10px;
            background-color: transparent;
        }}
        .av-logo-text {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {T['text_heading']};
            letter-spacing: -0.5px;
        }}
        .av-logo-sub {{
            font-size: 0.78rem;
            color: {T['text_secondary']};
            margin-top: -2px;
        }}

        /* --- Header bar --- */
        .av-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
        }}

        /* --- Hero Section --- */
        .av-hero {{
            text-align: center;
            padding: 2rem 0 1rem 0;
        }}
        .av-hero h2 {{
            font-size: 2rem;
            font-weight: 700;
            color: {T['text_heading']} !important;
            margin-bottom: 0.5rem;
        }}
        .av-hero p {{
            color: {T['text_secondary']} !important;
            font-size: 1.05rem;
            max-width: 560px;
            margin: 0 auto;
            line-height: 1.7;
        }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. STATE MANAGEMENT & ROUTING
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'role' not in st.session_state:
    st.session_state.role = None
if 'market' not in st.session_state:
    st.session_state.market = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def nav(page, role=None, market=None):
    """Set navigation state."""
    st.session_state.page = page
    st.session_state.role = role
    st.session_state.market = market


# ==========================================
# 5. INITIALIZATION & DATA ENGINE
# ==========================================
@st.cache_resource
def init_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


supabase = init_supabase()
geolocator = Nominatim(user_agent="autovalue_v3")
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
    except:
        return None


trained_models = load_models()


@st.cache_data(ttl=600)
def get_market_data(market_code):
    try:
        res = supabase.table("listings").select(
            "brand, model, title, price, mileage, location, url"
        ).eq("market", market_code).execute()
        df = pd.DataFrame(res.data)
        if not df.empty:
            df['brand'] = df['brand'].str.lower()
            df['model'] = df['model'].str.lower()
        return df
    except:
        return pd.DataFrame(columns=["brand", "model", "title", "price", "mileage", "location", "url"])


@st.cache_data
def get_coords(loc_string):
    if not loc_string or loc_string == "unbekannt":
        return None, None
    try:
        loc = geocode(loc_string)
        if loc:
            return loc.latitude, loc.longitude
    except:
        pass
    return None, None


def predict_price(market, input_data):
    if not trained_models:
        return 0.0, None
    m_code = "de" if market == "DE" else "us"
    model = trained_models[f"{m_code}_model"]
    encoder = trained_models[f"{m_code}_encoder"]
    num_cols = trained_models[f"{m_code}_num_cols"]
    cat_cols = (
        ["brand", "model", "transmission", "fuel"]
        if market == "DE"
        else [
            "brand", "model", "trim", "drivetrain", "fuel",
            "transmission", "body_style", "engine",
            "exterior_color", "interior_color", "usage_type",
        ]
    )
    df_input = pd.DataFrame([input_data])
    for col in num_cols:
        if col not in df_input.columns:
            df_input[col] = 0.0
    encoded_cats = encoder.transform(df_input[cat_cols])
    df_encoded = pd.DataFrame(
        encoded_cats,
        columns=encoder.get_feature_names_out(cat_cols),
        index=df_input.index,
    )
    X_final = pd.concat([df_input[num_cols], df_encoded], axis=1)
    prediction = model.predict(X_final)[0]
    shap_values = shap.TreeExplainer(model)(X_final)
    return prediction, shap_values


# ==========================================
# 6. UI VIEWS
# ==========================================

def view_header():
    """Render the top header bar with logo and back-navigation."""
    col_logo, col_spacer, col_nav = st.columns([2, 6, 2])
    with col_logo:
        base_path = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(base_path, "logo.png")
        if os.path.exists(logo_path):
            st.markdown(
                f"""
                <div class="av-logo-wrap">
                    <img src="data:image/png;base64,{_img_to_base64(logo_path)}" width="48" height="48" />
                    <div>
                        <div class="av-logo-text">AutoValue</div>
                        <div class="av-logo-sub">Vehicle Intelligence</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="av-logo-wrap">
                    <div>
                        <div class="av-logo-text">AutoValue</div>
                        <div class="av-logo-sub">Vehicle Intelligence</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with col_nav:
        if st.session_state.page != "home":
            st.write("")  # vertical spacer
            if st.button("Back to Dashboard", key="btn_back"):
                nav("home")
                st.rerun()
    st.markdown(f"<hr style='border:none;border-top:1px solid {T['divider']};margin:0.5rem 0 1.5rem 0;'>",
                unsafe_allow_html=True)


def _img_to_base64(path):
    """Convert an image file to a base64-encoded string for inline embedding."""
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def view_home():
    """Render the home / dashboard page."""
    # Hero
    st.markdown(
        f"""
        <div class="av-hero">
            <h2>Professional Vehicle Valuation</h2>
            <p>Leverage machine-learning models trained on real market data to
            accurately price vehicles across the German and US markets.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            f"""
            <div class="av-card">
                <span class="card-tag">Seller</span>
                <h3>Seller Intelligence</h3>
                <p>Enter your vehicle details to receive a data-driven market
                valuation. Understand which features drive the price up or down
                with SHAP-based explainability.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1a, c1b = st.columns(2)
        with c1a:
            if st.button("Start Valuation  --  DE", key="btn_s_de"):
                nav("app", "seller", "DE")
                st.rerun()
        with c1b:
            if st.button("Start Valuation  --  US", key="btn_s_us"):
                nav("app", "seller", "US")
                st.rerun()

    with col2:
        st.markdown(
            f"""
            <div class="av-card">
                <span class="card-tag">Buyer</span>
                <h3>Buyer Intelligence</h3>
                <p>Search real inventory listings, compare prices against
                the predicted fair value, and find the best deals on the
                market with location mapping.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c2a, c2b = st.columns(2)
        with c2a:
            if st.button("Search Inventory  --  DE", key="btn_b_de"):
                nav("app", "buyer", "DE")
                st.rerun()
        with c2b:
            if st.button("Search Inventory  --  US", key="btn_b_us"):
                nav("app", "buyer", "US")
                st.rerun()


def view_app():
    """Render the analysis / chat application page."""
    market = st.session_state.market
    role = st.session_state.role
    db_data = get_market_data(market)
    currency = "EUR" if market == "DE" else "USD"

    st.subheader(f"{role.capitalize()} Analysis  |  Market: {market}")

    tab_engine, tab_chat = st.tabs(["Analysis Engine", "AutoValue Assistant"])

    # --- Analysis Engine Tab ---
    with tab_engine:
        with st.form("valuation_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                brands = sorted(db_data['brand'].unique()) if not db_data.empty else ["mercedes-benz"]
                brand = st.selectbox("Brand", brands)
                models_list = (
                    sorted(db_data[db_data['brand'] == brand]['model'].unique())
                    if not db_data.empty
                    else ["c-klasse"]
                )
                model_name = st.selectbox("Model", models_list)
                variant = st.selectbox(
                    "Variant",
                    db_data[
                        (db_data['brand'] == brand) & (db_data['model'] == model_name)
                    ]['title'].unique(),
                ) if not db_data.empty else "Standard"
                age = st.number_input("Age (years)", 0, 30, 3)
            with c2:
                mileage = st.number_input("Mileage (km)", 0, 300000, 50000, 5000)
                trans = st.selectbox("Transmission", ["automatic", "manual"])
                fuel = st.selectbox("Fuel Type", ["benzin", "diesel", "elektro", "hybrid"])
            with c3:
                if market == "DE":
                    power = st.number_input("Power (PS)", 50, 800, 150)
                    opt1 = st.checkbox("Panoramic Roof")
                    opt2 = st.checkbox("AMG Line")
                else:
                    power = st.number_input("Cylinders", 3, 12, 6)
                    opt1 = st.checkbox("Accident-Free")
                    opt2 = st.checkbox("CPO Status")
            submitted = st.form_submit_button("Run Analysis")

        if submitted:
            with st.spinner("Calculating market value ..."):
                input_vals = {
                    "brand": brand,
                    "model": model_name,
                    "car_age": float(age),
                    "mileage": float(mileage),
                    "transmission": trans,
                    "fuel": fuel,
                }
                if market == "DE":
                    input_vals.update({
                        "power_ps": float(power),
                        "owners": 1.0,
                        "ausstattung_pano": 1.0 if opt1 else 0.0,
                        "ausstattung_amg_line": 1.0 if opt2 else 0.0,
                    })
                else:
                    input_vals.update({
                        "cylinders": float(power),
                        "has_accidents": 1.0 if opt1 else 0.0,
                        "is_cpo": 1.0 if opt2 else 0.0,
                        "doors": 4.0,
                        "seats": 5.0,
                        "trim": "unbekannt",
                        "drivetrain": "unbekannt",
                        "body_style": "unbekannt",
                        "engine": "unbekannt",
                        "exterior_color": "unbekannt",
                        "interior_color": "unbekannt",
                        "usage_type": "unbekannt",
                    })

                price, s_vals = predict_price(market, input_vals)
                st.markdown(f"<hr style='border:none;border-top:1px solid {T['divider']};margin:1.5rem 0;'>",
                            unsafe_allow_html=True)
                st.metric(f"Estimated Market Value ({variant})", f"{price:,.2f} {currency}")

                if role == "seller" and s_vals:
                    st.markdown("### Price Influence Factors")
                    fig = plt.figure(figsize=(10, 6))
                    shap.plots.waterfall(s_vals[0], show=False)
                    plt.subplots_adjust(left=0.35, right=0.9)
                    st.pyplot(fig)

                if role == "buyer" and not db_data.empty:
                    st.markdown("### Matching Listings")
                    matches = db_data[
                        (db_data['brand'] == brand) & (db_data['model'] == model_name)
                    ].head(10)
                    st.dataframe(
                        matches[["title", "price", "mileage", "url"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                    coords = [
                        {"lat": lt, "lon": ln}
                        for lt, ln in [get_coords(loc) for loc in matches['location']]
                        if lt is not None
                    ]
                    if coords:
                        st.map(pd.DataFrame(coords))

    # --- Chat Tab ---
    with tab_chat:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            chat_m = genai.GenerativeModel(
                "gemini-1.5-flash",
                system_instruction=(
                    f"You are the AutoValue expert for the {market} market. "
                    "Answer professionally and concisely."
                ),
            )
            for m in st.session_state.chat_history:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            if p := st.chat_input("Ask a question ..."):
                with st.chat_message("user"):
                    st.markdown(p)
                st.session_state.chat_history.append({"role": "user", "content": p})
                resp = chat_m.start_chat(
                    history=[
                        {
                            "role": "user" if m["role"] == "user" else "model",
                            "parts": [m["content"]],
                        }
                        for m in st.session_state.chat_history[:-1]
                    ]
                ).send_message(p)
                with st.chat_message("assistant"):
                    st.markdown(resp.text)
                st.session_state.chat_history.append({"role": "assistant", "content": resp.text})
        else:
            st.info("The chat assistant requires a Gemini API key. Add GEMINI_API_KEY to your Streamlit secrets.")


# ==========================================
# 7. MAIN ROUTER
# ==========================================
view_header()
if st.session_state.page == "home":
    view_home()
else:
    view_app()
