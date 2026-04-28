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
from helpers import (
    img_to_base64, get_encoder_categories, generate_recommendations,
    DE_BINARY_LABELS, DE_CONDITION_KEYS, DE_EQUIP_KEYS,
)

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
if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = True

# --- Sidebar: Theme Toggle & Navigation ---
with st.sidebar:
    st.markdown("### Einstellungen")
    theme_label = "Zum hellen Modus wechseln" if st.session_state.theme == "dark" else "Zum dunklen Modus wechseln"
    if st.button(theme_label, key="theme_toggle"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

    st.markdown("---")
    st.markdown(
        f"<p style='font-size:0.78rem; color: {'#94a3b8' if st.session_state.theme == 'dark' else '#64748b'};'>"
        "AutoValue v3.0 &middot; Fahrzeug-Intelligenz-Plattform</p>",
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
    """Load listings from Supabase. Handles both lowercase and capitalized column names."""
    try:
        res = supabase.table("listings").select("*").execute()
        df = pd.DataFrame(res.data)
        if df.empty:
            return pd.DataFrame(columns=["brand", "model", "title", "price", "mileage", "location", "url"])
        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        # Filter by market (column may have been 'Market' or 'market')
        if "market" in df.columns:
            df = df[df["market"].str.upper() == market_code.upper()]
        # Ensure required columns exist
        for col in ["brand", "model", "title", "price", "mileage", "location", "url"]:
            if col not in df.columns:
                df[col] = ""
        # Lowercase brand & model for consistent matching
        df["brand"] = df["brand"].astype(str).str.lower().str.strip()
        df["model"] = df["model"].astype(str).str.lower().str.strip()
        return df
    except Exception as e:
        st.sidebar.caption(f"DB-Fehler: {e}")
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
# 6. DISPLAY HELPERS
# ==========================================

def _fmt(s):
    """Capitalize brand/model for display. E.g. 'mercedes-benz' -> 'Mercedes-Benz', 'nx' -> 'NX'."""
    if not s:
        return s
    parts = s.split("-")
    return "-".join(p.upper() if (p.isalpha() and len(p) <= 3) else p.capitalize() for p in parts)


# ==========================================
# 7. UI VIEWS
# ==========================================

def view_header():
    """Render the top header bar with logo, sidebar toggle, and back-navigation."""
    # Inject CSS to hide sidebar when toggled off
    if not st.session_state.sidebar_visible:
        st.markdown(
            '<style>[data-testid="stSidebar"] { display: none; }</style>',
            unsafe_allow_html=True,
        )
    col_toggle, col_logo, col_spacer, col_nav = st.columns([0.5, 2.5, 5, 2])
    with col_toggle:
        if st.button("\u2630", key="sidebar_toggle", help="Seitenleiste ein-/ausblenden"):
            st.session_state.sidebar_visible = not st.session_state.sidebar_visible
            st.rerun()
    with col_logo:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if st.session_state.theme == "dark":
            logo_path = os.path.join(base_path, "dark_logo.png")
        else:
            logo_path = os.path.join(base_path, "light_logo.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join(base_path, "logo.png")
        if os.path.exists(logo_path):
            st.markdown(
                f'<div class="av-logo-wrap">'
                f'<img src="data:image/png;base64,{img_to_base64(logo_path)}" width="80" height="80" />'
                f'<div><div class="av-logo-text">AutoValue</div>'
                f'<div class="av-logo-sub">Fahrzeug-Intelligenz</div></div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="av-logo-wrap"><div>'
                '<div class="av-logo-text">AutoValue</div>'
                '<div class="av-logo-sub">Fahrzeug-Intelligenz</div></div></div>',
                unsafe_allow_html=True,
            )
    with col_nav:
        if st.session_state.page != "home":
            st.write("")
            if st.button("Zurück zum Dashboard", key="btn_back"):
                nav("home")
                st.rerun()
    st.markdown(f"<hr style='border:none;border-top:1px solid {T['divider']};margin:0.5rem 0 1.5rem 0;'>",
                unsafe_allow_html=True)



def view_home():
    """Render the home / dashboard page."""
    st.markdown(
        f"""
        <div class="av-hero">
            <h2>Professionelle Fahrzeugbewertung</h2>
            <p>Nutzen Sie Machine-Learning-Modelle, trainiert auf echten Marktdaten,
            für präzise Fahrzeugbewertungen auf dem deutschen und US-Markt.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            f"""
            <div class="av-card">
                <span class="card-tag">Verkäufer</span>
                <h3>Verkäufer-Intelligenz</h3>
                <p>Geben Sie Ihre Fahrzeugdaten ein und erhalten Sie eine
                datenbasierte Marktbewertung. Verstehen Sie, welche Merkmale
                den Preis beeinflussen.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1a, c1b = st.columns(2)
        with c1a:
            if st.button("Bewertung starten -- DE", key="btn_s_de"):
                nav("app", "seller", "DE")
                st.rerun()
        with c1b:
            if st.button("Bewertung starten -- US", key="btn_s_us"):
                nav("app", "seller", "US")
                st.rerun()

    with col2:
        st.markdown(
            f"""
            <div class="av-card">
                <span class="card-tag">Käufer</span>
                <h3>Käufer-Intelligenz</h3>
                <p>Durchsuchen Sie echte Inserate, vergleichen Sie Preise mit
                dem vorhergesagten Marktwert und finden Sie die besten
                Angebote auf dem Markt.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c2a, c2b = st.columns(2)
        with c2a:
            if st.button("Inventar suchen -- DE", key="btn_b_de"):
                nav("app", "buyer", "DE")
                st.rerun()
        with c2b:
            if st.button("Inventar suchen -- US", key="btn_b_us"):
                nav("app", "buyer", "US")
                st.rerun()


def view_app():
    """Render the analysis / chat application page."""
    market = st.session_state.market
    role = st.session_state.role
    db_data = get_market_data(market)
    currency = "EUR" if market == "DE" else "USD"
    csym = "\u20ac" if market == "DE" else "$"
    enc_cats = get_encoder_categories(trained_models, market)

    role_de = "Verkäufer" if role == "seller" else "Käufer"
    st.subheader(f"{role_de}-Analyse  |  Markt: {market}")
    tab_engine, tab_chat = st.tabs(["Analyse", "AutoValue Assistent"])

    with tab_engine:
        # Brand→Model fallback when db_data is unavailable
        BRAND_MODELS = {
            # DE
            "mercedes-benz": ["c-klasse", "e-klasse", "s-klasse", "a-klasse", "b-klasse",
                              "cla", "cls", "glc", "gle", "gla", "glb", "gls",
                              "eqe", "eqs", "eqa", "eqb", "eqc", "v-klasse", "g-klasse"],
            # US
            "ford": ["f-150", "f-250", "f-350"],
            "lexus": ["nx"],
        }
        bm1, bm2 = st.columns(2)
        with bm1:
            brand_options = enc_cats.get("brand", sorted(db_data['brand'].unique()) if not db_data.empty else ["mercedes-benz"])
            brand = st.selectbox("Marke", brand_options, key="sel_brand", format_func=_fmt)
        with bm2:
            # 1) Try db_data first, 2) fallback to BRAND_MODELS, 3) fallback to encoder
            model_options = BRAND_MODELS.get(brand, ["unknown"])
            if not db_data.empty:
                filtered = sorted(db_data[db_data['brand'] == brand]['model'].unique())
                if filtered:
                    model_options = filtered
            model_name = st.selectbox("Modell", model_options, key="sel_model", format_func=_fmt)

        # Advanced options toggle (outside form to avoid _arrow_right bug)
        show_advanced = True
        if role == "buyer":
            show_advanced = st.toggle("Erweiterte Optionen anzeigen", value=False, key="show_advanced")

        with st.form("valuation_form"):
            if market == "DE":
                _render_de_form_fields(enc_cats, role, show_advanced)
            else:
                _render_us_form_fields(enc_cats, db_data, brand, model_name, role, show_advanced)
            submitted = st.form_submit_button("Analyse starten")

        if submitted:
            with st.spinner("Marktwert wird berechnet ..."):
                input_vals = _collect_inputs(market, brand, model_name)
                price, s_vals = predict_price(market, input_vals)
                st.markdown(f"<hr style='border:none;border-top:1px solid {T['divider']};margin:1.5rem 0;'>",
                            unsafe_allow_html=True)
                st.metric("Geschätzter Marktwert", f"{csym}{price:,.2f}")

                if role == "seller" and s_vals:
                    st.markdown("### Einflussfaktoren auf den Preis")
                    fig = plt.figure(figsize=(10, 6))
                    shap.plots.waterfall(s_vals[0], show=False)
                    plt.subplots_adjust(left=0.35, right=0.9)
                    st.pyplot(fig)

                if role == "buyer" and not db_data.empty:
                    st.markdown("### Passende Angebote")
                    matches = db_data[(db_data['brand'] == brand) & (db_data['model'] == model_name)].head(10)
                    if not matches.empty:
                        st.dataframe(matches[["title", "price", "mileage", "url"]], use_container_width=True, hide_index=True)
                        coords = [{"lat": lt, "lon": ln} for lt, ln in [get_coords(l) for l in matches['location']] if lt]
                        if coords:
                            st.map(pd.DataFrame(coords))

                st.markdown("### Empfehlungen")
                recs = generate_recommendations(trained_models, market, input_vals, price, db_data, csym, role)
                if recs:
                    for r in recs:
                        st.info(r["text"])
                else:
                    st.caption("Keine relevanten Empfehlungen für diese Konfiguration gefunden.")

    # ── Chat Tab ──
    with tab_chat:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            chat_m = genai.GenerativeModel(
                "gemini-1.5-flash",
                system_instruction=f"Du bist der AutoValue-Experte für den {market}-Markt. Antworte professionell und präzise auf Deutsch.",
            )
            for m in st.session_state.chat_history:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            if p := st.chat_input("Stellen Sie eine Frage ..."):
                with st.chat_message("user"):
                    st.markdown(p)
                st.session_state.chat_history.append({"role": "user", "content": p})
                resp = chat_m.start_chat(
                    history=[{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in st.session_state.chat_history[:-1]]
                ).send_message(p)
                with st.chat_message("assistant"):
                    st.markdown(resp.text)
                st.session_state.chat_history.append({"role": "assistant", "content": resp.text})
        else:
            st.info("Der Chat-Assistent benötigt einen Gemini API-Key. Fügen Sie GEMINI_API_KEY zu Ihren Streamlit Secrets hinzu.")


def _render_de_form_fields(enc_cats, role, show_advanced):
    """Deutsche Markt-Eingabefelder."""
    st.markdown("#### Fahrzeugdaten")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Kilometerstand", 0, 500000, 50000, 5000, key="de_mileage")
        st.number_input("Alter (Jahre)", 0, 40, 3, key="de_age")
        st.number_input("Leistung (PS)", 30, 1000, 150, key="de_power")
    with c2:
        st.number_input("Vorbesitzer", 1, 10, 1, key="de_owners")
        trans_opts = enc_cats.get("transmission", ["automatic", "manual"])
        st.selectbox("Getriebe", trans_opts, key="de_trans", format_func=_fmt)
        fuel_opts = enc_cats.get("fuel", ["benzin", "diesel", "elektro", "hybrid"])
        st.selectbox("Kraftstoff", fuel_opts, key="de_fuel", format_func=_fmt)
    with c3:
        st.number_input("Garantie (Monate)", 0, 60, 0, key="de_garantie")

    st.markdown("#### Zustand")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.checkbox("TÜV neu", key="de_tuv")
    with d2:
        st.checkbox("Unfallfrei", key="de_unfall")
    with d3:
        st.checkbox("Mängel vorhanden", key="de_mangel")
    with d4:
        st.checkbox("Scheckheft gepflegt", key="de_scheckh")

    if show_advanced:
        st.markdown("#### Ausstattung")
        _render_de_equipment()


def _render_de_equipment():
    """DE Ausstattungs-Checkboxen."""
    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        st.checkbox("Panoramadach", key="de_pano")
        st.checkbox("AMG Line", key="de_amg")
    with e2:
        st.checkbox("Distronic", key="de_distronic")
        st.checkbox("Multibeam LED", key="de_multibeam")
    with e3:
        st.checkbox("4-Zonen Klima", key="de_klima4")
        st.checkbox("2-Zonen Klima", key="de_klima2")
    with e4:
        st.checkbox("Burmester 3D", key="de_burm3d")
        st.checkbox("Burmester Std.", key="de_burmstd")
    with e5:
        st.checkbox("8-fach Bereifung", key="de_reif8")
        st.checkbox("Allwetterreifen", key="de_reifall")


def _render_us_form_fields(enc_cats, db_data, brand, model_name, role, show_advanced):
    """US-Markt Eingabefelder."""
    st.markdown("#### Fahrzeugdaten")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Kilometerstand", 0, 500000, 30000, 5000, key="us_mileage")
        st.number_input("Alter (Jahre)", 0, 40, 3, key="us_age")
        st.number_input("Unfälle", 0, 10, 0, key="us_accidents")
    with c2:
        st.number_input("Vorbesitzer", 1, 10, 1, key="us_owners")
        st.number_input("Zylinder", 0, 16, 6, key="us_cyl")
        st.number_input("Türen", 2, 6, 4, key="us_doors")
    with c3:
        st.number_input("Sitze", 1, 12, 5, key="us_seats")
        dt_opts = enc_cats.get("drivetrain", ["unknown"])
        st.selectbox("Antrieb", dt_opts, key="us_drive", format_func=_fmt)
        fuel_opts = enc_cats.get("fuel", ["gasoline", "diesel", "electric", "hybrid"])
        st.selectbox("Kraftstoff", fuel_opts, key="us_fuel", format_func=_fmt)

    st.markdown("#### Klassifizierung")
    g1, g2 = st.columns(2)
    with g1:
        trans_opts = enc_cats.get("transmission", ["automatic", "manual"])
        st.selectbox("Getriebe", trans_opts, key="us_trans", format_func=_fmt)
        body_opts = enc_cats.get("body_style", ["sedan", "suv", "truck", "coupe"])
        st.selectbox("Karosserie", body_opts, key="us_body", format_func=_fmt)
    with g2:
        use_opts = enc_cats.get("usage_type", ["personal", "fleet"])
        st.selectbox("Nutzungsart", use_opts, key="us_usage", format_func=_fmt)

    if show_advanced:
        st.markdown("#### Erweiterte Optionen")
        _render_us_advanced(enc_cats)


def _render_us_advanced(enc_cats):
    """US erweiterte Klassifizierungsfelder."""
    f1, f2, f3 = st.columns(3)
    with f1:
        trim_opts = enc_cats.get("trim", ["unknown"])
        st.selectbox("Ausstattungslinie", trim_opts, key="us_trim_adv", format_func=_fmt)
    with f2:
        eng_opts = enc_cats.get("engine", ["unknown"])
        st.selectbox("Motor", eng_opts, key="us_engine", format_func=_fmt)
    with f3:
        ext_opts = enc_cats.get("exterior_color", ["unknown"])
        st.selectbox("Aussenfarbe", ext_opts, key="us_ext_color", format_func=_fmt)
    f4, f5 = st.columns(2)
    with f4:
        int_opts = enc_cats.get("interior_color", ["unknown"])
        st.selectbox("Innenfarbe", int_opts, key="us_int_color", format_func=_fmt)
    with f5:
        pass


def _b(key):
    """Read a checkbox boolean from session_state and return 1.0/0.0."""
    return 1.0 if st.session_state.get(key, False) else 0.0


def _collect_inputs(market, brand, model_name):
    """Collect all form widget values from session state into a model-input dict."""
    if market == "DE":
        return {
            "brand": brand, "model": model_name,
            "mileage": float(st.session_state.de_mileage),
            "car_age": float(st.session_state.de_age),
            "power_ps": float(st.session_state.de_power),
            "owners": float(st.session_state.de_owners),
            "transmission": st.session_state.de_trans,
            "fuel": st.session_state.de_fuel,
            "garantie_monate": float(st.session_state.de_garantie),
            "tuv_neu": _b("de_tuv"),
            "unfallfrei": _b("de_unfall"),
            "mangel_vorhanden": _b("de_mangel"),
            "scheckheft_gepflegt": _b("de_scheckh"),
            "ausstattung_pano": _b("de_pano"),
            "ausstattung_amg_line": _b("de_amg"),
            "ausstattung_distronic": _b("de_distronic"),
            "ausstattung_multibeam": _b("de_multibeam"),
            "ausstattung_klima_4_zonen": _b("de_klima4"),
            "ausstattung_klima_2_zonen": _b("de_klima2"),
            "ausstattung_burmester_3d": _b("de_burm3d"),
            "ausstattung_burmester_standard": _b("de_burmstd"),
            "bereifung_8_fach": _b("de_reif8"),
            "bereifung_allwetter": _b("de_reifall"),
        }
    else:
        acc = int(st.session_state.us_accidents)
        own = int(st.session_state.us_owners)
        usage = st.session_state.us_usage
        return {
            "brand": brand, "model": model_name,
            "mileage": float(st.session_state.us_mileage),
            "car_age": float(st.session_state.us_age),
            "accident_count": float(acc),
            "owner_count": float(own),
            "cylinders": float(st.session_state.us_cyl),
            "doors": float(st.session_state.us_doors),
            "seats": float(st.session_state.us_seats),
            "trim": st.session_state.get("us_trim_adv", st.session_state.get("us_trim", "unknown")),
            "drivetrain": st.session_state.us_drive,
            "fuel": st.session_state.us_fuel,
            "transmission": st.session_state.us_trans,
            "body_style": st.session_state.us_body,
            "engine": st.session_state.get("us_engine", "unknown"),
            "exterior_color": st.session_state.get("us_ext_color", "unknown"),
            "interior_color": st.session_state.get("us_int_color", "unknown"),
            "usage_type": usage,
            # Derived binary features
            "one_owner": 1.0 if own == 1 else 0.0,
            "has_accidents": 1.0 if acc > 0 else 0.0,
            "is_used": 1.0,
            "is_cpo": 0.0,
            "is_online": 0.0,
            "is_wholesale": 0.0,
            "personal_use": 1.0 if "personal" in str(usage).lower() else 0.0,
        }


# ==========================================
# 8. MAIN ROUTER
# ==========================================
view_header()
if st.session_state.page == "home":
    view_home()
else:
    view_app()
