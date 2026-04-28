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
import base64

# ==========================================
# 1. PAGE CONFIGURATION & DYNAMIC THEME
# ==========================================
st.set_page_config(
    page_title="AutoValue | Vehicle Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dynamisches CSS: Reagiert auf System-/Browser-Einstellungen
st.markdown("""
    <style>
        /* Verstecke Streamlit Standard-Elemente */
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}

        /* Dynamischer Hintergrund je nach System-Theme */
        @media (prefers-color-scheme: light) {
            .stApp { background-color: #ffffff !important; }
        }
        @media (prefers-color-scheme: dark) {
            /* Passt sich an dunkle Logos an (Tiefes Navy/Schwarz) */
            .stApp { background-color: #0b1120 !important; }
        }

        /* Modernes Card-Layout, das in beiden Themes funktioniert */
        .card { 
            padding: 2rem; 
            border-radius: 12px; 
            border: 1px solid rgba(128, 128, 128, 0.2);
            background-color: rgba(128, 128, 128, 0.05);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); 
            margin-bottom: 1.5rem;
            min-height: 220px;
        }

        /* Einheitliche Buttons, die die native Theme-Schriftfarbe übernehmen */
        .stButton>button { 
            width: 100%; 
            border-radius: 8px; 
            border: 1px solid rgba(128,128,128,0.3) !important;
            font-weight: 600;
            padding: 0.6rem;
            background-color: rgba(128, 128, 128, 0.1);
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: rgba(128, 128, 128, 0.2);
        }

        /* Transparenter Header */
        [data-testid="stHeader"] { background-color: transparent !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. STATE MANAGEMENT & ROUTING
# ==========================================
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'role' not in st.session_state: st.session_state.role = None
if 'market' not in st.session_state: st.session_state.market = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []


def set_page(page, role=None, market=None):
    st.session_state.page = page
    if role: st.session_state.role = role
    if market: st.session_state.market = market


# ==========================================
# 3. INITIALIZATION & DATA ENGINE
# ==========================================
@st.cache_resource
def init_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


supabase = init_supabase()
geolocator = Nominatim(user_agent="autovalue_v7")
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
        table_name = "listings" if market_code == "DE" else "listing_us"
        res = supabase.table(table_name).select("*").limit(3000).execute()
        df = pd.DataFrame(res.data)

        if not df.empty:
            if 'brand' in df.columns: df['brand'] = df['brand'].astype(str).str.strip().str.title()
            if 'model' in df.columns: df['model'] = df['model'].astype(str).str.strip().str.upper()
        return df
    except Exception as e:
        return pd.DataFrame()


@st.cache_data
def get_coords(loc_string):
    if not loc_string or loc_string == "unbekannt": return None, None
    try:
        loc = geocode(loc_string)
        if loc: return loc.latitude, loc.longitude
    except:
        pass
    return None, None


def predict_price(market, input_data):
    if not trained_models: return 0.0, None
    m_code = "de" if market == "DE" else "us"
    model = trained_models[f"{m_code}_model"]
    encoder = trained_models[f"{m_code}_encoder"]
    num_cols = trained_models[f"{m_code}_num_cols"]

    cat_cols = ["brand", "model", "transmission", "fuel"] if market == "DE" else \
        ["brand", "model", "trim", "drivetrain", "fuel", "transmission", "body_style", "engine", "exterior_color",
         "interior_color", "usage_type"]

    df_input = pd.DataFrame([input_data])
    for col in num_cols:
        if col not in df_input.columns: df_input[col] = 0.0

    for col in cat_cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).str.lower()

    encoded_cats = encoder.transform(df_input[cat_cols])
    df_encoded = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat_cols), index=df_input.index)
    X_final = pd.concat([df_input[num_cols], df_encoded], axis=1)

    prediction = model.predict(X_final)[0]
    shap_values = shap.TreeExplainer(model)(X_final)
    return prediction, shap_values


def get_base64_image(image_path):
    """Liest ein Bild ein und konvertiert es für die HTML-Integration"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None


# ==========================================
# 4. UI VIEWS
# ==========================================
def view_header():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        base_path = os.path.dirname(os.path.abspath(__file__))

        # Base64 Kodierung für dynamischen Theme-Wechsel
        light_b64 = get_base64_image(os.path.join(base_path, "light_logo.png"))
        dark_b64 = get_base64_image(os.path.join(base_path, "dark_logo.png"))

        if light_b64 and dark_b64:
            html_code = f"""
            <style>
                .logo-img {{ width: 100%; max-width: 250px; }}
                @media (prefers-color-scheme: dark) {{
                    .logo-light {{ display: none; }}
                    .logo-dark {{ display: block; }}
                }}
                @media (prefers-color-scheme: light) {{
                    .logo-light {{ display: block; }}
                    .logo-dark {{ display: none; }}
                }}
            </style>
            <img src="data:image/png;base64,{light_b64}" class="logo-img logo-light">
            <img src="data:image/png;base64,{dark_b64}" class="logo-img logo-dark">
            """
            st.markdown(html_code, unsafe_allow_html=True)
        elif light_b64:
            st.image(os.path.join(base_path, "light_logo.png"), use_container_width=True)
        else:
            st.subheader("AutoValue.")

    with col3:
        if st.session_state.page != 'home':
            st.write("<br>", unsafe_allow_html=True)
            if st.button("Zurück zum Start"):
                set_page("home")
                st.rerun()
    st.divider()


def view_home():
    st.markdown("<h2 style='text-align: center; margin-top: -1rem;'>Willkommen bei AutoValue.</h2>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: gray; margin-bottom: 2rem;'>Professionelle Fahrzeugbewertung durch Machine Learning.</p>",
        unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="card"><h3>Verkäufer-Intelligenz</h3><p>Eingeben. Bewerten. Verkaufen.</p><p style="font-size: 0.9rem; color: gray;">Analysieren Sie den Marktwert und Preis-Treiber Ihres Fahrzeugs.</p></div>',
            unsafe_allow_html=True)
        if st.button("Bewertung starten (DE)"):
            set_page("app", "seller", "DE")
            st.rerun()
        if st.button("Bewertung starten (US)"):
            set_page("app", "seller", "US")
            st.rerun()

    with col2:
        st.markdown(
            '<div class="card"><h3>Käufer-Intelligenz</h3><p>Finden. Vergleichen. Entscheiden.</p><p style="font-size: 0.9rem; color: gray;">Finden Sie faire Angebote und verstehen Sie Preisunterschiede.</p></div>',
            unsafe_allow_html=True)
        if st.button("Inventar suchen (DE)"):
            set_page("app", "buyer", "DE")
            st.rerun()
        if st.button("Inventar suchen (US)"):
            set_page("app", "buyer", "US")
            st.rerun()


def view_app():
    market = st.session_state.market
    role = st.session_state.role
    db_data = get_market_data(market)
    currency = "€" if market == "DE" else "$"

    role_title = "Verkäufer" if role == "seller" else "Käufer"
    st.subheader(f"{role_title}-Analyse | Markt: {market}")

    tab_engine, tab_chat = st.tabs(["Analysis Engine", "AutoValue Assistant"])

    with tab_engine:
        with st.form("valuation_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                brands = sorted(db_data['brand'].dropna().unique()) if not db_data.empty else ["Mercedes-Benz"]
                brand = st.selectbox("Marke", brands)

                models = sorted(
                    db_data[db_data['brand'] == brand]['model'].dropna().unique()) if not db_data.empty else [
                    "C-KLASSE"]
                model_name = st.selectbox("Modell", models)

                variants = sorted(db_data[(db_data['brand'] == brand) & (db_data['model'] == model_name)][
                                      'title'].dropna().unique()) if not db_data.empty else ["Standard"]
                variant = st.selectbox("Motorisierung / Version", variants)

                age = st.number_input("Alter (Jahre)", 0, 30, 3)

            with c2:
                mileage = st.number_input("Kilometerstand", 0, 300000, 50000, 5000)
                trans = st.selectbox("Getriebe", ["automatic", "manual"])
                fuel = st.selectbox("Kraftstoff", ["benzin", "diesel", "elektro", "hybrid"])

            with c3:
                if market == "DE":
                    st.write("Zusatzausstattung")
                    opt1 = st.checkbox("Panoramadach")
                    opt2 = st.checkbox("AMG Line")
                else:
                    st.write("Zustand & Details")
                    opt1 = st.checkbox("Unfallfrei")
                    opt2 = st.checkbox("CPO Status")

            submitted = st.form_submit_button("Analyse starten")

        if submitted:
            with st.spinner("Berechne Marktwert..."):
                if market == "DE":
                    if not db_data.empty and 'power_ps' in db_data.columns and variant in db_data['title'].values:
                        power = db_data[db_data['title'] == variant]['power_ps'].mode()[0]
                    else:
                        power = 150

                    input_vals = {"brand": brand, "model": model_name, "car_age": float(age), "mileage": float(mileage),
                                  "transmission": trans, "fuel": fuel, "power_ps": float(power), "owners": 1.0,
                                  "ausstattung_pano": 1.0 if opt1 else 0.0,
                                  "ausstattung_amg_line": 1.0 if opt2 else 0.0}
                else:
                    if not db_data.empty and variant in db_data['title'].values:
                        cyl = db_data[db_data['title'] == variant]['cylinders'].mode()[
                            0] if 'cylinders' in db_data.columns else 6
                        eng = db_data[db_data['title'] == variant]['engine'].mode()[
                            0] if 'engine' in db_data.columns else "unbekannt"
                    else:
                        cyl, eng = 6, "unbekannt"

                    input_vals = {"brand": brand, "model": model_name, "car_age": float(age), "mileage": float(mileage),
                                  "transmission": trans, "fuel": fuel, "cylinders": float(cyl), "engine": eng,
                                  "has_accidents": 1.0 if opt1 else 0.0, "is_cpo": 1.0 if opt2 else 0.0, "doors": 4.0,
                                  "seats": 5.0, "trim": "unbekannt", "drivetrain": "unbekannt",
                                  "body_style": "unbekannt",
                                  "exterior_color": "unbekannt", "interior_color": "unbekannt",
                                  "usage_type": "unbekannt"}

                price, s_vals = predict_price(market, input_vals)
                st.divider()
                st.metric(f"Marktwert ({variant})", f"{price:,.2f} {currency}")

                if role == "seller" and s_vals is not None:
                    st.markdown("### Einflussfaktoren auf den Preis")
                    # Weißer Hintergrund (als "Karte"), damit SHAP Diagramm auch in Dark Mode lesbar bleibt
                    fig = plt.figure(figsize=(10, 6))
                    fig.patch.set_facecolor('white')
                    ax = fig.add_subplot(111)
                    ax.set_facecolor('white')
                    shap.plots.waterfall(s_vals[0], show=False)
                    plt.subplots_adjust(left=0.35, right=0.9)
                    st.pyplot(fig)

                if role == "buyer" and not db_data.empty:
                    st.markdown("### Passende Angebote im Inventar")
                    matches = db_data[(db_data['brand'] == brand) & (db_data['model'] == model_name) & (
                                db_data['title'] == variant)].head(10)

                    if not matches.empty:
                        url_col = "carfax_url" if market == "US" and "carfax_url" in matches.columns else "url"
                        if url_col not in matches.columns: matches[url_col] = "Kein Link verfügbar"

                        st.dataframe(matches[["title", "price", "mileage", url_col]],
                                     column_config={
                                         "price": st.column_config.NumberColumn("Preis", format="%d " + currency),
                                         url_col: st.column_config.LinkColumn("Inserat öffnen")
                                     },
                                     use_container_width=True, hide_index=True)

                        coords = [{"lat": lt, "lon": ln} for lt, ln in [get_coords(l) for l in matches['location']] if
                                  lt]
                        if coords: st.map(pd.DataFrame(coords))
                    else:
                        st.info("Aktuell keine exakten Treffer für diese Motorisierung im Inventar.")

    with tab_chat:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            chat_m = genai.GenerativeModel('gemini-1.5-flash',
                                           system_instruction=f"Du bist der AutoValue Experte für den {market} Automarkt. Antworte seriös und professionell.")

            for m in st.session_state.chat_history:
                with st.chat_message(m["role"]): st.markdown(m["content"])

            if p := st.chat_input("Fragen zum Markt oder Werterhalt?"):
                with st.chat_message("user"): st.markdown(p)
                st.session_state.chat_history.append({"role": "user", "content": p})

                with st.spinner("Analysiere..."):
                    history_formatted = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
                                         for m in st.session_state.chat_history[:-1]]
                    resp = chat_m.start_chat(history=history_formatted).send_message(p)

                    with st.chat_message("assistant"): st.markdown(resp.text)
                    st.session_state.chat_history.append({"role": "assistant", "content": resp.text})
        else:
            st.warning("Chat Assistant nicht konfiguriert (API Key fehlt).")


# ==========================================
# 5. MAIN ROUTER
# ==========================================
view_header()
if st.session_state.page == 'home':
    view_home()
else:
    view_app()