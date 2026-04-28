"""AutoValue helper functions: labels, recommendations, utilities."""
import base64
import pandas as pd
import numpy as np

# ── Feature label mappings ─────────────────────────────────────────
DE_BINARY_LABELS = {
    "tuv_neu": "TUV New",
    "scheckheft_gepflegt": "Service Book",
    "bereifung_8_fach": "8-Tire Set",
    "bereifung_allwetter": "All-Weather Tires",
    "unfallfrei": "Accident-Free",
    "mangel_vorhanden": "Defects Present",
    "ausstattung_distronic": "Distronic",
    "ausstattung_multibeam": "Multibeam LED",
    "ausstattung_klima_4_zonen": "4-Zone Climate",
    "ausstattung_klima_2_zonen": "2-Zone Climate",
    "ausstattung_burmester_3d": "Burmester 3D",
    "ausstattung_burmester_standard": "Burmester Standard",
    "ausstattung_amg_line": "AMG Line",
    "ausstattung_pano": "Panoramic Roof",
}

DE_CONDITION_KEYS = ["tuv_neu", "unfallfrei", "mangel_vorhanden", "scheckheft_gepflegt"]
DE_EQUIP_KEYS = [
    "bereifung_8_fach", "bereifung_allwetter",
    "ausstattung_distronic", "ausstattung_multibeam",
    "ausstattung_klima_4_zonen", "ausstattung_klima_2_zonen",
    "ausstattung_burmester_3d", "ausstattung_burmester_standard",
    "ausstattung_amg_line", "ausstattung_pano",
]


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_encoder_categories(trained_models, market):
    """Extract valid categories for each categorical feature."""
    if not trained_models:
        return {}
    m_code = "de" if market == "DE" else "us"
    encoder = trained_models[f"{m_code}_encoder"]
    return {
        name: sorted(cats.tolist())
        for name, cats in zip(encoder.feature_names_in_, encoder.categories_)
    }


def predict_price_fast(trained_models, market, input_data):
    """Predict price without SHAP (fast, for recommendations)."""
    if not trained_models:
        return 0.0
    m_code = "de" if market == "DE" else "us"
    model = trained_models[f"{m_code}_model"]
    encoder = trained_models[f"{m_code}_encoder"]
    num_cols = trained_models[f"{m_code}_num_cols"]
    cat_cols = (
        ["brand", "model", "transmission", "fuel"] if market == "DE"
        else ["brand", "model", "trim", "drivetrain", "fuel", "transmission",
              "body_style", "engine", "exterior_color", "interior_color", "usage_type"]
    )
    df = pd.DataFrame([input_data])
    for col in num_cols:
        if col not in df.columns:
            df[col] = 0.0
    enc = encoder.transform(df[cat_cols])
    df_enc = pd.DataFrame(enc, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
    X = pd.concat([df[num_cols], df_enc], axis=1)
    return float(model.predict(X)[0])


def generate_recommendations(trained_models, market, input_data, base_price, db_data, currency_symbol):
    """Generate savings/upgrade recommendations by toggling features."""
    recs = []
    if not trained_models:
        return recs

    if market == "DE":
        # Toggle each binary feature
        all_binary = DE_CONDITION_KEYS + DE_EQUIP_KEYS
        for key in all_binary:
            if key not in input_data:
                continue
            alt = dict(input_data)
            current_val = alt[key]
            alt[key] = 0.0 if current_val == 1.0 else 1.0
            alt_price = predict_price_fast(trained_models, market, alt)
            diff = base_price - alt_price
            label = DE_BINARY_LABELS.get(key, key)
            if current_val == 1.0 and diff > 100:
                recs.append({
                    "text": f"Without '{label}' the value drops by ~{abs(diff):,.0f} {currency_symbol}.",
                    "saving": diff, "type": "value_driver"
                })
            elif current_val == 0.0 and alt_price > base_price + 100:
                gain = alt_price - base_price
                recs.append({
                    "text": f"Adding '{label}' could increase value by ~{gain:,.0f} {currency_symbol}.",
                    "saving": gain, "type": "upgrade"
                })

        # Try alternative models from same brand
        if not db_data.empty:
            brand = input_data.get("brand", "")
            current_model = input_data.get("model", "")
            alt_models = [m for m in db_data[db_data["brand"] == brand]["model"].unique()
                          if m != current_model][:5]
            for alt_m in alt_models:
                alt = dict(input_data)
                alt["model"] = alt_m
                try:
                    alt_price = predict_price_fast(trained_models, market, alt)
                    diff = base_price - alt_price
                    if diff > 500:
                        recs.append({
                            "text": f"A '{alt_m}' with the same specs would cost ~{diff:,.0f} {currency_symbol} less.",
                            "saving": diff, "type": "alternative"
                        })
                except:
                    pass

    else:  # US
        # Try alternative models
        if not db_data.empty:
            brand = input_data.get("brand", "")
            current_model = input_data.get("model", "")
            alt_models = [m for m in db_data[db_data["brand"] == brand]["model"].unique()
                          if m != current_model][:5]
            for alt_m in alt_models:
                alt = dict(input_data)
                alt["model"] = alt_m
                try:
                    alt_price = predict_price_fast(trained_models, market, alt)
                    diff = base_price - alt_price
                    if diff > 500:
                        recs.append({
                            "text": f"A '{alt_m}' with the same specs would cost ~${diff:,.0f} less.",
                            "saving": diff, "type": "alternative"
                        })
                except:
                    pass

        # Try fewer cylinders
        if input_data.get("cylinders", 0) > 4:
            alt = dict(input_data)
            alt["cylinders"] = max(4.0, alt["cylinders"] - 2.0)
            try:
                alt_price = predict_price_fast(trained_models, market, alt)
                diff = base_price - alt_price
                if diff > 300:
                    recs.append({
                        "text": f"A {int(alt['cylinders'])}-cylinder engine would save ~${diff:,.0f}.",
                        "saving": diff, "type": "downgrade"
                    })
            except:
                pass

    recs.sort(key=lambda r: r["saving"], reverse=True)
    return recs[:5]
