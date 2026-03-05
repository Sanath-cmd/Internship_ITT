import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go


model = pickle.load(open("house_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
default_values = pickle.load(open("default_feature_values.pkl", "rb"))
categorical_cols = pickle.load(open("categorical_cols.pkl", "rb"))
final_feature_order = pickle.load(open("final_feature_order.pkl", "rb"))


st.title("🏠 House Price Predictor 🏠")
st.write("Enter important features. Remaining will use default values.")

user_inputs = {}


user_inputs["OverallQual"] = st.slider("Overall Quality", 1, 10, 5)
user_inputs["GrLivArea"] = st.slider("Living Area (sq ft)", 500, 5000, 1500)
user_inputs["GarageCars"] = st.slider("Garage Cars", 0, 5, 2)
user_inputs["TotalBsmtSF"] = st.slider("Total Basement SF", 0, 3000, 800)
user_inputs["YearBuilt"] = st.slider("Year Built", 1900, 2025, 2000)


user_inputs["Neighborhood"] = st.selectbox(
    "Neighborhood",
    ["NAmes","Randall","CollgCr","OldTown","Edwards","Somerst","NridgHt",
     "Gilbert","Sawyer","NWAmes","SawyerW","Mitchel","BrkSide","Crawfor",
     "IDOTRR","Timber","NoRidge","StoneBr","SWISU","ClearCr","MeadowV",
     "BrDale","Blmngtn","Veenker","NPkVill","Blueste"]
)

user_inputs["ExterQual"] = st.selectbox("Exterior Quality", ['TA','Gd','Ex','Fa'])
user_inputs["KitchenQual"] = st.selectbox("Kitchen Quality", ['TA','Gd','Ex','Fa'])
user_inputs["GarageFinish"] = st.selectbox("Garage Finish", ['Unf','Fin','Rfn'])
user_inputs["SaleCondition"] = st.selectbox(
    "Sale Condition",
    ['Normal','Abnorml','Family','Partial','Alloca','AdjLand','Con',
     'ConLD','ConLI','ConLw','Duplex','PreCon']
)


if st.button("Predict Price"):

    input_data = default_values.copy()
    input_data.update(user_inputs)

    input_df = pd.DataFrame([input_data])
    input_df['Age_house'] = 2026 - input_df['YearBuilt']



    for col in categorical_cols:
        if col not in input_df.columns:
            input_df[col] = default_values[col]

    encoded = encoder.transform(input_df[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    numeric_df = input_df.drop(columns=categorical_cols)

    final_df = pd.concat(
        [numeric_df.reset_index(drop=True),
         encoded_df.reset_index(drop=True)],
        axis=1
    )

    final_df = final_df.reindex(columns=final_feature_order, fill_value=0)

    final_scaled = scaler.transform(final_df)

    prediction = model.predict(final_scaled)[0]
    prediction = np.expm1(prediction)

    st.success(f"Predicted House Price: ${prediction:,.2f}")
    st.metric("Model RMSE", "26,288")
    st.metric("Model Type", "XGBoost Regressor")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={"text": "Predicted Price"},
        gauge={"axis": {"range": [0, 500000]}}
    ))

    st.plotly_chart(fig)

with st.sidebar:
    st.title("About The App")
    st.write("""
This app predicts house prices using an XGBoost Regressor.
User inputs top 10 influential features.
Remaining features use default dataset values.
""")