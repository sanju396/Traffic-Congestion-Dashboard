# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os


def section_title(title, icon =" "):
    st.markdown(
        f"""
        <h3 style="color:#4E73DF; padding: 8px 0; font-size:26px;">
            {icon} {title}
        </h3>
        """,
        unsafe_allow_html=True
    )



DEFAULT_CSV_PATH = "Banglore_traffic_Dataset.csv"
TARGET_COL = "Congestion Level"

st.set_page_config(page_title="Traffic Congestion Dashboard", layout="wide")
st.title("🚦 Bangalore Traffic Congestion — Interactive Dashboard")



st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Dataset loaded successfully!")
else:
    if os.path.exists(DEFAULT_CSV_PATH):
        df = pd.read_csv(DEFAULT_CSV_PATH)
        st.sidebar.info(f"Loaded default dataset.")
    else:
        st.sidebar.error("No dataset found. Upload a CSV.")
        st.stop()



section_title("Dataset Preview")
st.dataframe(df.head(20))

st.subheader("Data Cleaning & Preprocessing")

df.columns = [c.strip() for c in df.columns]


if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["year"] = df["Date"].dt.year.fillna(df["Date"].dt.year.mode()[0]).astype(int)
    df["month"] = df["Date"].dt.month.fillna(df["Date"].dt.month.mode()[0]).astype(int)
    df["day"] = df["Date"].dt.day.fillna(df["Date"].dt.day.mode()[0]).astype(int)
    st.write("Extracted `year`, `month`, `day`.")



for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    else:
        df[col] = df[col].fillna(df[col].mean())



categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()



if TARGET_COL not in df.columns:
    st.error(f"Target column '{TARGET_COL}' missing.")
    st.stop()

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(df[TARGET_COL].mean())



encoders = {}
encoded_names = []

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    le.fit(df[col])
    df[col + "_code"] = le.transform(df[col])
    encoders[col] = le
    encoded_names.append(col + "_code")

numeric_for_model = [c for c in numeric_cols if c != TARGET_COL] + encoded_names



area_candidates = ["Area Name", "Area", "area", "area name"]
road_candidates = ["Road/Intersection Name", "Road/Intersection", "Road", "road"]

area_col = next((c for c in area_candidates if c in df.columns), None)
road_col = next((c for c in road_candidates if c in df.columns), None)

if area_col:
    st.success(f"Area column detected: **{area_col}**")
else:
    st.info("No Area column found.")

if road_col:
    st.success(f"Road column detected: **{road_col}**")
else:
    st.info("No Road column found.")

area_list = sorted(df[area_col].astype(str).unique()) if area_col else []
road_list = sorted(df[road_col].astype(str).unique()) if road_col else []



with st.expander("Show Area & Road Mappings"):
    if area_col and area_col in encoders:
        area_map = pd.DataFrame({
            "Area Name": encoders[area_col].classes_,
            "Code": encoders[area_col].transform(encoders[area_col].classes_)
        })
        st.markdown("### Area → Code")
        st.dataframe(area_map)

    if road_col and road_col in encoders:
        road_map = pd.DataFrame({
            "Road Name": encoders[road_col].classes_,
            "Code": encoders[road_col].transform(encoders[road_col].classes_)
        })
        st.markdown("### Road → Code")
        st.dataframe(road_map)



st.sidebar.header("Route Query")

source_area = st.sidebar.selectbox("Source Area", ["(none)"] + area_list) if area_list else "(none)"
dest_area = st.sidebar.selectbox("Destination Area", ["(none)"] + area_list) if area_list else "(none)"

st.subheader("Route / Area Analysis")

if source_area != "(none)":
    src = df[df[area_col] == source_area]
    st.write(f"Records in **{source_area}**:", len(src))
    if not src.empty:
        st.metric("Mean Congestion", round(src[TARGET_COL].mean(), 2))

if dest_area != "(none)":
    dst = df[df[area_col] == dest_area]
    st.write(f"Records in **{dest_area}**:", len(dst))
    if not dst.empty:
        st.metric("Mean Congestion", round(dst[TARGET_COL].mean(), 2))


st.subheader("Train Model")

X = df[numeric_for_model]
y = df[TARGET_COL]

mask = ~X.isnull().any(axis=1)
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
st.write(f"MAE: {mean_absolute_error(y_test, preds):.4f}")
st.write(f"R²: {r2_score(y_test, preds):.4f}")

fi = pd.Series(model.feature_importances_, index=numeric_for_model).sort_values(ascending=False)
st.bar_chart(fi.head(12))

joblib.dump(model, "traffic_congestion_model.pkl")
st.sidebar.success("Model saved as traffic_congestion_model.pkl")



st.subheader("Interactive Prediction")

INTEGER_COLUMNS = [
    "Cyclist Count", "Pedestrian Count", "Vehicle Count", "Traffic Volume",
    "year", "month", "day"
]

user_input = {}

for col in numeric_for_model:
    if col.endswith("_code"):
        orig = col[:-5]
        if orig in encoders:
            options = encoders[orig].classes_.tolist()
            choice = st.selectbox(f"{orig}", options)
            code = int(encoders[orig].transform([choice])[0])
            user_input[col] = code
    else:
        if col in INTEGER_COLUMNS:
            minv = int(df[col].min())
            maxv = int(df[col].max())
            val = st.number_input(f"{col}", value=int(df[col].mean()), min_value=minv, max_value=maxv, step=1)
        else:
            val = st.number_input(f"{col}", value=float(df[col].mean()))
        user_input[col] = val

input_df = pd.DataFrame([user_input], columns=numeric_for_model)

st.write("Input Encoded:")
st.dataframe(input_df)

if st.button("Predict Congestion"):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Congestion Level: **{pred:.2f}**")




st.subheader("Correlation Heatmap")
if st.checkbox("Show Heatmap"):
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
