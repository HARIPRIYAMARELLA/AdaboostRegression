import streamlit as st
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Load dataset
df = pd.read_csv("insurance.csv")

# Encode categorical columns
encoder = LabelEncoder()

for col in df.columns:
    
    if df[col].dtype == "object":
        
        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )

# Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    
    X,
    y,
    test_size=0.2,
    random_state=42
    
)

# Base model
base_model = DecisionTreeRegressor(
    max_depth=4
)

# AdaBoost model
model = AdaBoostRegressor(
    
    estimator=base_model,
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
    
)

model.fit(X_train, y_train)

# Create pickle
pickle.dump(
    model,
    open("adaboost_regression.pkl", "wb")
)

# Load pickle
loaded_model = pickle.load(
    open("adaboost_regression.pkl", "rb")
)

st.title("Insurance Cost Prediction using AdaBoost")

st.write("Enter Details")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

sex = st.selectbox(
    "Sex",
    ["male","female"]
)

bmi = st.number_input(
    "BMI",
    value=25.0
)

children = st.number_input(
    "Children",
    min_value=0,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes","no"]
)

region = st.selectbox(
    "Region",
    [
        "southwest",
        "southeast",
        "northwest",
        "northeast"
    ]
)

# Encoding user inputs
sex = 1 if sex=="male" else 0

smoker = 1 if smoker=="yes" else 0

region_dict = {
    
    "southwest":3,
    "southeast":2,
    "northwest":1,
    "northeast":0
    
}

region = region_dict[region]

if st.button("Predict Insurance Cost"):

    input_data = [[
        
        age,
        sex,
        bmi,
        children,
        smoker,
        region
        
    ]]

    prediction = loaded_model.predict(
        input_data
    )

    st.success(
        f"Estimated Insurance Cost: ${prediction[0]:,.2f}"
    )