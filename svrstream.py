import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("svr_dataset.csv")

# ---------------- OUTLIER HANDLING ---------------- #
def iqr(col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr_value = q3 - q1

    lower = q1 - 1.5 * iqr_value
    upper = q3 + 1.5 * iqr_value

    df[col] = df[col].clip(lower, upper)

iqr("income")
iqr("loan_amount")
iqr("credit_score")

# ---------------- HANDLE MISSING VALUES ---------------- #
for col in ["age", "income", "loan_amount", "credit_score"]:
    df[col] = df[col].fillna(df[col].median())

for col in ["city", "employment_type"]:
    df[col] = df[col].fillna(df[col].mode()[0])

# ---------------- ONE HOT ENCODING ---------------- #
df = pd.get_dummies(
    df,
    columns=["city", "employment_type"],
    drop_first=False
)

# ---------------- FEATURES & TARGET ---------------- #
X = df.drop("target", axis=1)
y = df["target"]

# Convert to numeric
X = X.astype(float)
y = y.astype(float)

# ---------------- TRAIN TEST SPLIT ---------------- #
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- SCALING ---------------- #
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ---------------- MODEL ---------------- #
model = SVR(kernel="rbf")
model.fit(x_train_scaled, y_train)

# ---------------- STREAMLIT UI ---------------- #
st.title("Loan Default Prediction")

age = st.number_input(
    "Enter age",
    min_value=18,
    max_value=100,
    value=25
)

income = st.number_input(
    "Enter income",
    value=50000.0
)

loan_amount = st.number_input(
    "Enter loan amount",
    value=100000.0
)

credit_score = st.number_input(
    "Enter credit score",
    min_value=0,
    max_value=900,
    value=700
)

city = st.selectbox(
    "Select City",
    ["Bangalore", "Chennai", "Hyderabad", "Mumbai"]
)

employment_type = st.selectbox(
    "Employment Type",
    ["Salaried", "Self-Employed", "Unemployed"]
)

# ---------------- USER INPUT ---------------- #
input_data = pd.DataFrame([{
    "age": age,
    "income": income,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
    "city": city,
    "employment_type": employment_type
}])

# Same encoding as training data
input_data = pd.get_dummies(
    input_data,
    columns=["city", "employment_type"]
)

# Match training columns
input_data = input_data.reindex(
    columns=X.columns,
    fill_value=0
)

# Convert numeric
input_data = input_data.astype(float)

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)

st.success(
    f"Predicted Value: {prediction[0]:.2f}"
)
