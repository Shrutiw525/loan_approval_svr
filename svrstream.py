import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#iqr 
df=pd.read_csv("svr_dataset.csv")
def iqr(col):
    q1=df[col].quantile(0.25)
    q3=df[col].quantile(0.75)
    iqr=q3-q1
    lower_bound=q1-1.5*iqr
    upper_bound=q3+1.5*iqr
    df[col]=df[col].clip(lower_bound,upper_bound)
iqr("income")
iqr("loan_amount")
iqr("credit_score")
df.plot(kind="box",layout=(3,2),subplots=True)
df["income"].fillna(df["income"].median(),inplace=True)
df["loan_amount"].fillna(df["loan_amount"].median(),inplace=True)
df["credit_score"].fillna(df["credit_score"].median(),inplace=True)
df.isnull().sum()
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(sparse_output=False)
encoded_data=encoder.fit_transform(df[["city"]])
encoded_df=pd.DataFrame(encoded_data,columns=encoder.get_feature_names_out(["city"])).astype(int)
df=pd.concat([df.drop("city",axis=1),encoded_df],axis=1)
encoder=OneHotEncoder(sparse_output=False)
encoded_data=encoder.fit_transform(df[["employment_type"]])
encoded_df=pd.DataFrame(encoded_data,columns=encoder.get_feature_names_out(["employment_type"])).astype(int)
df=pd.concat([df.drop("employment_type",axis=1),encoded_df],axis=1)
x=df.drop("target",axis=1)
y=df["target"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)   
#scaling the values
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
from sklearn.svm import SVR
model=SVR(kernel="rbf")
model.fit(x_train,y_train)

#prediction from user input
import streamlit as st
st.title("Loan Default Prediction")
st.write("SVR")
age=st.number_input("Enter your age:",min_value=18,max_value=100) #age should be between 18 and 100
income=st.number_input("Enter your income:")
loan_amount=st.number_input("Enter the loan amount:")
credit_score=st.number_input("Enter your credit score:")
#city is taken as mumbai,delhi,bangalore and map it to cityA,cityB,cityC
city=st.selectbox("Select your city:",["Banglore","Chennai","Hyderabad","Mumbai"])
employment_type=st.selectbox("Select your employment type:",["Salaried","Self-Employed","Unemployed"])
if city=="Banglore":
    city_Bangalore=1
    city_Chennai=0
    city_Hyderabad=0
    city_Mumbai=0
elif city=="Chennai":
    city_Bangalore=0
    city_Chennai=1
    city_Hyderabad=0
    city_Mumbai=0
elif city=="Hyderabad":
    city_Bangalore=0
    city_Chennai=0
    city_Hyderabad=1
    city_Mumbai=0
elif city=="Mumbai":
    city_Bangalore=0
    city_Chennai=0
    city_Hyderabad=0
    city_Mumbai=1
if employment_type=="Salaried":
    employment_type_Salaried=1
    employment_type_Self_Employed=0
    employment_type_Unemployed=0
elif employment_type=="Self-Employed":
    employment_type_Salaried=0
    employment_type_Self_Employed=1
    employment_type_Unemployed=0
elif employment_type=="Unemployed":
    employment_type_Salaried=0
    employment_type_Self_Employed=0
    employment_type_Unemployed=1
input_data=[[age,income,loan_amount,credit_score,city_Bangalore,city_Chennai,city_Hyderabad,city_Mumbai,employment_type_Salaried,employment_type_Self_Employed,employment_type_Unemployed]]
input_data_scaled=scaler.transform(input_data)
prediction=model.predict(input_data_scaled)
if prediction[0]==1:
    st.write("You are likely to default on the loan.")
else:
    st.write("You are unlikely to default on the loan.")
