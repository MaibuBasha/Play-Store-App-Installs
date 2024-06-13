
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns",None)
#UI & Driver Code Library
import streamlit as st
import joblib
ohe=joblib.load("ohe.pkl")
sc=joblib.load("sc.pkl")
xgb=joblib.load("xgb.pkl")
######Helper Functions######
def process_size(size):
    if 'mb' in size:
        return float(size.strip("mb"))
    elif 'kb' in size:
        return float(size.strip("kb")) / 1024
    elif 'gb' in size:
        return float(size.strip("gb")) * 1024
    else:
        return float(size)
def data_cleaning_process(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()
    #Data Validation
    df['Size'] = df['Size'].apply(lambda x: process_size(x) if isinstance(x, str) else x)
    df=df.rename(columns={'Size':'Size_In_MB'})
    conversion_rates = {'usd': 74, 'cad': 59, 'eur': 88, 'vnd': 0.0032, 'gbp': 102, 'brl': 14, 'krw': 0.067}
    df['Price'] = df.apply(lambda row: row['Price'] * conversion_rates[row['Currency']] if row['Currency'] in conversion_rates else row['Price'], axis=1)
    df['Currency'] = 'inr'
    del df['Currency']
    df=df.rename(columns={"Price":"Price_In_INR"})
    ############ Encoding ######################
    df['Editors Choice'].replace({ True:1, False:0}, inplace=True)
    df['Ad Supported'].replace({ True:1, False:0}, inplace=True)
    df['In App Purchases'].replace({ True:1, False:0}, inplace=True)
    df["Minimum Android"] = df["Minimum Android"].apply(lambda x:str(x)[0:3] if isinstance(x,str) else x)
    df["Minimum Android"] = df["Minimum Android"].astype(float)
    ##### One Hot Encoding #####
    ohedata = ohe.transform(df.loc[:, ["Category", "Content Rating"]]).toarray()
    ohedata=pd.DataFrame(ohedata,columns=ohe.get_feature_names_out())
    df = pd.concat([df, ohedata], axis=1)
    df = df.drop(["Category", "Content Rating"],axis=1)
    ################## Scaling ##################
    df.loc[:,["Rating","Rating Count","Price_In_INR","Size_In_MB","Minimum Android","Released","Last Updated"]] = sc.transform(df.loc[:,["Rating","Rating Count","Price_In_INR","Size_In_MB","Minimum Android","Released","Last Updated"]])
    return df

####### UI & Driver Code ##########

# Refer streamlit documentation given below for the ui code

# https://docs.streamlit.io/library/api-reference

# Introduction
st.title("Estimation of Maximum Installs of an app in Google Play Store")
st.image("google.png")
st.write("Analysed and Estimated Maximum Installs of an app in Google Play Store")

#Data
st.header("Data Taken for Analysis")
inputdata=pd.read_csv("UserInputData.csv")
st.dataframe(inputdata.head())
st.subheader("Enter Below Details..")
col1, col2, col3 = st.columns(3)
with col1:
    Category = st.selectbox("Category", np.sort(inputdata["Category"].unique()), index=None, placeholder="--Select--")
with col2:
    CR = st.selectbox("Content Rating", np.sort(inputdata[inputdata["Category"]==Category]["Content Rating"].unique()), index=None, placeholder="--Select--")
with col3: 
    MA = st.selectbox("Minimum Android Version required", np.sort(inputdata[(inputdata["Category"]==Category)&(inputdata["Content Rating"]==CR)]["Minimum Android"].unique()), index=None, placeholder="--Select--")
raw_input = inputdata[(inputdata.Category == Category) & (inputdata["Content Rating"] == CR) & 
                    (inputdata["Minimum Android"]==MA)].drop('Maximum Installs', axis=1).reset_index(drop=True)
if st.button("Estimate Installs"):
    st.write("Input Data..")
    st.dataframe(raw_input.drop("Last Updated", axis=1))
    raw = raw_input.copy()
    inputdata1 = data_cleaning_process(raw)
    raw_input['Maximum Installs'] = xgb.predict(inputdata1)
    raw_input['Maximum Installs'] = round(raw_input['Maximum Installs'])
    del raw_input['Last Updated']
    st.write("Predicted Results are...")
    st.dataframe(raw_input.sort_values(by="Maximum Installs",ascending=False).reset_index(drop=True).head())
    st.write("Thank You")
