import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Dataset overview
st.markdown("# Dataset Used")
st.sidebar.markdown("# Dataset source: ")
st.sidebar.write("*Platform:* Kaggle")
st.sidebar.write("*Titles:*")
st.sidebar.write("Customer Segmentation: Clustering 🛍️🛒🛒")
st.sidebar.write("*Authors:*")
st.sidebar.write("KARNIKA KAPOOR")
st.sidebar.write("*Url:* ")
st.sidebar.write("https://www.kaggle.com/code/karnikakapoor/customer-segmentation-clustering/input")
st.sidebar.write("*About:* This dataset contains marketing data with information on customers' demographics, purchases, and campaign responses.")
st.sidebar.write("*Features:* ")
st.sidebar.write("ID: The unique identifier for each customer.")
st.sidebar.write("Year_Birth: The birth year of the customer.")
st.sidebar.write("Education: The educational level of the customer.")
st.sidebar.write("Marital_Status: The marital status of the customer.")
st.sidebar.write("Income: The annual income of the customer.")
st.sidebar.write("Kidhome: The number of children in the customer's household.")
st.sidebar.write("Teenhome: The number of teenagers in the customer's household.")
st.sidebar.write("Dt_Customer: The date when the customer made their latest shop.")
st.sidebar.write("Recency: The number of days since the customer's last purchase.")
st.sidebar.write("MntWines: The amount spent on wines.")
st.sidebar.write("MntFruits: The amount spent on fruits.")
st.sidebar.write("MntMeatProducts: The amount spent on meat products.")
st.sidebar.write("MntFishProducts: The amount spent on fish products.")
st.sidebar.write("MntSweetProducts: The amount spent on sweet products.")
st.sidebar.write("MntGoldProds: The amount spent on gold products.")
st.sidebar.write("NumDealsPurchases: The number of purchases made with a deals.")
st.sidebar.write("NumWebPurchases: The number of purchases made through the company's website.")
st.sidebar.write("NumCatalogPurchases: The number of purchases made using a catalog.")
st.sidebar.write("NumStorePurchases: The number of purchases made directly in stores.")
st.sidebar.write("NumWebVisitsMonth: The number of visits to the company's website.")
st.sidebar.write("AcceptedCmp3: Whether the customer accepted the third campaign.")
st.sidebar.write("AcceptedCmp4: Whether the customer accepted the forth campaign.")
st.sidebar.write("AcceptedCmp5: Whether the customer accepted the fifth campaign.")
st.sidebar.write("AcceptedCmp1: Whether the customer accepted the first campaign.")
st.sidebar.write("AcceptedCmp2: Whether the customer accepted the second campaign.")
st.sidebar.write("Complain: Whether the customer has made any complaints.")
st.sidebar.write("Z_CostContact: The cost of contacting the customer.")
st.sidebar.write("Z_Revenue: The revenue generated from contacting the customer.")
st.sidebar.write("Response: Whether the customer responded to the last campaign.")
st.sidebar.write(" ")


def main():
    """
    The main function
    """

    st.header("Raw Dataset: \n")
    df = pd.read_csv('marketing_data.csv')
    st.write(df)

    # To replace the NA values with the mean values instead of dropping them to avoid missing important information

    mean_income = df['Income'].mean()
    mean_income

    df['Income'] = df['Income'].fillna(mean_income)

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d/%m/%Y")
    dates = []
    for i in df["Dt_Customer"]:
        i = i.date()
        dates.append(i)

    # Feature Engineering
    # Age of customer
    df["Age"] = 2021 - df["Year_Birth"]

    # Total spendings on various items
    df["Spent"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df[
        "MntSweetProducts"] + df["MntGoldProds"]

    # Deriving living situation by marital status
    df["Living_With"] = df["Marital_Status"].replace({"Married": "Partner", "Together": "Partner",
                                                      "Absurd": "Alone", "Widow": "Alone", "YOLO": "Alone",
                                                      "Divorced": "Alone", "Single": "Alone", })

    # Feature indicating total children living in the household
    df["Children"] = df["Kidhome"] + df["Teenhome"]

    # Feature for total members in the household
    df["Family_Size"] = df["Living_With"].replace({"Alone": 1, "Partner": 2}) + df["Children"]

    # Feature to confirm parenthood
    df["Is_Parent"] = np.where(df.Children > 0, 1, 0)

    # Segmenting education levels in three groups
    df["Education"] = df["Education"].replace(
        {"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduate": "Undergraduate", "Master": "Postgraduate",
         "PhD": "Postgraduate"})

    # For clarity
    df = df.rename(
        columns={"MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat", "MntFishProducts": "Fish",
                 "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"})

    # Dropping some of the redundant features
    to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
    df = df.drop(to_drop, axis=1)

    # Dropping the outliers by setting a cap on Age and income.
    df = df[(df["Age"] < 90)]
    df = df[(df["Income"] < 600000)]

    # Correlation Matrix

    # Select only numeric columns
    numeric_data = df.select_dtypes(include=[np.number])

    # One-hot encode categorical columns
    categorical_data = pd.get_dummies(df[['Education', 'Living_With']])

    # Concatenate numeric and one-hot encoded categorical columns
    processed_data = pd.concat([numeric_data, categorical_data], axis=1)

    # Calculate the correlation matrix
    corrmat = processed_data.corr()

    # Get list of categorical variables
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)

    # Label Encoding the object dtypes.
    LE = LabelEncoder()
    for i in object_cols:
        df[i] = df[[i]].apply(LE.fit_transform)

    # Creating a copy of data
    df_copy = df.copy()
    # creating a subset of dataframe by dropping the features on deals accepted and promotions
    cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
    new_dataset = df_copy.drop(cols_del, axis=1)

    st.header("Clean Dataset: \n")
    st.write(new_dataset)

if __name__ == "__main__":
    main()
