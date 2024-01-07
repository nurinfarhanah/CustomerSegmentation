import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from PIL import Image
import io

def main():
    """
    The main function
    """
    df = pd.read_csv('marketing_data.csv')


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

    # Feature scaling
    scaler = StandardScaler()
    scaler.fit(new_dataset)
    scaled_ds = pd.DataFrame(scaler.transform(new_dataset), columns=new_dataset.columns)


    # Initiating PCA to reduce dimensions aka features to 2
    pca = PCA(n_components=2)
    pca.fit(scaled_ds)
    PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["PC1", "PC2"]))

    #A 2D Projection Of Data In The Reduced Dimension
    x =PCA_ds["PC1"]
    y =PCA_ds["PC2"]

    # Elbow Method to determine the number of clusters to be formed
    print('Elbow Method to determine the number of clusters to be formed:')
    elbow_model = KElbowVisualizer(KMeans(), k=(2, 10))  # Adjust the range of k as needed
    elbow_model.fit(PCA_ds)

    #Initiating the KMeans Clustering Model
    kmeans = KMeans(n_clusters=4, random_state=42)

    #Fit the model and predict clusters
    cluster_assignment = kmeans.fit_predict(PCA_ds)
    PCA_ds["Clusters"] = cluster_assignment

    # Adding a new Clusters feature to the original dataframe
    df["Clusters"] = cluster_assignment

    st.header("Model Evaluation \n Note: This model evaluation can be used if the number of clusters(k) used is the same as the value stated in the Elbow Method which is **4**.")
    st.title("Distribution of Clusters")
    #Plotting countplot of clusters
    colour_palette = ["#1F78B4", "#33A02C", "#E31A1C", "#FF7F00"]
    fig, ax = plt.subplots()
    sns.countplot(x=df["Clusters"], palette=colour_palette, ax=ax)
    ax.set_title("Distribution Of The Clusters")
    st.pyplot(fig)


    st.title("Cluster's Profile Based On Income And Spending")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=df["Spent"], y=df["Income"], hue=df["Clusters"], palette=colour_palette, ax=ax)
    ax.set_title("Cluster's Profile Based On Income And Spending")
    ax.legend()
    st.pyplot(fig)

    st.title("Cluster's Profile Based On Spending")
    fig, ax = plt.subplots()
    # Swarmplot
    sns.swarmplot(x=df["Clusters"], y=df["Spent"], color="#CBEDDD", alpha=0.5, ax=ax)
    # Boxenplot
    sns.boxenplot(x=df["Clusters"], y=df["Spent"], palette=colour_palette, ax=ax)
    ax.set_title("Cluster's Profile Based On Spending")
    st.pyplot(fig)


    st.title("Promotions Accepted by Customers")
    #Creating a feature to get a sum of accepted promotions
    df["Total_Promos"] = df["AcceptedCmp1"]+ df["AcceptedCmp2"]+ df["AcceptedCmp3"]+ df["AcceptedCmp4"]+ df["AcceptedCmp5"]
    #Plotting count of total campaign accepted.
    fig, ax = plt.subplots()
    sns.countplot(x=df["Total_Promos"], hue=df["Clusters"], palette=colour_palette, ax=ax)
    ax.set_title("Count Of Promotion Accepted")
    ax.set_xlabel("Number Of Total Accepted Promotions")
    ax.legend()
    st.pyplot(fig)

    st.title("Number of Deals Purchased")
    #Plotting the number of deals purchased
    fig, ax = plt.subplots()
    sns.boxenplot(y=df["NumDealsPurchases"], x=df["Clusters"], palette=colour_palette, ax=ax)
    ax.set_title("Number of Deals Purchased")
    st.pyplot(fig)


if __name__ == "__main__":
    main()
