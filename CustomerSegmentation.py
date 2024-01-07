#Importing libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn import metrics

#Loading the dataset
df = pd.read_csv('marketing_data.csv')
print("Number of datapoints:", len(df))
df.head()

#Information on features
df.info()

#To replace the NA values with the mean values instead of dropping them to avoid missing important information

mean_income = df['Income'].mean()
mean_income

df['Income'] = df['Income'].fillna(mean_income)

df.isna().sum()

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d/%m/%Y")
dates = []
for i in df["Dt_Customer"]:
    i = i.date()
    dates.append(i)

# Dates of the newest and oldest recorded customer
print("The newest customer's enrollment date in the records:", max(dates))
print("The oldest customer's enrollment date in the records:", min(dates))

print("Total categories in the feature Marital_Status:\n", df["Marital_Status"].value_counts(), "\n")
print("Total categories in the feature Education:\n", df["Education"].value_counts())

#Feature Engineering
#Age of customer
df["Age"] = 2021-df["Year_Birth"]

#Total spendings on various items
df["Spent"] = df["MntWines"]+ df["MntFruits"]+ df["MntMeatProducts"]+ df["MntFishProducts"]+ df["MntSweetProducts"]+ df["MntGoldProds"]

#Deriving living situation by marital status
df["Living_With"]=df["Marital_Status"].replace({"Married":"Partner", "Together":"Partner",
                                                    "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone",
                                                    "Divorced":"Alone", "Single":"Alone",})

#Feature indicating total children living in the household
df["Children"]=df["Kidhome"]+df["Teenhome"]

#Feature for total members in the household
df["Family_Size"] = df["Living_With"].replace({"Alone": 1, "Partner":2})+ df["Children"]

#Feature to confirm parenthood
df["Is_Parent"] = np.where(df.Children> 0, 1, 0)

#Segmenting education levels in three groups
df["Education"]=df["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduate":"Undergraduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

#For clarity
df=df.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})

#Dropping some of the redundant features
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
df = df.drop(to_drop, axis=1)

#To plot some selected features
#Setting up colors prefrences
sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#FFF9ED"})
color_palette = ["#1F78B4", "#33A02C", "#E31A1C", "#FF7F00", "#6A3D9A", "#B15928"] #color blind friendly
cmap = colors.ListedColormap(["#1F78B4", "#33A02C", "#E31A1C", "#FF7F00", "#6A3D9A", "#B15928"])
#Plotting following features
To_Plot = [ "Income", "Recency", "Age", "Spent", "Is_Parent"]
print("Relative Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(df[To_Plot], hue= "Is_Parent",palette= (["#6A3D9A","#B15928"]))
#Taking hue
plt.show()

#Dropping the outliers by setting a cap on Age and income.
df = df[(df["Age"]<90)]
df = df[(df["Income"]<600000)]
print("The total number of data-points after removing the outliers are:", len(df))

#Correlation Matrix

# Select only numeric columns
numeric_data = df.select_dtypes(include=[np.number])

# One-hot encode categorical columns
categorical_data = pd.get_dummies(df[['Education', 'Living_With']])

# Concatenate numeric and one-hot encoded categorical columns
processed_data = pd.concat([numeric_data, categorical_data], axis=1)

# Calculate the correlation matrix
corrmat = processed_data.corr()

# Plot the heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(corrmat, annot=True, cmap=cmap, center=0)
plt.show()

#Get list of categorical variables
s = (df.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)

#Label Encoding the object dtypes.
LE=LabelEncoder()
for i in object_cols:
    df[i]=df[[i]].apply(LE.fit_transform)

df.info()

#Creating a copy of data
df_copy = df.copy()
# creating a subset of dataframe by dropping the features on deals accepted and promotions
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
new_dataset = df_copy.drop(cols_del, axis=1)

#Feature scaling
scaler = StandardScaler()
scaler.fit(new_dataset)
scaled_ds = pd.DataFrame(scaler.transform(new_dataset),columns= new_dataset.columns )
print("All features are now scaled")

#Scaled data to be used for dimension reduction
print("Dataframe to be used for further modelling:")
scaled_ds.head()

#Initiating PCA to reduce dimensions aka features to 2
pca = PCA(n_components=2)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["PC1","PC2"]))

#A 2D Projection Of Data In The Reduced Dimension
x =PCA_ds["PC1"]
y =PCA_ds["PC2"]

#To plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.scatter(x,y, c="blue", marker="o" )
ax.set_title("A 2D Projection Of Data In The Reduced Dimension")
plt.show()

# Elbow Method to determine the number of clusters to be formed
print('Elbow Method to determine the number of clusters to be formed:')
elbow_model = KElbowVisualizer(KMeans(), k=(2, 10))  # Adjust the range of k as needed
elbow_model.fit(PCA_ds)
elbow_model.show()

#Initiating the KMeans Clustering Model
kmeans = KMeans(n_clusters=4, random_state=42)

#Fit the model and predict clusters
cluster_assignment = kmeans.fit_predict(PCA_ds)
PCA_ds["Clusters"] = cluster_assignment

# Adding a new Clusters feature to the original dataframe
df["Clusters"] = cluster_assignment

# Plotting the clusters in 2D
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(x, y, s=40, c=PCA_ds["Clusters"], marker='o', cmap=cmap)
ax.set_title("The Plot Of The Clusters")
ax.legend(*scatter.legend_elements(), title="Clusters")

# Add cluster labels
for cluster_label in range(kmeans.n_clusters):
    cluster_center = kmeans.cluster_centers_[cluster_label]
    ax.text(cluster_center[0], cluster_center[1], "*", color='black',
            ha='center', va='center', weight='bold', fontsize=20)
plt.show()

#Plotting countplot of clusters
colour_palette = ["#1F78B4", "#33A02C", "#E31A1C", "#FF7F00"]
pl = sns.countplot(x=df["Clusters"], palette= colour_palette)
pl.set_title("Distribution Of The Clusters")
plt.show()

pl = sns.scatterplot(data = df,x=df["Spent"], y=df["Income"],hue=df["Clusters"], palette= colour_palette)
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()

plt.figure()
pl=sns.swarmplot(x=df["Clusters"], y=df["Spent"], color= "#CBEDDD", alpha=0.5 )
pl=sns.boxenplot(x=df["Clusters"], y=df["Spent"], palette=colour_palette)
plt.show()

#Creating a feature to get a sum of accepted promotions
df["Total_Promos"] = df["AcceptedCmp1"]+ df["AcceptedCmp2"]+ df["AcceptedCmp3"]+ df["AcceptedCmp4"]+ df["AcceptedCmp5"]
#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=df["Total_Promos"],hue=df["Clusters"], palette= colour_palette)
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()

#Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=df["NumDealsPurchases"],x=df["Clusters"], palette= colour_palette)
pl.set_title("Number of Deals Purchased")
plt.show()

Profile = [ "Kidhome","Teenhome", "Age", "Children", "Family_Size", "Is_Parent", "Education","Living_With"]

for i in Profile:
    plt.figure()
    sns.jointplot(x=df[i], y=df["Spent"], hue =df["Clusters"], kind="kde", palette=colour_palette)
    plt.show()

