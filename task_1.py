import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder

#load data
df=pd.read_csv('train.csv')
print("---Initial Data Shape---")
print(df.shape)
print("\n---Data Overview---")
print(df.info())

#check and handle missing value
print("\n---Missing Values Before Imputation----")
print(df.isnull().sum())

#fill missing age with median
df['Age']=df['Age'].fillna(df['Age'].median())

# Fill missing Embarked with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

#feature selection
#drop Cabin (too many nulls), Name, Ticket, and PassengerID (not predictive)
df = df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'])
print("\n---Missing Values After Imputation & Selection---")
print(df.isnull().sum())

#encoding categorical variables
#label encode sex(male:1, female:0)
le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])

#one-hot encode embarked
df=pd.get_dummies(df, columns=['Embarked'], drop_first=True)

#Normalizatoin/Feature Scaling
scaler=StandardScaler()
numerical_cols=['Age','Fare','SibSp','Parch']
df[numerical_cols]=scaler.fit_transform(df[numerical_cols])
print("\n---Preprocessed Data Preview---")
print(df.head())

#plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True, cmap='coolwarm',fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()