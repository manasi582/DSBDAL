import pandas as pd
import numpy as np
df = pd.read_csv("Documents/college/sem6/DSBDA/pracpractice/train.csv")
df.head()
df.isnull().sum()
df.describe()
df.info()
df.shape
df.dtypes
df['Age'] = df['Age'].astype(float)
df['Survived'] = df['Survived'].astype(int)
df['Fare'] = (df['Fare'] - df['Fare'].min()) / (df['Fare'].max() - df['Fare'].min())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

df = pd.get_dummies(df, columns=['Embarked'])
df.head()
df.info()

df.head()