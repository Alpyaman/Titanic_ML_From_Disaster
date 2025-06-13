import pandas as pd
import numpy as np

def preprocess(filepath):
    df = pd.read_csv(filepath)

    # Keep PassengerId for test (will be dropped for training features later)
    # If you want to drop in train, do it outside this function

    # Example feature engineering and cleaning

    # Title extraction from Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Simplify titles
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'
    )
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Fill missing Age based on Title median
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

    # Fill missing Fare with median
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Fill Embarked missing values with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Create FamilySize feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Example: Fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # Fill any remaining numeric NaNs with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Encode categorical variables - example with one-hot encoding
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

    # Drop columns unlikely to be useful or redundant
    drop_cols = ['Name', 'Ticket', 'Cabin']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    return df
