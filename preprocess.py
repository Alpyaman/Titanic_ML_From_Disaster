"""
Titanic Dataset Preprocessing Module
This module provides data preprocessing and feature engineering functions
for the Titanic survival prediction problem.
Author: Alp Yaman
Date: 2025
"""

import pandas as pd
import numpy as np

def preprocess(filepath):
    """
    Preprocess Titanic dataset with feature engineering and data cleaning.

    This function performs comprehensive preprocessing including:
    - Title extraction from passenger names
    - Missing value imputation
    - Feature engineering (family size, fare per person)
    - One-hot encoding for categorical variables

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing Titanic data (train or test).

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe with engineered features and cleaned data.
        PassengerId is retained for test set submission.

    Examples
    --------
    >>> train_df = preprocess('titanic_data/train.csv')
    >>> test_df = preprocess('titanic_data/test.csv')

    Notes
    -----
    - Age is imputed using median age per title group
    - Fare is imputed with overall median
    - Embarked is imputed with mode (most common port)
    - Categorical variables are one-hot encoded
    """
    
    df = pd.read_csv(filepath)

    # Keep PassengerId for test (will be dropped for training features later)
    # If you want to drop in train, do it outside this function

    # ========== Feature Engineering ==========

    # Extract title from passenger name (Mr., Mrs., Miss., Master., etc.)
    # This captures social status and gender information
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Simplify rare titles into a single 'Rare' category
    # This reduces dimensionality and improves generalization

    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')

    # Standardize similar titles
    df['Title'] = df['Title'].replace('Mlle', 'Miss')  # Mademoiselle -> Miss
    df['Title'] = df['Title'].replace('Ms', 'Miss')    # Ms -> Miss
    df['Title'] = df['Title'].replace('Mme', 'Mrs')    # Madame -> Mrs

    # ========== Missing Value Imputation ==========
    # Fill missing Age using median age of same title group
    # People with the same title tend to have similar ages
    df['Age'] = df.groupby('Title')['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Fill missing Fare with overall median fare
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Fill missing Embarked (port of embarkation) with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # ========== Additional Feature Creation ==========

    # Create FamilySize: total family members aboard (including passenger)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Create FarePerPerson: fare divided by family size
    # Captures individual ticket cost, which may correlate with survival
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # ========== Final Data Cleaning ==========

    # Fill any remaining numeric NaNs with column medians
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # One-hot encode categorical variables
    # drop_first=True to avoid multicollinearity (dummy variable trap)
    df = pd.get_dummies(
        df,
        columns=['Sex', 'Embarked', 'Title'],
        drop_first=True
    )

    # Drop columns that are not useful for modeling
    drop_cols = ['Name', 'Ticket', 'Cabin']
    df.drop(
        columns=[col for col in drop_cols if col in df.columns],
        inplace=True
    )
    
    return df

