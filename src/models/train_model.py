import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from joblib import dump

# Read data
data = pd.read_parquet(f"{Path(__file__).parent.parent.parent}/data/processed/pakistan_processed.parquet")

# Get training data (known gname)
full_train = data[data['gname'] != 'Unknown'].reset_index(drop=True)

# count the number of occurrences of each gname in the training subset
value_counts = full_train['gname'].value_counts()

# filter the training subset to only include rows where the value in column 'gname' appears at least 10 times
full_train = full_train.loc[full_train['gname'].isin(value_counts.index[value_counts >= 10])].reset_index(drop=True)

# Split training data into X and y sets
X = full_train.drop(columns=['gname'])
y = full_train['gname']

# Transform response from string to numeric data
le = LabelEncoder()
y = le.fit_transform(y)

# Create lists of numerical and categorical columns in X data
numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(exclude=np.number).columns

# Create a preprocessor for tree-based models
preprocessor = ColumnTransformer([
    ('cat', Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), categorical_cols),
    ('num', Pipeline([
        ('imputer', SimpleImputer(fill_value=0)),
        ('normalizer', Normalizer('max'))
        ]), numeric_cols)
    ])

# Stacking Model
estimators = [('xgb', XGBClassifier()), ('lgbm', LGBMClassifier()), ('rf', RandomForestClassifier())]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3)

# ML Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', clf)])

# Train pipeline preprocessor and model
pipeline.fit(X, y)

# Save the pickled object to disk.
dump(pipeline, f"{Path(__file__).parent.parent.parent}/models/final_stacking_model.joblib")
