# The preprocessing. remove redundant columns. fill or remove holes
# assign values to the nominal variables. ie. 1 for saiyan 2 for human. ie parameterize

# in order correct Typos/wrong data -> check correlations and drop if too high -> impute missing values ->
# drop useless columns/lines with too many missing vals -> encode categorical vals using one hot -> standardizing numerics
# -> subset into villain hero -> split into test and train -> train a lin reg -> eval
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
import pandas as pd

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        super().__init__(threshold, features_to_keep = None)
        
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            corr_matrix = X.corr().abs()
        else:
            X = pd.DataFrame(X)
            corr_matrix = X.corr().abs()
            
class OutlierRemoval(BaseEstimator, TransformerMixin):  # IQR based, removes outlier by clipping them
    def __init__(self, multiplier=1.5, numerical_features=None):
        self.multiplier = multiplier
        self.numerical_features = numerical_features
        self.lower_bound = {}
        self.upper_bound = {}
        
    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        features = self.numerical_features or df.select_dtypes(include=['number']).columns
        
        for col in features:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bound[col] = Q1 - self.multiplier * IQR
            self.upper_bound[col] = Q3 + self.multiplier * IQR
            
        return self
    
    def transform(self, X):
        df = pd.DataFrame(X).copy()
        features = self.numerical_features or df.select_dtypes(include=['number']).columns

        for col in features:
            df[col] = df[col].clip(lower=self.lower_bound[col], upper=self.upper_bound[col])
        
        return df

def win_probOutlierRemoval(X):
    df = pd.DataFrame(X).copy()
    df[df > 1] = df[df > 1] * 0.1   # multiplies the outlier with 0.1 as this makes some sense with the comman just being wrong there
    return df

def clean_role_feature(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ['hero', 'h3ro', 'her0']:
            return 'Hero'
        elif value == 'villain':
            return 'Villain'
    return None

def clean_role(X):
    df = pd.DataFrame(X).copy()
    for col in role_feature:
        df[col] = df[col].apply(clean_role_feature)
    return df

numerical_features = ["power_level", "weight", "height", "age", "speed", "battle_iq", "ranking", "training_time"]
categorical_features = ["skin_type", "eye_color", "gender", "hair_color", "universe", "body_type" ,"job", "species", "abilities", "special_attack", "secret_code"]
role_feature = ["role"]
name_feature = ["name"]
target_feature = ["win_prob"]

target_transform = Pipeline(steps=[     # minimum change in medians and quartals +-=0.01
    ("outlier removal", FunctionTransformer(win_probOutlierRemoval)), # clips the win prob to 0 and 1 some are over 1 maybe a different approach is better
    ("imputer", SimpleImputer(strategy="mean"))
    ]
)

numeric_transform = Pipeline(steps=[
    ("outlier removal", OutlierRemoval(multiplier=1.5)),
    ("scalar", RobustScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False))
    ]
)

categorical_transform = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="error", sparse_output=False))
    ]
)

role_transform = Pipeline(steps=[
    ("Spelling correction", FunctionTransformer(clean_role)),
    ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False))   # none is kept as category
    ]
)

preprocessor = ColumnTransformer( transformers= [
    ("num", numeric_transform, numerical_features),
    ("cat", categorical_transform, categorical_features),
    ("role", role_transform, role_feature),
    ]
)