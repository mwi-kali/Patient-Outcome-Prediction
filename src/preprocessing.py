from .config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder



def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('onehot', OneHotEncoder(drop='first'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, NUMERICAL_FEATURES),
        ('cat', cat_pipe, CATEGORICAL_FEATURES)
    ], remainder='drop')
    return preprocessor