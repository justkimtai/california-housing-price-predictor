from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_processing_pipeline(df):
    # Assume 'ocean_proximity' is categorical if using other datasets
    numeric_features = df.select_dtypes(include=['float64', 'int']).columns
    numeric_features = numeric_features.drop('MedHouseVal')
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    # If categorical features exist:
    # categorical_features = ['ocean_proximity']
    # categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), 
    # ('cat', categorical_transformer, categorical_features),
    ])
    return preprocessor