# House Price Prediction Model
# Complete pipeline for predicting house prices using machine learning

# Step 1 -  Importing Libraries & Datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.ParserError: If the file is not a valid CSV.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f"Error reading {file_path}: {e}")
    
    return df

# Step 2 - Preprocess Data incl. EDA, Feature Engg., Clean Up, normalizing (with oneHotEncoder for cat. features & standardScaler for discrete features)
def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the housing dataset:
    - EDA (info, describe, plots)
    - Handle missing values
    - Feature engineering
    - Outlier removal
    - OneHotEncoding for categorical features
    - StandardScaling for numerical features
    
    Args:
        df (pd.DataFrame): Raw input dataset.
    
    Returns:
        X (pd.DataFrame): Feature matrix (preprocessed but not yet transformed).
        y (pd.Series): Target variable.
        preprocessor (ColumnTransformer): Fitted preprocessing pipeline.
    """

    # -------------------------
    # Dataset Overview
    # -------------------------
    print("\nDataset Info:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # -------------------------
    # Exploratory Data Analysis (EDA)
    # -------------------------
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)

    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Distribution of target variable
    axes[0, 0].hist(df['median_house_value'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribution of House Prices')

    # Correlation heatmap
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
    axes[0, 1].set_title('Correlation Matrix')

    # Ocean proximity distribution
    df['ocean_proximity'].value_counts().plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Distribution by Ocean Proximity')

    # Price vs Income scatter plot
    axes[1, 1].scatter(df['median_income'], df['median_house_value'], alpha=0.6)
    axes[1, 1].set_title('House Price vs Median Income')

    plt.tight_layout()
    plt.show()

    # -------------------------
    # Feature Engineering
    # -------------------------
    print("\nFeature Engineering:")
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    print("Created new features: rooms_per_household, bedrooms_per_room, population_per_household")

    # -------------------------
    # Data Cleaning
    # -------------------------
    print("\n" + "="*50)
    print("DATA CLEANING")
    print("="*50)

    # Handle missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"Filled missing values in {col} with median")

    # Outlier removal (IQR on target)
    Q1, Q3 = df['median_house_value'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR

    before = len(df)
    df = df[(df['median_house_value'] >= lower) & (df['median_house_value'] <= upper)]
    print(f"Removed {before - len(df)} outliers")

    # -------------------------
    # Encoding + Scaling
    # -------------------------
    # Separate features and target
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    categorical_features = ['ocean_proximity']
    numerical_features = [col for col in X.columns if col not in categorical_features]
    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )

    print("✅ Preprocessing pipeline ready (StandardScaler + OneHotEncoder)")

    return X, y, preprocessor

# Step 3 - Split Dataset
def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Splits dataset into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Step 4 - Train Models
def train_models(X_train, y_train, preprocessor):
    """
    Trains multiple regression models and returns trained pipelines.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_train)
        results[name] = pipeline
    return results

# Step 5 - Evaluate Models
def evaluate_models(results, X_test, y_test, categorical_features, numerical_features):
    """
    Evaluates trained models, compares performance, and returns the best model.
    """
    metrics = {}

    for name, pipeline in results.items():
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Pipeline': pipeline
        }

    # Convert metrics to dataframe for comparison
    comparison_df = pd.DataFrame({
        'Model': list(metrics.keys()),
        'RMSE': [metrics[m]['RMSE'] for m in metrics.keys()],
        'MAE': [metrics[m]['MAE'] for m in metrics.keys()],
        'R²': [metrics[m]['R²'] for m in metrics.keys()]
    }).round(2)

    print("\nModel Comparison:\n", comparison_df)

    # Select best model by RMSE
    best_model_name = min(metrics.keys(), key=lambda x: metrics[x]['RMSE'])
    best_pipeline = metrics[best_model_name]['Pipeline']

    print(f"\nBest Model: {best_model_name}")
    print(f"Best RMSE: ${metrics[best_model_name]['RMSE']:,.2f}")

    # Plot actual vs predicted for best model
    y_pred_best = best_pipeline.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_best, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'{best_model_name}: Actual vs Predicted')
    plt.show()

    return best_pipeline, best_model_name, metrics

# Step 6 - Inference for Users
def predict_house_price(best_pipeline, input_data):
    """
    Makes prediction on new user-provided data.
    input_data: dict containing feature values
    """
    new_data = pd.DataFrame([input_data])

    # Add engineered features
    new_data['rooms_per_household'] = new_data['total_rooms'] / new_data['households']
    new_data['bedrooms_per_room'] = new_data['total_bedrooms'] / new_data['total_rooms']
    new_data['population_per_household'] = new_data['population'] / new_data['households']

    return best_pipeline.predict(new_data)[0]


def main():
    # Dataset path
    dataset_path = "C:/Users/warre/Documents/Python/simple-neural-network/1-pricePredictor/housing.csv"
    
    # Load dataset
    df = load_data(dataset_path)
    # Show basic info
    print("✅ Dataset loaded successfully")
    print(f"Shape: {df.shape}")
    print(df.head())
    # Pre-process Dataset
    X, y, preprocessor = preprocess_data(df)
    # 
    categorical_features = ['ocean_proximity']
    numerical_features = [
        'longitude', 'latitude', 'housing_median_age', 
        'total_rooms', 'total_bedrooms', 'population', 
        'households', 'median_income'
    ]
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    # Split Dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    # Train Models
    trained_models = train_models(X_train, y_train, preprocessor)
    # Evaluate Models
    best_pipeline, best_model_name, metrics = evaluate_models(
        trained_models, X_test, y_test, categorical_features, numerical_features
    )
    example_input = {
        'longitude': -122.23, 'latitude': 37.88, 'housing_median_age': 25.0,
        'total_rooms': 5000.0, 'total_bedrooms': 800.0, 'population': 2000.0,
        'households': 700.0, 'median_income': 6.5, 'ocean_proximity': 'NEAR BAY'
    }
    predicted_price = predict_house_price(best_pipeline, example_input)
    print(f"\nExample Prediction: ${predicted_price:,.2f}")



if __name__ == "__main__":
    main()

