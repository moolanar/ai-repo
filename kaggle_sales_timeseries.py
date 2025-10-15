https://www.kaggle.com/code/nilanjansaha123/time-series-full-0-5-rmsle
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/kaggle/input/store-sales-time-series-forecasting/oil.csv
/kaggle/input/store-sales-time-series-forecasting/sample_submission.csv
/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv
/kaggle/input/store-sales-time-series-forecasting/stores.csv
/kaggle/input/store-sales-time-series-forecasting/train.csv
/kaggle/input/store-sales-time-series-forecasting/test.csv
/kaggle/input/store-sales-time-series-forecasting/transactions.csv
# Store Sales Time Series Forecasting - Complete Implementation
# Based on Kaggle "Store Sales - Time Series Forecasting" Competition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

# Time Series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import lightgbm as lgb
import xgboost as xgb

#Deep Learning (Optional - uncomment if using)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ============================================================================
# 1. DATA LOADING AND SIMULATION
# ============================================================================

def load_store_sales_data():
    """
    Load the actual Kaggle Store Sales dataset
    """
    print("Loading datasets...")
    
    # Define data path
    data_path = '/kaggle/input/store-sales-time-series-forecasting'
    
    # Load all datasets
    train_df = pd.read_csv(f'{data_path}/train.csv')
    test_df = pd.read_csv(f'{data_path}/test.csv')
    oil_df = pd.read_csv(f'{data_path}/oil.csv')
    transactions_df = pd.read_csv(f'{data_path}/transactions.csv')
    holidays_df = pd.read_csv(f'{data_path}/holidays_events.csv')
    
    print(f"✓ Train data: {train_df.shape}")
    print(f"✓ Test data: {test_df.shape}")
    print(f"✓ Oil data: {oil_df.shape}")
    print(f"✓ Transactions data: {transactions_df.shape}")
    print(f"✓ Holidays data: {holidays_df.shape}")
    
    # Convert date columns
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    oil_df['date'] = pd.to_datetime(oil_df['date'])
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])
    
    # Merge additional features into train data
    print("Merging additional datasets...")
    
    # Merge oil prices
    train_df = train_df.merge(oil_df, on='date', how='left')
    
    # Fill missing oil prices with forward fill and backward fill
    train_df['dcoilwtico'] = train_df['dcoilwtico'].fillna(method='ffill').fillna(method='bfill')
    train_df.rename(columns={'dcoilwtico': 'oil_price'}, inplace=True)
    
    # Merge transactions
    train_df = train_df.merge(transactions_df, on=['date', 'store_nbr'], how='left')
    
    # Fill missing transactions with median per store
    train_df['transactions'] = train_df.groupby('store_nbr')['transactions'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Process holidays
    # Create holiday indicators
    holidays_df['is_holiday'] = 1
    national_holidays = holidays_df[holidays_df['locale'] == 'National'][['date', 'is_holiday']]
    regional_holidays = holidays_df[holidays_df['locale'] == 'Regional'][['date', 'locale_name', 'is_holiday']]
    local_holidays = holidays_df[holidays_df['locale'] == 'Local'][['date', 'locale_name', 'is_holiday']]
    
    # Merge national holidays
    train_df = train_df.merge(national_holidays, on='date', how='left')
    train_df['is_holiday'] = train_df['is_holiday'].fillna(0)
    
    # Add transferred flag for holidays
    transferred_holidays = holidays_df[holidays_df['transferred'] == True][['date']].copy()
    transferred_holidays['is_transferred'] = 1
    train_df = train_df.merge(transferred_holidays, on='date', how='left')
    train_df['is_transferred'] = train_df['is_transferred'].fillna(0)
    
    # Add holiday type information
    holiday_types = holidays_df.groupby('date')['type'].first().reset_index()
    train_df = train_df.merge(holiday_types, on='date', how='left')
    train_df['holiday_type'] = train_df['type'].fillna('None')
    train_df.drop('type', axis=1, inplace=True)
    
    print("✓ All datasets merged successfully")
    
    # Basic data info
    print(f"\nFinal dataset shape: {train_df.shape}")
    print(f"Date range: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Number of stores: {train_df['store_nbr'].nunique()}")
    print(f"Number of product families: {train_df['family'].nunique()}")
    print(f"Total sales records: {len(train_df):,}")
    
    # Display sample data
    print("\nSample data:")
    print(train_df.head())
    
    return train_df, test_df, oil_df, transactions_df, holidays_df

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def create_time_features(df):
    """Create time-based features from date column"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Basic time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['week'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    
    # Weekend/weekday
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    return df

def create_lag_features(df, target_col='sales', lags=[1, 2, 3, 7, 14, 30]):
    """Create lagged features for time series"""
    df = df.copy()
    df = df.sort_values(['store_nbr', 'family', 'date'])
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby(['store_nbr', 'family'])[target_col].shift(lag)
    
    return df

def create_rolling_features(df, target_col='sales', windows=[7, 14, 30]):
    """Create rolling statistics features"""
    df = df.copy()
    df = df.sort_values(['store_nbr', 'family', 'date'])
    
    for window in windows:
        # Rolling mean
        df[f'{target_col}_rolling_mean_{window}'] = df.groupby(['store_nbr', 'family'])[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling std
        df[f'{target_col}_rolling_std_{window}'] = df.groupby(['store_nbr', 'family'])[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        
        # Rolling max/min
        df[f'{target_col}_rolling_max_{window}'] = df.groupby(['store_nbr', 'family'])[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).max()
        )
        df[f'{target_col}_rolling_min_{window}'] = df.groupby(['store_nbr', 'family'])[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).min()
        )
    
    return df

def create_target_encoding(df, categorical_cols, target_col='sales'):
    """Create target encoding for categorical variables"""
    df = df.copy()
    
    for col in categorical_cols:
        if col in df.columns:  # Check if column exists
            # Mean encoding with smoothing
            global_mean = df[target_col].mean()
            encoding = df.groupby(col)[target_col].agg(['mean', 'count']).reset_index()
            
            # Smoothing factor
            alpha = 10
            encoding['smoothed_mean'] = (encoding['mean'] * encoding['count'] + global_mean * alpha) / (encoding['count'] + alpha)
            
            # Map back to dataframe
            encoding_dict = dict(zip(encoding[col], encoding['smoothed_mean']))
            df[f'{col}_target_enc'] = df[col].map(encoding_dict)
    
    return df

# ============================================================================
# 3. MULTIVARIATE TIME SERIES MODELS
# ============================================================================

class MultivariateTimeSeriesForecaster:
    """
    Comprehensive multivariate time series forecaster for store sales
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        
    def preprocess_data(self, df):
        """Complete data preprocessing pipeline"""
        print("Starting data preprocessing...")
        
        # Create time features
        df = create_time_features(df)
        print("✓ Time features created")
        
        # Create lag features
        df = create_lag_features(df)
        print("✓ Lag features created")
        
        # Create rolling features
        df = create_rolling_features(df)
        print("✓ Rolling features created")
        
        # Target encoding for categorical variables
        categorical_cols = ['family', 'holiday_type']
        df = create_target_encoding(df, categorical_cols)
        print("✓ Target encoding completed")
        
        # Label encoding for remaining categorical variables
        for col in categorical_cols:
            if col in df.columns:  # Check if column exists
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        print("✓ Label encoding completed")
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrix for modeling"""
        feature_cols = [
            'store_nbr', 'onpromotion', 'oil_price', 'is_holiday', 'is_transferred',
            'year', 'month', 'day', 'dayofweek', 'dayofyear', 'week', 'quarter',
            'month_sin', 'month_cos', 'day_sin', 'day_cos', 'dayofyear_sin', 'dayofyear_cos',
            'is_weekend', 'is_month_start', 'is_month_end'
        ]
        
        # Add transactions if available
        if 'transactions' in df.columns:
            feature_cols.append('transactions')
        
        # Add encoded categorical features
        categorical_encoded = [col for col in df.columns if col.endswith('_encoded')]
        feature_cols.extend(categorical_encoded)
        
        # Add target encoded features
        target_encoded = [col for col in df.columns if col.endswith('_target_enc')]
        feature_cols.extend(target_encoded)
        
        # Add lag features
        lag_cols = [col for col in df.columns if 'lag_' in col]
        feature_cols.extend(lag_cols)
        
        # Add rolling features
        rolling_cols = [col for col in df.columns if 'rolling_' in col]
        feature_cols.extend(rolling_cols)
        
        # Filter existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = feature_cols
        
        return df[feature_cols]
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train multiple models for ensemble"""
        print("Training models...")
        
        # 1. Linear Regression
        self.models['linear'] = LinearRegression()
        self.models['linear'].fit(X_train, y_train)
        print("✓ Linear Regression trained")
        
        # 2. Random Forest
        self.models['rf'] = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X_train, y_train)
        print("✓ Random Forest trained")
        
        # 3. Gradient Boosting
        self.models['gbm'] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.models['gbm'].fit(X_train, y_train)
        print("✓ Gradient Boosting trained")
        
        # 4. LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            self.models['lgb'] = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
            )
        else:
            self.models['lgb'] = lgb.train(
                params,
                train_data,
                num_boost_round=500
            )
        print("✓ LightGBM trained")
        
        # 5. XGBoost
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        
        if X_val is not None and y_val is not None:
            self.models['xgb'].fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.models['xgb'].fit(X_train, y_train)
        print("✓ XGBoost trained")
    
    def predict_ensemble(self, X_test):
        """Make ensemble predictions"""
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'lgb':
                pred = model.predict(X_test, num_iteration=model.best_iteration)
            else:
                pred = model.predict(X_test)
            predictions[name] = pred
        
        # Simple average ensemble
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        return ensemble_pred, predictions
    
    def evaluate_model(self, y_true, y_pred, model_name="Model"):
        """Evaluate model performance"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n{model_name} Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

# ============================================================================
# 4. DEEP LEARNING APPROACH (LSTM)
# ============================================================================

def create_lstm_sequences(data, sequence_length=30, target_col='sales'):
    """Create sequences for LSTM training"""
    sequences = []
    targets = []
    
    # Group by store and family
    for (store, family), group in data.groupby(['store_nbr', 'family']):
        group = group.sort_values('date')
        values = group[target_col].values
        
        for i in range(len(values) - sequence_length):
            sequences.append(values[i:i+sequence_length])
            targets.append(values[i+sequence_length])
    
    return np.array(sequences), np.array(targets)

def build_lstm_model(sequence_length, n_features=1):
    """Build LSTM model for multivariate time series"""
    # Uncomment if using TensorFlow/Keras
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model
    
    print("LSTM model building requires TensorFlow/Keras - uncomment code above")
    return None

# ============================================================================
# 5. MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    print("=== Store Sales Time Series Forecasting ===\n")
    
    # 1. Load real data
    print("1. Loading real Kaggle data...")
    train_df, test_df, oil_df, transactions_df, holidays_df = load_store_sales_data()
    df = train_df.copy()  # Use the loaded training data
    
    # 2. Initialize forecaster
    forecaster = MultivariateTimeSeriesForecaster()
    
    # 3. Preprocess data
    print("\n2. Preprocessing data...")
    df_processed = forecaster.preprocess_data(df)
    
    # Remove rows with NaN values (due to lagging)
    df_processed = df_processed.dropna()
    print(f"Dataset shape after preprocessing: {df_processed.shape}")
    
    # 4. Prepare features and target
    print("\n3. Preparing features...")
    X = forecaster.prepare_features(df_processed)
    y = df_processed['sales'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(forecaster.feature_columns)}")
    
    # 5. Time-based train/validation split
    print("\n4. Splitting data...")
    split_date = pd.to_datetime('2017-01-01')
    train_mask = df_processed['date'] < split_date
    val_mask = df_processed['date'] >= split_date
    
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # 6. Train models
    print("\n5. Training models...")
    forecaster.train_models(X_train, y_train, X_val, y_val)
    
    # 7. Make predictions
    print("\n6. Making predictions...")
    ensemble_pred, individual_preds = forecaster.predict_ensemble(X_val)
    
    # 8. Evaluate models
    print("\n7. Model Evaluation:")
    print("="*50)
    
    # Evaluate individual models
    for name, pred in individual_preds.items():
        forecaster.evaluate_model(y_val, pred, name.upper())
    
    # Evaluate ensemble
    forecaster.evaluate_model(y_val, ensemble_pred, "ENSEMBLE")
    
    # 9. Feature importance (using Random Forest)
    print("\n8. Top 15 Feature Importances (Random Forest):")
    print("="*50)
    feature_importance = pd.DataFrame({
        'feature': forecaster.feature_columns,
        'importance': forecaster.models['rf'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")
    
    # 10. Visualizations
    print("\n9. Creating visualizations...")
    
    # Plot predictions vs actual
    plt.figure(figsize=(15, 10))
    
    # Sample a subset for visualization
    sample_idx = np.random.choice(len(y_val), min(1000, len(y_val)), replace=False)
    
    plt.subplot(2, 2, 1)
    plt.scatter(y_val[sample_idx], ensemble_pred[sample_idx], alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Sales (Ensemble)')
    
    # Sample a specific store-family for visualization
    sample_data = df_processed[
        (df_processed['store_nbr'] == 1) & 
        (df_processed['family'] == 'GROCERY I') &
        (df_processed['date'] >= split_date - pd.DateOffset(days=60))
    ].copy()
    
    if len(sample_data) > 0:
        plt.subplot(2, 2, 2)
        plt.plot(sample_data['date'], sample_data['sales'], label='Actual', linewidth=2)
        
        # Get predictions for this subset
        sample_X = forecaster.prepare_features(sample_data)
        if len(sample_X) > 0:
            sample_pred, _ = forecaster.predict_ensemble(sample_X)
            plt.plot(sample_data['date'], sample_pred, label='Predicted', linewidth=2)
        
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title('Time Series: Store 1 - GROCERY I')
        plt.legend()
        plt.xticks(rotation=45)
    
    # Residuals plot
    plt.subplot(2, 2, 3)
    residuals = y_val - ensemble_pred
    plt.scatter(ensemble_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    
    # Feature importance plot
    plt.subplot(2, 2, 4)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Analysis Complete ===")
    
    return forecaster, df_processed, X_val, y_val, ensemble_pred

# ============================================================================
# 6. ADDITIONAL ANALYSIS FUNCTIONS
# ============================================================================

def analyze_seasonality(df):
    """Analyze seasonal patterns in the data"""
    # Aggregate by date
    daily_sales = df.groupby('date')['sales'].sum().reset_index()
    daily_sales.set_index('date', inplace=True)
    
    # Seasonal decomposition
    decomposition = seasonal_decompose(daily_sales['sales'], model='additive', period=365)
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    plt.tight_layout()
    plt.show()

def cross_validation_time_series(forecaster, X, y, n_splits=5):
    """Perform time series cross validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = {'rmse': [], 'mae': [], 'r2': []}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        # Train only LightGBM for speed
        train_data = lgb.Dataset(X_train_cv, label=y_train_cv)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1
        }
        
        model = lgb.train(params, train_data, num_boost_round=100)
        pred = model.predict(X_val_cv)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_val_cv, pred))
        mae = mean_absolute_error(y_val_cv, pred)
        r2 = r2_score(y_val_cv, pred)
        
        cv_scores['rmse'].append(rmse)
        cv_scores['mae'].append(mae)
        cv_scores['r2'].append(r2)
    
    print(f"\nCross Validation Results:")
    print(f"RMSE: {np.mean(cv_scores['rmse']):.4f} (+/- {np.std(cv_scores['rmse']) * 2:.4f})")
    print(f"MAE: {np.mean(cv_scores['mae']):.4f} (+/- {np.std(cv_scores['mae']) * 2:.4f})")
    print(f"R²: {np.mean(cv_scores['r2']):.4f} (+/- {np.std(cv_scores['r2']) * 2:.4f})")

# ============================================================================
# 7. RUN THE COMPLETE PIPELINE
# ============================================================================

if __name__ == "__main__":
    # Execute main pipeline
    forecaster, df_processed, X_val, y_val, ensemble_pred = main()
    
    # Additional analyses
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSES")
    print("="*60)
    
    # Seasonality analysis
    print("\n10. Analyzing seasonality...")
    analyze_seasonality(df_processed)
    
    # Cross validation
    print("\n11. Time series cross validation...")
    X_full = forecaster.prepare_features(df_processed)
    y_full = df_processed['sales'].values
    cross_validation_time_series(forecaster, X_full,y_full)
2025-08-29 20:45:58.591670: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1756500358.929955      36 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1756500359.024324      36 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
=== Store Sales Time Series Forecasting ===

1. Loading real Kaggle data...
Loading datasets...
✓ Train data: (3000888, 6)
✓ Test data: (28512, 5)
✓ Oil data: (1218, 2)
✓ Transactions data: (83488, 3)
✓ Holidays data: (350, 6)
Merging additional datasets...
✓ All datasets merged successfully

Final dataset shape: (3008016, 11)
Date range: 2013-01-01 00:00:00 to 2017-08-15 00:00:00
Number of stores: 54
Number of product families: 33
Total sales records: 3,008,016

Sample data:
   id       date  store_nbr      family  sales  onpromotion  oil_price  \
0   0 2013-01-01          1  AUTOMOTIVE    0.0            0      93.14   
1   1 2013-01-01          1   BABY CARE    0.0            0      93.14   
2   2 2013-01-01          1      BEAUTY    0.0            0      93.14   
3   3 2013-01-01          1   BEVERAGES    0.0            0      93.14   
4   4 2013-01-01          1       BOOKS    0.0            0      93.14   

   transactions  is_holiday  is_transferred holiday_type  
0        1746.0         1.0             0.0      Holiday  
1        1746.0         1.0             0.0      Holiday  
2        1746.0         1.0             0.0      Holiday  
3        1746.0         1.0             0.0      Holiday  
4        1746.0         1.0             0.0      Holiday  

2. Preprocessing data...
Starting data preprocessing...
✓ Time features created
✓ Lag features created
✓ Rolling features created
✓ Target encoding completed
✓ Label encoding completed
Dataset shape after preprocessing: (2954556, 49)

3. Preparing features...
Feature matrix shape: (2954556, 44)
Number of features: 44

4. Splitting data...
Training set: 2550042 samples
Validation set: 404514 samples

5. Training models...
Training models...
✓ Linear Regression trained
✓ Random Forest trained
✓ Gradient Boosting trained
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[540]	valid_0's rmse: 225.781
✓ LightGBM trained
✓ XGBoost trained

6. Making predictions...

7. Model Evaluation:
==================================================

LINEAR Performance:
RMSE: 295.0731
MAE: 85.2980
R²: 0.9526

RF Performance:
RMSE: 257.1836
MAE: 66.4039
R²: 0.9640

GBM Performance:
RMSE: 230.4632
MAE: 60.5290
R²: 0.9711

LGB Performance:
RMSE: 225.7808
MAE: 59.3368
R²: 0.9722

XGB Performance:
RMSE: 228.4417
MAE: 57.5884
R²: 0.9716

ENSEMBLE Performance:
RMSE: 225.1547
MAE: 60.2189
R²: 0.9724

8. Top 15 Feature Importances (Random Forest):
==================================================
 1. sales_rolling_mean_7      0.8775
 2. sales_lag_7               0.0383
 3. sales_lag_14              0.0220
 4. transactions              0.0176
 5. sales_lag_1               0.0069
 6. sales_rolling_max_7       0.0065
 7. sales_rolling_std_7       0.0057
 8. dayofweek                 0.0030
 9. sales_rolling_min_7       0.0020
10. dayofyear                 0.0019
11. day_sin                   0.0019
12. sales_rolling_std_14      0.0018
13. day                       0.0018
14. is_weekend                0.0015
15. sales_rolling_mean_14     0.0014

9. Creating visualizations...

=== Analysis Complete ===

============================================================
ADDITIONAL ANALYSES
============================================================

10. Analyzing seasonality...

11. Time series cross validation...
Fold 1/5
Fold 2/5
Fold 3/5
Fold 4/5
Fold 5/5

Cross Validation Results:
RMSE: 231.2321 (+/- 145.1323)
MAE: 55.8086 (+/- 29.6201)
R²: 0.9398 (+/- 0.0463)
 
