import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data from the given filepath."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unused columns (Unnamed: 0, date, index) if present."""
    cols_to_drop = [col for col in ['Unnamed: 0', 'index', 'date'] if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df


def split_data(df: pd.DataFrame, target: str = 'sales',
               test_size: float = 0.30, random_state: int = 42):
    """
    Split data into train (70%), validation (15%), test (15%).
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_features(X_train: pd.DataFrame, X_val: pd.DataFrame,
                    X_test: pd.DataFrame, cat_col: str = 'state_holiday'):
    """
    One-hot encode the categorical column.
    Fits the encoder on training data only.
    Returns: X_train_final, X_val_final, X_test_final, encoder
    """
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore')

    X_train_cat = encoder.fit_transform(X_train[[cat_col]])
    X_val_cat = encoder.transform(X_val[[cat_col]])
    X_test_cat = encoder.transform(X_test[[cat_col]])

    encoded_cols = encoder.get_feature_names_out([cat_col])

    X_train_cat = pd.DataFrame(X_train_cat.toarray(), columns=encoded_cols, index=X_train.index)
    X_val_cat = pd.DataFrame(X_val_cat.toarray(), columns=encoded_cols, index=X_val.index)
    X_test_cat = pd.DataFrame(X_test_cat.toarray(), columns=encoded_cols, index=X_test.index)

    X_train_num = X_train.drop(cat_col, axis=1)
    X_val_num = X_val.drop(cat_col, axis=1)
    X_test_num = X_test.drop(cat_col, axis=1)

    X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
    X_val_final = pd.concat([X_val_num, X_val_cat], axis=1)
    X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)

    return X_train_final, X_val_final, X_test_final, encoder


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """
    Apply StandardScaler. Fits on training data only.
    Returns: X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def preprocess_new_data(new_data: pd.DataFrame, encoder: OneHotEncoder,
                        scaler: StandardScaler = None,
                        cat_col: str = 'state_holiday') -> pd.DataFrame:
    """
    Preprocess unseen data using a fitted encoder (and optional scaler).
    """
    new_cat = encoder.transform(new_data[[cat_col]])
    encoded_cols = encoder.get_feature_names_out([cat_col])
    new_cat = pd.DataFrame(new_cat.toarray(), columns=encoded_cols, index=new_data.index)

    new_num = new_data.drop(cat_col, axis=1)
    new_final = pd.concat([new_num, new_cat], axis=1)

    if scaler is not None:
        new_final = scaler.transform(new_final)

    return new_final
