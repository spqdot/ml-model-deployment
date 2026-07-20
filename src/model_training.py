from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def train_linear_regression(X_train, y_train) -> LinearRegression:
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train,
                        n_estimators: int = 100,
                        max_depth: int = 5,
                        random_state: int = 42) -> RandomForestRegressor:
    """Train a Random Forest Regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train,
                  n_estimators: int = 100,
                  max_depth: int = 5,
                  learning_rate: float = 0.1,
                  random_state: int = 42) -> XGBRegressor:
    """Train an XGBoost Regressor."""
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model
