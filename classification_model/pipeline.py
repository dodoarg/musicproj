from feature_engine.encoding import DecisionTreeEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification_model.config.core import config

popularity_pipe = Pipeline(
    [
        (
            "dt encoder",
            DecisionTreeEncoder(
                variables=config.model_config.categorical_features,
                random_state=config.model_config.random_state,
            ),
        ),
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier()),
    ]
)
