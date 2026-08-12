"""Transformations fitted on the training fold only."""

from dataclasses import dataclass, field
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from scipy.stats import chi2_contingency
from imblearn.over_sampling import SMOTE, RandomOverSampler
from predicting_flight_arrival_delays.config import (
    SEED,
    MIN_CATEGORY_COUNT, 
    CORRELATION_THRESHOLD, 
    CATEGORICAL_ASSOCIATION_THRESHOLD,
    MIN_MUTUAL_INFO, MI_SAMPLE_SIZE,
)
from imblearn.under_sampling import RandomUnderSampler

OTHER = "OTHER"


@dataclass
class Transformer:
    """Fit-on-train.

    min_category_count: categories seen fewer times than this in training are folded
        into a single OTHER bucket.
    correlation_threshold: for any pair of numeric features correlated above this, the
        second is dropped.
    categorical_association_threshold: for any pair of categorical features correlated above this, the
        second is dropped.
    min_mutual_info: features whose mutual information with the target falls below this
        are dropped.
    mi_sample_size: mutual information is expensive on millions of rows; it is estimated
        on a random sample of this size.
    seed: seed to use for random operations.
    """

    min_category_count: int = MIN_CATEGORY_COUNT
    correlation_threshold: float = CORRELATION_THRESHOLD
    categorical_association_threshold: float = CATEGORICAL_ASSOCIATION_THRESHOLD
    min_mutual_info: float = MIN_MUTUAL_INFO
    mi_sample_size: int = MI_SAMPLE_SIZE

    categorical_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)

    category_keep: dict[str, set] = field(default_factory=dict, init=False)
    impute_values: dict[str, float] = field(default_factory=dict, init=False)
    scaler: StandardScaler | None = field(default=None, init=False)
    dropped_features: list[str] = field(default_factory=list, init=False)

    encoding: str = "onehot"
    seed: int = SEED


    # ------------------------------------------------------------------
    # Scaling, imputation, rare-category grouping
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "Transformer":
        """Learn the category buckets, imputation values and scaling statistics.

        Categories and imputation are learned first, then applied before fitting the
        scaler.

        Args:
            X: Training features.

        Returns:
            The fitted transformer, for chaining.
        """
        self.categorical_columns = self.categorical_columns or [
            c for c in X.columns if X[c].dtype == object
        ]
        self.numeric_columns = self.numeric_columns or list(
            X.select_dtypes(include=["number", "bool"]).columns
        )

        for col in self.categorical_columns:
            counts = X[col].value_counts()
            keep = set(counts[counts >= self.min_category_count].index)
            self.category_keep[col] = keep
            logger.info(
                f"{col}: keeping {len(keep)} of {len(counts)} categories "
                f"(threshold {self.min_category_count})"
            )

        for col in self.numeric_columns:
            self.impute_values[col] = float(X[col].median())

        prepared = self._apply_categories(X.copy())
        prepared = self._apply_imputation(prepared)
        self.scaler = StandardScaler().fit(prepared[self.numeric_columns])

        return self



    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformation to any set of rows.

        Categories unseen during fitting and missing values
        both fall into OTHER: the model has no reliable signal for either.

        Args:
            X: Features to transform.

        Returns:
            A transformed copy; the input is left untouched.
        """
        if self.scaler is None:
            raise RuntimeError("Transformer must be fitted before transform()")

        X = self._apply_categories(X.copy())
        X = self._apply_imputation(X)
        X[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])
        return X



    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit on X and transform it in one call.

        Args:
            X: Training features.

        Returns:
            The transformed training features.
        """
        return self.fit(X).transform(X)


    def _apply_categories(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace rare, unseen and missing category values with OTHER.

        Args:
            X: Features to modify in place.

        Returns:
            The same DataFrame, with categorical columns collapsed.
        """
        for col, keep in self.category_keep.items():
            if col in X.columns:
                X[col] = X[col].where(X[col].isin(keep), OTHER)
        return X

    def _apply_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing numeric values with the medians learned during fitting.

        Args:
            X: Features to modify in place.

        Returns:
            The same DataFrame, with numeric gaps filled.
        """
        for col, value in self.impute_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)
        return X

    # ------------------------------------------------------------------
    # Feature selection -- run BEFORE one-hot encoding, on the training fold
    # ------------------------------------------------------------------

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> "Transformer":
        """Decide which columns to drop, using only model-agnostic criteria.

        Filters applied:
        1. constant columns
        2. near-duplicate numeric features
        3. redundant categorical features, where one determines the other
        4. columns carrying essentially no information about the target, measured by
           mutual information.

        Args:
            X: Training features, before one-hot encoding.
            y: Training target.

        Returns:
            The transformer, with dropped_features populated, for chaining.
        """
        dropped: list[str] = []

        constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        if constant:
            logger.info(f"Constant, dropped: {constant}")
        dropped.extend(constant)

        numeric_cols = [
            c for c in self.numeric_columns if c in X.columns and c not in constant
        ]
        categorical_cols = [
            c for c in self.categorical_columns if c in X.columns and c not in constant
        ]
        categorical_cols = sorted(
            categorical_cols, key=lambda c: X[c].nunique(), reverse=True
        )

        correlated: list[str] = []
        if len(numeric_cols) > 1:
            corr = X[numeric_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            correlated = [
                c for c in upper.columns if (upper[c] > self.correlation_threshold).any()
            ]
            if correlated:
                logger.info(f"Correlated, dropped: {correlated}")
            dropped.extend(correlated)

        idx = (
            X.sample(self.mi_sample_size, random_state=self.seed).index
            if len(X) > self.mi_sample_size
            else X.index
        )

        associated: list[str] = []
        for a, b in combinations(categorical_cols, 2):
            if a in associated or b in associated: 
                continue
            v = cramers_v(X.loc[idx, a], X.loc[idx, b])
            if v > self.categorical_association_threshold:
                associated.append(b)
                logger.info(f"{b} is {v:.2f} associated with {a} -- dropping {b}")
        dropped.extend(associated)

        uninformative: list[str] = []

        remaining_numeric = [c for c in numeric_cols if c not in dropped]
        if remaining_numeric:
            mi = mutual_info_classif(
                X.loc[idx, remaining_numeric], y.loc[idx], random_state=self.seed
            )
            uninformative.extend(
                c for c, score in zip(remaining_numeric, mi) if score < self.min_mutual_info
            )

        remaining_categorical = [c for c in categorical_cols if c not in dropped]
        if remaining_categorical:
            codes = X.loc[idx, remaining_categorical].apply(
                lambda s: s.astype("category").cat.codes
            )
            mi = mutual_info_classif(codes, y.loc[idx], discrete_features=True, random_state=self.seed)
            uninformative.extend(
                c for c, score in zip(remaining_categorical, mi) if score < self.min_mutual_info
            )

        if uninformative:
            logger.info(f"Uninformative, dropped: {uninformative}")
        dropped.extend(uninformative)

        self.dropped_features = sorted(set(dropped))
        logger.info(
            f"Feature selection: dropping {len(self.dropped_features)} of {X.shape[1]} columns "
            f"({len(constant)} constant, {len(correlated)} correlated, "
            f"{len(associated)} associated, {len(uninformative)} uninformative)"
        )
        return self



    def apply_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop the columns chosen by select_features.

        Args:
            X: Any set of features carrying the same columns as the training data.

        Returns:
            The same rows without the dropped columns.
        """
        return X.drop(columns=[c for c in self.dropped_features if c in X.columns])


    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise the fitted state next to the model it belongs to.

        Args:
            path: Destination file; parent directories are created as needed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.success(f"Saved transformer to {path}")

    @staticmethod
    def load(path: Path) -> "Transformer":
        """Restore a transformer saved by save().

        Args:
            path: File written by save().

        Returns:
            The fitted transformer, ready to transform new rows.
        """
        return joblib.load(path)


def encode_categoricals(
    X: pd.DataFrame,
    categorical_columns: list[str],
    encoding: str = "onehot",
) -> pd.DataFrame:
    """Encode categorical columns.

    Two schemes:

    -onehot: one binary column per category.
    -native: pandas category dtype.

    Applied after OTHER-grouping.

    Args:
        X: Features with categorical columns still as labels.
        categorical_columns: Which columns to encode; absent ones are ignored.
        encoding: Either "onehot" or "native".

    Returns:
        A copy with the categorical columns encoded.

    Raises:
        ValueError: If the encoding is not one of the two supported schemes.
    """
    present = [c for c in categorical_columns if c in X.columns]

    if encoding == "native":
        X = X.copy()
        for c in present:
            X[c] = X[c].astype("category")
        return X

    if encoding == "onehot":
        encoded = pd.get_dummies(X, columns=present, drop_first=False)
        bool_cols = encoded.select_dtypes(include="bool").columns
        encoded[bool_cols] = encoded[bool_cols].astype("int8")
        return encoded
    
    raise ValueError(f"Unknown encoding '{encoding}'. Use 'onehot' or 'native'.")


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Measure the association between two categorical variables, on a 0-1 scale.

    Args:
        x: First categorical column.
        y: Second categorical column.

    Returns:
        0 when the two are independent, 1 when one fully determines the other.
    """
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return 0.0

    chi2 = chi2_contingency(table)[0]
    n = table.values.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def align_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    encoding: str = "onehot",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make test match train in shape, so the model sees the input it expects.

    Args:
        train: Reference features.
        test: Features to align.
        encoding: Which scheme was used to encode; determines what needs aligning.

    Returns:
        The train and test features, in that order.
    """
    if encoding == "native":
        test = test.copy()
        for c in train.columns:
            if isinstance(train[c].dtype, pd.CategoricalDtype):
                test[c] = test[c].astype(
                    pd.CategoricalDtype(categories=train[c].cat.categories)
                )
        return train, test[train.columns]

    missing = [c for c in train.columns if c not in test.columns]
    for c in missing:
        test[c] = 0
    if missing:
        logger.info(f"Added {len(missing)} missing one-hot columns to test set")

    return train, test[train.columns]


def resample_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    method: str,
    encoding: str,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Rebalance the training fold only.

    Must be called after encode_categoricals()/align_columns().

    Args:
        X: Training features, already encoded and column-aligned.
        y: Training target, aligned with X.
        method: One of "none", "undersample", "oversample", "smote".
        encoding: The encoding used to produce X ("onehot" or "native"); only used
            to validate compatibility with "smote".
        random_state: Seed for reproducible resampling.

    Returns:
        The resampled (X, y). Unchanged if method == "none".

    Raises:
        ValueError: If method is unknown, or method="smote" with encoding="native".
    """
    if method == "none":
        logger.warning(f"No resampling applied (method='none')")        
        return X, y

    if method == "smote" and encoding != "onehot":
        raise ValueError(
            "SMOTE requires numeric features: use encoding='onehot'. "
            "For 'native' (category dtype) encoding, use 'undersample' or "
            "'oversample' instead."
        )

    samplers = {
        "undersample": RandomUnderSampler(random_state=random_state),
        "oversample": RandomOverSampler(random_state=random_state),
        "smote": SMOTE(random_state=random_state),
    }
    if method not in samplers:
        raise ValueError(
            f"Unknown resampling method '{method}'. "
            f"Use one of: 'none', {', '.join(repr(m) for m in samplers)}."
        )

    sampler = samplers[method]
    before = y.value_counts().to_dict()
    X_res, y_res = sampler.fit_resample(X, y)
    after = y_res.value_counts().to_dict()
    logger.info(f"Resampling ({method}): {before} -> {after}")

    return X_res, y_res