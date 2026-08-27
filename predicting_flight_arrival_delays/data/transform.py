"""Transformations fitted on the training fold only."""

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import joblib
from loguru import logger
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from predicting_flight_arrival_delays.config import (
    CATEGORICAL_ASSOCIATION_THRESHOLD,
    CORRELATION_THRESHOLD,
    DATE_COLUMN,
    MAX_ONEHOT_CATEGORIES,
    MI_SAMPLE_SIZE,
    MIN_CATEGORY_COUNT,
    MIN_MUTUAL_INFO,
    SEED,
    SERVICE_COLUMNS2,
)

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
    id_columns: Numeric columns that are actually categorical.   
    max_onehot_categories: a categorical holding more values than this is not worth a
        column per category. It is given a historical delay rate like the columns
        below, and under onehot encoding the original is then dropped.
    delay_rate_columns: Columns used to derive historical delay-rate features. 
        Original columns are retained regardless of their width or encoding assigned to the model. Optional.
    delay_rate_shrinkage: shrinkage strength (the "k" in
        (sum + k*global_rate) / (count + k)) pulling rare categories' rates toward
        the global rate.
    """

    min_category_count: int = MIN_CATEGORY_COUNT
    correlation_threshold: float = CORRELATION_THRESHOLD
    categorical_association_threshold: float = CATEGORICAL_ASSOCIATION_THRESHOLD
    min_mutual_info: float = MIN_MUTUAL_INFO
    mi_sample_size: int = MI_SAMPLE_SIZE

    categorical_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    id_columns: list[str] = field(
        default_factory=lambda: ["OriginAirportID", "DestAirportID"]
    )

    category_keep: dict[str, set] = field(default_factory=dict, init=False)
    impute_values: dict[str, float] = field(default_factory=dict, init=False)
    scaler: StandardScaler | None = field(default=None, init=False)
    dropped_features: list[str] = field(default_factory=list, init=False)

    delay_rate_columns: list[str] = field(default_factory=list)
    delay_rate_shrinkage: float = 50.0
    delay_rate_stats_: dict[str, dict[str, pd.Series]] = field(default_factory=dict, init=False)
    global_delay_rate_: float | None = field(default=None, init=False)

    max_onehot_categories: int = MAX_ONEHOT_CATEGORIES

    wide_columns_: list[str] = field(default_factory=list, init=False)

    encoding: str = "onehot"
    seed: int = SEED

    def __post_init__(self) -> None:
        self._categorical_columns_given = bool(self.categorical_columns)
        self._numeric_columns_given = bool(self.numeric_columns)

    def _ids_as_labels(self, X: pd.DataFrame) -> pd.DataFrame:
        """Read the id columns as labels rather than as magnitudes.

        Args:
            X: Features to modify in place.

        Returns:
            The same DataFrame, with the id columns holding strings.
        """
        for col in self.id_columns:
            if col in X.columns and not pd.api.types.is_object_dtype(X[col]):
                X[col] = X[col].astype("Int64").astype("string").astype(object)
        return X

    # ------------------------------------------------------------------
    # Historical delay rates
    # ------------------------------------------------------------------

    def rate_columns(self) -> list[str]:
        """Every column a historical delay rate is computed for.

        Returns:
            The column names, named ones first.
        """
        return list(self.delay_rate_columns) + [
            c for c in self.wide_columns_ if c not in self.delay_rate_columns
        ]

    def _find_wide_columns(self, X: pd.DataFrame) -> list[str]:
        """Pick out the categoricals too wide to spend one column per category on.

        Args:
            X: Training features, ids already read as labels.

        Returns:
            The object columns holding more than max_onehot_categories values, minus
            those already named in delay_rate_columns.
        """
        wide = [
            c
            for c in X.columns
            if pd.api.types.is_object_dtype(X[c])
            and c not in self.delay_rate_columns
            and X[c].nunique(dropna=False) > self.max_onehot_categories
        ]
        if wide:
            logger.info(
                f"Too wide to one-hot (over {self.max_onehot_categories} values): {wide}. "
                f"They are kept as a delay rate"
                + (", and as themselves under native encoding." if self.encoding == "native"
                   else " only, the original being dropped.")
            )
        return wide

    def _drop_wide_originals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove the wide originals, but only where they would cost a column each.

        Under native encoding a category costs nothing extra, so the column stays and
        the model gets both readings of it.

        Args:
            df: Features, rates already attached.

        Returns:
            The same frame, minus the wide originals under onehot encoding.
        """
        if self.encoding != "onehot":
            return df
        return df.drop(columns=[c for c in self.wide_columns_ if c in df.columns])

    def _fit_delay_rates_internal(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> pd.DataFrame:
        """Learn per-category historical delay rates, with shrinkage for rare ones.

        Args:
            X: Training features, including delay_rate_columns as raw values.
            y: Training target, aligned with X.
            dates: Each row's flight date, aligned with X.

        Returns:
            A new DataFrame, same index as X, with one "<col>DelayRate" column
            per entry in self.delay_rate_columns.
        """
        self.global_delay_rate_ = float(y.mean())
        order = np.argsort(dates.to_numpy(), kind="stable")

        result = pd.DataFrame(index=X.index)
        self.delay_rate_stats_ = {}

        for col in self.rate_columns():
            if col not in X.columns:
                continue
            values_sorted = X[col].to_numpy()[order]
            y_sorted = y.to_numpy()[order].astype(float)
            keys = pd.Series(values_sorted)

            grouped_sum = pd.Series(y_sorted).groupby(keys, sort=False).cumsum().to_numpy()
            grouped_count = (
                pd.Series(np.ones(len(y_sorted))).groupby(keys, sort=False).cumsum().to_numpy()
            )

            prior_sum = grouped_sum - y_sorted
            prior_count = grouped_count - 1

            k = self.delay_rate_shrinkage
            shrunk_sorted = (prior_sum + k * self.global_delay_rate_) / (prior_count + k)

            shrunk = np.empty(len(shrunk_sorted))
            shrunk[order] = shrunk_sorted
            result[f"{col}DelayRate"] = shrunk

            final_sum = pd.Series(y_sorted).groupby(keys, sort=False).sum()
            final_count = pd.Series(np.ones(len(y_sorted))).groupby(keys, sort=False).sum()
            self.delay_rate_stats_[col] = {"sum": final_sum, "count": final_count}

            logger.info(
                f"{col}DelayRate: learned rates for {len(final_count)} categories "
                f"(shrinkage k={k}, global rate={self.global_delay_rate_:.3f})"
            )

        return result

    def _apply_delay_rates_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score any rows against the frozen training-period delay-rate stats.

        Args:
            X: Features to score, including delay_rate_columns as raw values.

        Returns:
            A new DataFrame, same index as X, with one "<col>DelayRate" column
            per entry in self.delay_rate_columns.
        """
        result = pd.DataFrame(index=X.index)
        k = self.delay_rate_shrinkage

        for col in self.rate_columns():
            if col not in X.columns:
                continue
            stats = self.delay_rate_stats_.get(
                col, {"sum": pd.Series(dtype=float), "count": pd.Series(dtype=float)}
            )
            sums = X[col].map(stats["sum"]).fillna(0.0)
            counts = X[col].map(stats["count"]).fillna(0.0)
            result[f"{col}DelayRate"] = (sums + k * self.global_delay_rate_) / (counts + k)

        return result

    # ------------------------------------------------------------------
    # Scaling, imputation, rare-category grouping
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Transformer":
        """Learn the category buckets, imputation values and scaling statistics.

        Args:
            X: Training features, including DATE_COLUMN.
            y: Training target, aligned with X.

        Returns:
            The fitted transformer, for chaining.
        """
        df = self._ids_as_labels(X.copy())
        self.category_keep = {}
        self.impute_values = {}
        self.wide_columns_ = self._find_wide_columns(df)

        if self.rate_columns():
            dates = df[DATE_COLUMN]
            rate_cols = self._fit_delay_rates_internal(df, y, dates)
            df = pd.concat([df, rate_cols], axis=1)

        df = self._drop_wide_originals(df)
        df = df.drop(columns=[c for c in SERVICE_COLUMNS2 if c in df.columns])

        if not self._categorical_columns_given:
            self.categorical_columns = [c for c in df.columns if df[c].dtype == object]
        if not self._numeric_columns_given:
            self.numeric_columns = list(
                df.select_dtypes(include=["number", "bool"]).columns
            )

        for col in self.categorical_columns:
            counts = df[col].value_counts()
            keep = set(counts[counts >= self.min_category_count].index)
            self.category_keep[col] = keep
            logger.info(
                f"{col}: keeping {len(keep)} of {len(counts)} categories "
                f"(threshold {self.min_category_count})"
            )

        for col in self.numeric_columns:
            self.impute_values[col] = float(df[col].median())

        prepared = self._apply_categories(df)
        prepared = self._apply_imputation(prepared)
        self.scaler = StandardScaler().fit(prepared[self.numeric_columns])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformation to any set of rows.

        Drops everything in config.SERVICE_COLUMNS2.

        Categories unseen during fitting and missing values
        both fall into OTHER: the model has no reliable signal for either.

        Args:
            X: Features to transform.

        Returns:
            A transformed copy; the input is left untouched.
        """
        if self.scaler is None:
            raise RuntimeError("Transformer must be fitted before transform()")
        df = self._ids_as_labels(X.copy())

        if self.rate_columns():
            rate_cols = self._apply_delay_rates_internal(df)
            df = pd.concat([df, rate_cols], axis=1)

        df = self._drop_wide_originals(df)
        df = df.drop(columns=[c for c in SERVICE_COLUMNS2 if c in df.columns])

        df = self._apply_categories(df)
        df = self._apply_imputation(df)
        df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])
        return df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit on X and transform it in one call.

        Args:
            X: Training features.
            y: Training target; see fit().

        Returns:
            The transformed training features. Never carries SERVICE_COLUMNS2
            (e.g. "FlightDate") - both fit() and transform() drop it internally.
        """
        return self.fit(X, y).transform(X)

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
    # Feature selection - run BEFORE one-hot encoding, on the training fold
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

        if len(X) > self.mi_sample_size:
            positions = np.random.default_rng(self.seed).choice(
                len(X), self.mi_sample_size, replace=False
            )
            X_samp = X.iloc[positions]
            y_samp = y.iloc[positions]
        else:
            X_samp, y_samp = X, y

        associated: list[str] = []
        for a, b in combinations(categorical_cols, 2):
            if a in associated or b in associated:
                continue
            v = cramers_v(X_samp[a], X_samp[b])
            if v > self.categorical_association_threshold:
                associated.append(b)
                logger.info(f"{b} is {v:.2f} associated with {a} - dropping {b}")
        dropped.extend(associated)

        uninformative: list[str] = []

        remaining_numeric = [c for c in numeric_cols if c not in dropped]
        if remaining_numeric:
            mi = mutual_info_classif(
                X_samp[remaining_numeric], y_samp, random_state=self.seed
            )
            uninformative.extend(
                c for c, score in zip(remaining_numeric, mi) if score < self.min_mutual_info
            )

        remaining_categorical = [c for c in categorical_cols if c not in dropped]
        if remaining_categorical:
            codes = X_samp[remaining_categorical].apply(
                lambda s: s.astype("category").cat.codes
            )
            mi = mutual_info_classif(codes, y_samp, discrete_features=True, random_state=self.seed)
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

    -onehot: one binary column per category, stored SPARSE (pandas SparseDtype) to
        keep the DataFrame itself lightweight. Numeric columns are downcast to
        float32. This is still a DataFrame (needed for align_columns, which relies
        on column names) - call to_sparse_matrix() afterwards, once alignment is
        done, to get a true scipy.sparse matrix for fitting/resampling.
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
        encoded = pd.get_dummies(X, columns=present, drop_first=False, sparse=True)
        dense_numeric = [
            c for c in encoded.columns if not isinstance(encoded[c].dtype, pd.SparseDtype)
        ]
        for c in dense_numeric:
            if pd.api.types.is_float_dtype(encoded[c]):
                encoded[c] = encoded[c].astype("float32")
        return encoded

    raise ValueError(f"Unknown encoding '{encoding}'. Use 'onehot' or 'native'.")


def _split_sparse_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separate the dense columns from the sparse (one-hot) ones, order preserved."""
    sparse_cols = [c for c in X.columns if isinstance(X[c].dtype, pd.SparseDtype)]
    dense_cols = [c for c in X.columns if c not in sparse_cols]
    return dense_cols, sparse_cols


def sparse_column_order(X: pd.DataFrame) -> list[str]:
    """The column order to_sparse_matrix() will produce for X.

    The matrix loses the names, so anything that needs to reproduce its layout
    later - a registered model's columns.json, above all - has to record this
    order rather than X's own.

    Args:
        X: A onehot-encoded DataFrame, as passed to to_sparse_matrix().

    Returns:
        Dense column names first, then the sparse ones.
    """
    dense_cols, sparse_cols = _split_sparse_columns(X)
    return dense_cols + sparse_cols


def to_sparse_matrix(X: pd.DataFrame) -> sp.csr_matrix:
    """Convert a onehot-encoded DataFrame (mix of dense numeric + sparse dummy
    columns) into a single scipy.sparse CSR matrix, ready for model fitting or
    resampling.

    Call this after align_columns() - alignment needs column names, which are
    lost once converted to a raw sparse matrix. Column order is preserved as
    dense columns first, then sparse (one-hot) columns; the same split is applied
    consistently to any DataFrame passed in, so calling this separately on
    already-aligned train/validation/test features yields matrices whose columns
    correspond to each other.

    Args:
        X: A DataFrame as returned by encode_categoricals(..., encoding="onehot"),
            optionally already passed through align_columns().

    Returns:
        A CSR sparse matrix, float32, with X.shape[0] rows and X.shape[1] columns.
    """
    dense_cols, sparse_cols = _split_sparse_columns(X)

    parts = []
    if dense_cols:
        parts.append(sp.csr_matrix(X[dense_cols].astype("float32").to_numpy()))
    if sparse_cols:
        parts.append(X[sparse_cols].sparse.to_coo().tocsr().astype("float32"))

    if not parts:
        raise ValueError("X has no columns to convert.")
    return sp.hstack(parts, format="csr") if len(parts) > 1 else parts[0]


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
    test = test.copy()
    for c in missing:
        test[c] = pd.Series(False, index=test.index).astype(train[c].dtype)
    if missing:
        logger.info(f"Added {len(missing)} missing one-hot columns to test set")

    return train, test[train.columns]


def align_to_training_columns(
    X: pd.DataFrame,
    training_columns: list[str] | tuple[str, ...],
    transformer: Transformer,
) -> pd.DataFrame:
    """Rebuild the exact matrix a fitted model expects, from encoded features.

    Used by every path that scores rows with an already-fitted model -
    inference and offline evaluation alike. Unlike align_columns(), which pairs
    two live DataFrames during training, this one aligns against a recorded list
    of column names, so it needs no reference frame and cannot be thrown off by
    dtypes.

    Args:
        X: Encoded features for a batch, from encode_categoricals.
        training_columns: The columns, in order, the model was fitted on.
        transformer: The transformer that produced X, for its categorical
            columns and their training-time category sets.

    Returns:
        X reindexed to the training-time columns, in the training-time order.
        Columns the batch does not carry are filled with zeros.
    """
    X = X.reindex(columns=list(training_columns), fill_value=0)

    if transformer.encoding == "native":
        for col in transformer.categorical_columns:
            if col in X.columns:
                categories = sorted(transformer.category_keep.get(col, set())) + [OTHER]
                X[col] = X[col].astype(pd.CategoricalDtype(categories=categories))

    return X


def resample_training_data(
    X: pd.DataFrame | sp.csr_matrix,
    y: pd.Series,
    method: str,
    encoding: str,
    random_state: int = SEED,
) -> tuple[pd.DataFrame | sp.csr_matrix, pd.Series]:
    """Rebalance the training fold only.

    Must be called after encode_categoricals()/align_columns() (and, for onehot
    encoding, after to_sparse_matrix() - X may be either a DataFrame or a scipy
    sparse matrix; all supported resamplers accept both).

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
        logger.warning("No resampling applied (method='none')")
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