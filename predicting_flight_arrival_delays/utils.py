"""Generic utility methods"""

import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import joblib
from loguru import logger
import mlflow
import mlflow.artifacts
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
import requests
import yaml


def to_pascal_case(name: str) -> str:
    """Convert a field name to PascalCase, handling snake_case and mixed-case alike.
    Args:
        name: The field name to convert.

    Returns:
        The name in PascalCase.
    """
    parts = name.split("_")
    return "".join(part[:1].upper() + part[1:] if part else "" for part in parts)



def safe_relative_path(path: Path) -> str:
    """Safely format a file path relative to current working directory if possible.

    Args:
        path (Path): Target path to format.

    Returns:
        str: Relative path string if path is under CWD, otherwise absolute string path.
    """
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)
    


def fetch(url: str, params: dict, timeout=180) -> dict:
    """Send a GET request to an endpoint and return the decoded JSON.
    
    Args:
        url: Endpoint to call.
        params: Query parameters, passed through to requests.

    Returns:
        The parsed JSON response.

    Raises:
        requests.HTTPError: If the response carries an error status.
    """
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# MLflow model bundles: model + transformer + training columns, registered and
# recovered as one self-contained unit.
# ---------------------------------------------------------------------------
 
def register_model_bundle(
    model: Any,
    transformer: Any,
    columns: list[str],
    registered_model_name: str,
    signature_sample: pd.DataFrame | None = None,
    artifact_path: str = "model",
    alias: str | None = None,
) -> None:
    """Log a model together with its transformer and training columns as one bundle.
 
    Must be called inside an active MLflow run (i.e. within an
    `mlflow.start_run()` block). transformer and columns are logged into the SAME
    artifact_path as the model itself - not loosely on the run - so a registered
    model version is self-contained: everything needed to prepare data for it is
    recoverable from the registry alone.
 
    Args:
        model: Fitted, scikit-learn-compatible estimator to register. Must expose
            predict_proba if signature_sample is given.
        transformer: Fitted object exposing a '.save(path)' method (e.g. this
            project's Transformer).
        columns: Exact feature column names/order the model was fitted on.
        registered_model_name: Name to register this model version under.
        signature_sample: A few rows of already-transformed features, used to
            infer the model's input/output signature and a logged input example.
            Optional; skipped (no signature) if not given.
        artifact_path: Subpath, under the run's artifacts, model/transformer/
            columns are stored together under. Defaults to "model".
        alias: If given, this version is promoted under this alias immediately
            after registration. Optional; no alias is set if not given.
    """
    with tempfile.TemporaryDirectory() as tmp:
        transformer_path = Path(tmp) / "transformer.joblib"
        columns_path = Path(tmp) / "columns.json"
 
        transformer.save(transformer_path)
        columns_path.write_text(json.dumps(list(columns)))
 
        signature, input_example = None, None
        if signature_sample is not None:
            signature = infer_signature(
                signature_sample, model.predict_proba(signature_sample)[:, 1]
            )
            input_example = signature_sample.head(5)
 
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )
        mlflow.log_artifact(str(transformer_path), artifact_path=artifact_path)
        mlflow.log_artifact(str(columns_path), artifact_path=artifact_path)
 
        if alias is not None:
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(
                registered_model_name, alias, model_info.registered_model_version
            )
 
 
def _resolve_run_id(registered_model_name: str, stage: str) -> str:
    """Resolve the run that produced a registered model version.
 
    Args:
        registered_model_name: Name the model version was registered under.
        stage: Registry stage or version alias. "None" or "" resolves to the
            latest version instead.
 
    Returns:
        The run id that logged that model version.
 
    Raises:
        FileNotFoundError: If no registered version exists.
    """
    client = mlflow.MlflowClient()
 
    if stage not in ("None", ""):
        return client.get_model_version_by_alias(registered_model_name, stage).run_id
 
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    if not versions:
        raise FileNotFoundError(f"No registered versions found for {registered_model_name}")
    return max(versions, key=lambda v: int(v.version)).run_id
 



def load_model_bundle(
    registered_model_name: str,
    stage: str = "None",
    artifact_path: str = "model",
) -> tuple[Any, Any, list[str], str]:
    """Load a model, its transformer and training columns, all from the registry.
 
    Args:
        registered_model_name: Name the model version was registered under.
        stage: Registry stage or version alias to resolve. "None" (MLflow's
            default for unstaged models) resolves to the latest version.
        artifact_path: Must match the artifact_path used in register_model_bundle.
 
    Returns:
        (estimator, transformer, columns, run_id), columns in training order.
        run_id is returned so callers can read back other params/tags logged on
        that run (e.g. "variant").
 
    Raises:
        FileNotFoundError: If no registered version exists.
    """
    run_id = _resolve_run_id(registered_model_name, stage)
 
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/{artifact_path}")
 
    transformer_path = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/{artifact_path}/transformer.joblib"
    )
    transformer = joblib.load(transformer_path)
 
    columns_path = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/{artifact_path}/columns.json"
    )
    columns = json.loads(Path(columns_path).read_text())
 
    return model, transformer, columns, run_id

def get_run_params(run_id: str) -> dict[str, str]:
    """Fetch all parameters logged on an MLflow run.

    Args:
        run_id: The run to fetch parameters from.

    Returns:
        Mapping of param name to value.
    """
    return mlflow.MlflowClient().get_run(run_id).data.params



# ---------------------------------------------------------------------------
# DVC
# ---------------------------------------------------------------------------

def get_dvc_data_hash(output_path: str, dvc_lock_path: Path = Path("dvc.lock")) -> str:
    """The DVC hash currently recorded for a specific pipeline output path.

    Args:
        output_path: The path as it appears in dvc.lock's "outs", e.g.
            "data/processed/final/all".
        dvc_lock_path: Path to dvc.lock. Defaults to the repo root's dvc.lock.

    Returns:
        The md5 hash DVC has recorded for that output, or "not_found" if
        dvc.lock is missing or does not list that path.
    """
    try:
        lock = yaml.safe_load(dvc_lock_path.read_text())
        for stage in lock["stages"].values():
            for out in stage.get("outs", []):
                if out["path"] == output_path:
                    return out["md5"]
    except Exception as e:
        logger.exception(f"Could not resolve DVC hash: {e}")
    return "not_found"

# ---------------------------------------------------------------------------
# GIT
# ---------------------------------------------------------------------------
def get_git_dirty() -> bool | None:
    """Whether there are uncommitted changes right now.

    If True, the git commit that MLflow auto-tags on this run (mlflow.source.git.commit)
    may not reflect the code that actually ran - there were local edits not yet
    committed at run time. None if git itself is unavailable.

    Returns:
        True/False, or None if git status could not be determined.
    """
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"]).decode()
        return bool(status.strip())
    except Exception:
        return None
