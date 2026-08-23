"""Shared Great Expectations plumbing for the data suites.
"""

import json
from pathlib import Path

import great_expectations as gx
from loguru import logger
import pandas as pd
import pyarrow.parquet as pq


SAMPLE_ROWS = 200_000


def load_parquet_sample(
    path: Path, n_rows: int = SAMPLE_ROWS, columns: list[str] | None = None
) -> pd.DataFrame:
    """Read up to n_rows spread across the file, without loading it whole.

    Args:
        path: Parquet file to sample.
        n_rows: Upper bound on the rows returned.
        columns: Restrict to these columns; None reads them all.

    Returns:
        A DataFrame of at most n_rows.
    """
    parquet = pq.ParquetFile(path)
    n_groups = parquet.metadata.num_row_groups
    stride = max(1, n_groups // 8)

    frames, taken = [], 0
    for index in range(0, n_groups, stride):
        frame = parquet.read_row_group(index, columns=columns).to_pandas()
        frames.append(frame)
        taken += len(frame)
        if taken >= n_rows:
            break

    return pd.concat(frames, ignore_index=True).head(n_rows)


def validate(df: pd.DataFrame, expectations: list, name: str):
    """Run a list of expectations against a DataFrame.

    Args:
        df: The data to validate.
        expectations: Instances from gx.expectations.
        name: Prefix for the source, asset, suite and checkpoint names.

    Returns:
        The checkpoint result, whose .success says whether everything held.
    """
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas(name=f"{name}_source")
    data_asset = data_source.add_dataframe_asset(name=f"{name}_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch_definition")

    suite = context.suites.add(
        gx.core.expectation_suite.ExpectationSuite(name=f"{name}_suite")
    )
    for expectation in expectations:
        suite.add_expectation(expectation)
    context.suites.add_or_update(suite)

    validation_definition = context.validation_definitions.add(
        gx.core.validation_definition.ValidationDefinition(
            name=f"{name}_validation", data=batch_definition, suite=suite
        )
    )
    checkpoint = context.checkpoints.add(
        gx.checkpoint.checkpoint.Checkpoint(
            name=f"{name}_checkpoint", validation_definitions=[validation_definition]
        )
    )

    return checkpoint.run(batch_parameters={"dataframe": df})


def failures(checkpoint_result) -> list[str]:
    """Summarise the expectations that did not hold, for an assertion message.

    Args:
        checkpoint_result: What validate() returned.

    Returns:
        One readable line per failed expectation; empty when everything passed.
    """
    described = json.loads(checkpoint_result.describe())
    lines = []

    for validation in described.get("validation_results", []):
        for expectation in validation.get("expectations", []):
            if expectation["success"]:
                continue
            kwargs = expectation.get("kwargs", {})
            target = kwargs.get("column") or kwargs.get("column_A") or "table"
            lines.append(
                f"{target}: {expectation['expectation_type']} -> {expectation.get('result')}"
            )
    return lines


def show_results(checkpoint_result) -> None:
    """Print every expectation's outcome. Used when running as a script."""
    described = json.loads(checkpoint_result.describe())

    for index, validation in enumerate(described.get("validation_results", []), 1):
        logger.info(f"Validation {index}: {validation['success']}")

        for expectation in validation.get("expectations", []):
            kwargs = expectation.get("kwargs", {})
            target = kwargs.get("column") or kwargs.get("column_A") or "table"
            logger.info(
                f"{target}: {expectation['expectation_type']} -> {expectation['success']}"
            )
            if not expectation["success"]:
                logger.warning(kwargs)
                logger.warning(expectation.get("result"))

    if described.get("success"):
        logger.success("Overall success: True")
    else:
        logger.error("Overall success: False")
