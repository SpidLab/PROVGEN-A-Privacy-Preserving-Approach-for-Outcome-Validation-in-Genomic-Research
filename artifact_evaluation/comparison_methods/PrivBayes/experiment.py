"""PrivBayes baseline wrapper used by the artifact evaluation pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
import DataSynthesizer.DataGenerator as data_generator_module


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = Path(__file__).resolve().parent


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    return value


def _normalize_parent_key(raw_key: str) -> str:
    if "np." not in raw_key:
        return raw_key

    parsed = eval(raw_key, {"np": np, "__builtins__": {}}, {})
    parsed = _to_builtin(parsed)
    if not isinstance(parsed, list):
        parsed = [parsed]
    return repr(parsed)


def _sanitize_description_file(description_file: Path) -> None:
    description = json.loads(description_file.read_text(encoding="utf-8"))
    conditional = description.get("conditional_probabilities", {})
    changed = False

    for attribute, probabilities in list(conditional.items()):
        if not isinstance(probabilities, dict):
            continue
        normalized: dict[str, list[float]] = {}
        for raw_key, distribution in probabilities.items():
            new_key = _normalize_parent_key(raw_key)
            if new_key != raw_key:
                changed = True
            normalized[new_key] = distribution
        conditional[attribute] = normalized

    if changed:
        description_file.write_text(json.dumps(description, indent=4) + "\n", encoding="utf-8")


def generate_synthetic_from_csv(
    *,
    input_data_path: Path,
    epsilon: float,
    generation_count: int,
    description_file: Path,
    synthetic_data_file: Path | None,
) -> None:
    input_data = pd.read_csv(input_data_path, index_col=None)

    # DataSynthesizer serializes some parent instances as ``np.int64(...)`` under
    # newer Python/NumPy combinations. Its generator later evals those strings in
    # the module global scope, so register ``np`` and normalize the JSON to plain
    # Python literals for compatibility across versions.
    data_generator_module.np = np

    threshold_value = 4
    categorical_attributes = {column: True for column in input_data.columns.values if column != "SNP_ID"}
    candidate_keys = {"PATIENT_ID": True}
    degree_of_bayesian_network = 2

    describer = DataDescriber(category_threshold=threshold_value)
    describer.describe_dataset_in_correlated_attribute_mode(
        dataset_file=str(input_data_path),
        epsilon=epsilon,
        k=degree_of_bayesian_network,
        attribute_to_is_categorical=categorical_attributes,
        attribute_to_is_candidate_key=candidate_keys,
    )
    description_file.parent.mkdir(parents=True, exist_ok=True)
    describer.save_dataset_description_to_file(description_file)
    _sanitize_description_file(description_file)

    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(generation_count, description_file)
    if synthetic_data_file is not None:
        synthetic_data_file.parent.mkdir(parents=True, exist_ok=True)
        generator.save_synthetic_data(synthetic_data_file)


def run(dataset_name: str, snp_count: int, epsilon: float, generation_count: int, idx: int) -> int:
    input_data_path = EXPERIMENT_ROOT / "data" / "cleansed" / dataset_name / f"data_{snp_count}_{idx}.csv"
    description_file = METHOD_ROOT / "descriptions" / f"{dataset_name}_{epsilon}_{snp_count}_{idx}.json"
    synthetic_data_file = EXPERIMENT_ROOT / "generated" / "privbayes" / f"{dataset_name}_{epsilon}_{snp_count}_{idx}.csv"
    generate_synthetic_from_csv(
        input_data_path=input_data_path,
        epsilon=epsilon,
        generation_count=generation_count,
        description_file=description_file,
        synthetic_data_file=synthetic_data_file,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    print([Path(__file__).name, *args])
    dataset_name, snp_count, epsilon, generation_count, idx = (
        args[0],
        int(args[1]),
        float(args[2]),
        int(args[3]),
        int(args[4]),
    )
    return run(dataset_name, snp_count, epsilon, generation_count, idx)


if __name__ == "__main__":
    raise SystemExit(main())
