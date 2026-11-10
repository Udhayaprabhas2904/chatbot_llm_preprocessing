
"""Helpers for planning experiment configurations."""


def build_experiment_matrix(
    queries,
    models,
    conditions=("baseline", "preprocessed"),
):
    if not queries:
        raise ValueError("Queries must not be empty.")

    if not models:
        raise ValueError("Models must not be empty.")

    if not conditions:
        raise ValueError("Conditions must not be empty.")

    if any(not str(query).strip() for query in queries):
        raise ValueError("Queries must not contain empty strings.")

    if len(set(conditions)) != len(conditions):
        raise ValueError("Duplicate conditions are not allowed.")

    invalid = set(conditions) - {"baseline", "preprocessed"}

    if invalid:
        raise ValueError(f"Unsupported conditions: {sorted(invalid)}")

    return [
        {
            "query": query,
            "model": model,
            "condition": condition,
        }
        for query in queries
        for model in models
        for condition in conditions
    ]
