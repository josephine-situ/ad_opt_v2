"""Exact tree embedding into Gurobi (adapted from ad_opt/scripts/optimization.py)."""

from __future__ import annotations

import json
from typing import Any, Iterator

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from sklearn.tree import _tree

from campaign_opt.modeling import _prep_xy


TreePath = tuple[list[tuple[int, str, float]], float, int | None]
EPSILON = 1e-5


def _as_float(value: Any) -> float:
    """Parse XGBoost numeric values that may be bracket-wrapped strings."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    return float(text)


def _parse_xgb_tree(node: dict, current_conds: list[tuple[int, str, float]]) -> Iterator[TreePath]:
    if "leaf" in node:
        node_id = int(node["nodeid"]) if "nodeid" in node else None
        yield (current_conds, float(node["leaf"]), node_id)
        return
    try:
        feat_id = int(node["split"].replace("f", ""))
    except ValueError:
        return
    threshold = float(node["split_condition"])
    yes_id = node["yes"]
    no_id = node["no"]
    yes_child = next(c for c in node["children"] if c["nodeid"] == yes_id)
    no_child = next(c for c in node["children"] if c["nodeid"] == no_id)
    yield from _parse_xgb_tree(yes_child, current_conds + [(feat_id, "lt", threshold)])
    yield from _parse_xgb_tree(no_child, current_conds + [(feat_id, "ge", threshold)])


def _parse_sklearn_tree(tree, current_conds: list[tuple[int, str, float]]) -> Iterator[TreePath]:
    tree_ = tree.tree_
    if tree_.feature[0] == _tree.TREE_UNDEFINED:
        return

    def recurse(node_id: int, conds: list[tuple[int, str, float]]) -> Iterator[TreePath]:
        if tree_.feature[node_id] == _tree.TREE_UNDEFINED:
            yield (conds, float(tree_.value[node_id][0, 0, 0]), int(node_id))
            return
        feat_id = int(tree_.feature[node_id])
        threshold = float(tree_.threshold[node_id])
        left = int(tree_.children_left[node_id])
        right = int(tree_.children_right[node_id])
        yield from recurse(left, conds + [(feat_id, "lt", threshold)])
        yield from recurse(right, conds + [(feat_id, "ge", threshold)])

    yield from recurse(0, current_conds)


def _leaf_budget_interval(
    dynamic_conds: list[tuple[str, float]],
) -> tuple[float, float] | None:
    """Processed-budget interval implied by a leaf's budget splits (empty if unconstrained)."""
    low = float("-inf")
    high = float("inf")
    for op, thr in dynamic_conds:
        if op == "lt":
            high = min(high, float(thr) - EPSILON)
        elif op == "ge":
            low = max(low, float(thr))
    if low > high:
        return None
    return low, high


def _intervals_overlap(
    a: tuple[float, float],
    b: tuple[float, float],
) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _booster_leaf_nodes_at_budgets(
    pipeline,
    x_raw_row: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    budget_lo: float,
    budget_hi: float,
) -> dict[int, set[int]]:
    """Map tree index -> leaf node ids the booster uses at lo/mid/hi raw budgets."""
    import xgboost as xgb

    estimator = pipeline.named_steps["model"]
    booster = estimator.get_booster()
    probes = sorted({float(budget_lo), float(budget_hi), (float(budget_lo) + float(budget_hi)) / 2})
    allowed: dict[int, set[int]] = {}
    for budget in probes:
        probe = x_raw_row.copy()
        probe["daily_budget"] = budget
        if target not in probe.columns:
            probe[target] = 0.0
        x_raw, _ = _prep_xy(probe, target, feature_cols)
        x_proc = np.asarray(pipeline[:-1].transform(x_raw), dtype=np.float32)
        leaf_ids = booster.predict(xgb.DMatrix(x_proc), pred_leaf=True)[0]
        for tree_idx, node_id in enumerate(leaf_ids):
            allowed.setdefault(int(tree_idx), set()).add(int(node_id))
    return allowed


def _tighten_allowed_leaf_nodes(
    allowed: dict[int, set[int]],
    pipeline,
    x_raw_row: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    budget_lo: float,
    budget_hi: float,
) -> dict[int, set[int]]:
    """
    When several leaves survive static pruning, keep only those the booster uses at
  midpoint budget (intersected with the lo/mid/hi probe set).
    """
    mid_budget = (float(budget_lo) + float(budget_hi)) / 2.0
    mid_nodes = _booster_leaf_nodes_at_budgets(
        pipeline,
        x_raw_row,
        target,
        feature_cols,
        mid_budget,
        mid_budget,
    )
    tightened: dict[int, set[int]] = {}
    for tree_idx, nodes in allowed.items():
        if len(nodes) <= 1:
            tightened[tree_idx] = nodes
            continue
        narrowed = nodes & mid_nodes.get(tree_idx, set())
        tightened[tree_idx] = narrowed if narrowed else nodes
    return tightened


def get_tree_path_sets(pipeline) -> tuple[list[list[TreePath]], float, str]:
    """
    Return (paths_per_tree, base_or_scale, kind).
    kind is 'xgboost' (sum trees + base) or 'random_forest' (mean trees).
    """
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "get_booster"):
        booster = estimator.get_booster()
        config = json.loads(booster.save_config())
        base_score = _as_float(config["learner"]["learner_model_param"]["base_score"])
        paths = []
        for tree_json in booster.get_dump(dump_format="json"):
            paths.append(list(_parse_xgb_tree(json.loads(tree_json), [])))
        return paths, base_score, "xgboost"

    if hasattr(estimator, "estimators_"):
        paths = [list(_parse_sklearn_tree(est, [])) for est in estimator.estimators_]
        return paths, float(len(estimator.estimators_)), "random_forest"

    raise TypeError(f"Unsupported tree model for embedding: {type(estimator)}")


def _budget_affine(pipeline) -> tuple[int, float, float]:
    """Index and StandardScaler (mean, scale) for ``daily_budget`` in processed features."""
    preprocessor = pipeline.named_steps["prep"]
    feature_names = list(preprocessor.get_feature_names_out())
    budget_idx = feature_names.index("num__daily_budget")
    num = preprocessor.named_transformers_["num"]
    pos = list(num.feature_names_in_).index("daily_budget")
    mean = float(num.mean_[pos])
    scale = float(num.scale_[pos])
    return budget_idx, mean, scale


def _budget_scale(pipeline) -> tuple[int, float]:
    """Backward-compatible: returns (budget_idx, scale) only."""
    budget_idx, _, scale = _budget_affine(pipeline)
    return budget_idx, scale


def _processed_budget_bounds(
    budget_lo: float,
    budget_hi: float,
    budget_mean: float,
    budget_scale: float,
) -> tuple[float, float]:
    """Scaled budget range used for Big-M (matches sklearn StandardScaler)."""
    if budget_scale:
        return (
            (float(budget_lo) - budget_mean) / budget_scale,
            (float(budget_hi) - budget_mean) / budget_scale,
        )
    return float(budget_lo) - budget_mean, float(budget_hi) - budget_mean


def embed_tree_prediction(
    model: gp.Model,
    *,
    tree_paths: list[list[TreePath]],
    x_proc_row: np.ndarray,
    budget_var: Any,
    budget_lo: float,
    budget_hi: float,
    budget_idx: int,
    budget_mean: float,
    budget_scale: float,
    model_kind: str,
    base_or_n_trees: float,
    name_prefix: str,
    allowed_leaf_nodes: dict[int, set[int]] | None = None,
) -> Any:
    """
    Embed one tree ensemble as a Gurobi expression in ``budget_var``.

    Budget splits use processed features ``(budget - mean) / scale`` (same as training).
    """
    x_proc_row = np.asarray(x_proc_row, dtype=np.float32).ravel()
    min_lhs, max_proc = _processed_budget_bounds(
        budget_lo, budget_hi, budget_mean, budget_scale
    )
    max_lhs = max_proc * 1.05
    budget_range = (min_lhs, max_proc)
    if budget_scale:
        lhs = (budget_var - budget_mean) / budget_scale
    else:
        lhs = budget_var - budget_mean
    tree_sum = gp.LinExpr()

    for t_idx, paths in enumerate(tree_paths):
        leaf_vars: list[Any] = []
        leaf_vals: list[float] = []

        for leaf_idx, path in enumerate(paths):
            conds, leaf_val = path[0], path[1]
            node_id = path[2] if len(path) > 2 else None
            if (
                allowed_leaf_nodes is not None
                and node_id is not None
                and node_id not in allowed_leaf_nodes.get(t_idx, set())
            ):
                continue

            feasible = True
            dynamic_conds: list[tuple[str, float]] = []
            for feat_idx, op, thr in conds:
                if feat_idx == budget_idx:
                    dynamic_conds.append((op, thr))
                elif op == "lt" and not (float(x_proc_row[feat_idx]) < float(thr)):
                    feasible = False
                    break
                elif op == "ge" and not (float(x_proc_row[feat_idx]) >= float(thr)):
                    feasible = False
                    break

            if not feasible:
                continue

            if dynamic_conds:
                leaf_interval = _leaf_budget_interval(dynamic_conds)
                if leaf_interval is None or not _intervals_overlap(leaf_interval, budget_range):
                    continue

            z = model.addVar(vtype=GRB.BINARY, name=f"{name_prefix}_z_{t_idx}_{leaf_idx}")
            leaf_vars.append(z)
            leaf_vals.append(float(leaf_val))

            for op, thr in dynamic_conds:
                if op == "lt":
                    # ad_opt clamps to 0 because Cost uses StandardScaler(with_mean=False).
                    # Here budget is (x-mean)/scale and can be negative — do not clamp.
                    bound = float(thr) - EPSILON
                    m_val = max(max_lhs - bound, 0.0)
                    model.addConstr(
                        lhs <= bound + m_val * (1 - z),
                        name=f"{name_prefix}_lt_{t_idx}_{leaf_idx}",
                    )
                elif op == "ge":
                    bound = float(thr)
                    m_val = max(bound - min_lhs, 0.0)
                    model.addConstr(
                        lhs >= bound - m_val * (1 - z),
                        name=f"{name_prefix}_ge_{t_idx}_{leaf_idx}",
                    )

        if leaf_vars:
            model.addConstr(gp.quicksum(leaf_vars) == 1, name=f"{name_prefix}_one_{t_idx}")
            tree_sum += gp.LinExpr(leaf_vals, leaf_vars)

    pred_var = model.addVar(lb=-GRB.INFINITY, name=f"{name_prefix}_pred")
    if model_kind == "xgboost":
        model.addConstr(
            pred_var == tree_sum + float(base_or_n_trees),
            name=f"{name_prefix}_def",
        )
    else:
        n_trees = max(base_or_n_trees, 1.0)
        model.addConstr(pred_var == tree_sum / n_trees, name=f"{name_prefix}_def")
    return pred_var
