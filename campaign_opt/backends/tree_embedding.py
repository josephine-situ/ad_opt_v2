"""Exact tree embedding into Gurobi (adapted from ad_opt/scripts/optimization.py)."""

from __future__ import annotations

import json
from typing import Any, Iterator

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from sklearn.tree import _tree

from campaign_opt.training_matrix import prep_xy


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


def _processed_to_raw_budget(
    thr: float,
    budget_mean: float,
    budget_scale: float,
) -> float:
    if budget_scale:
        return float(thr) * float(budget_scale) + float(budget_mean)
    return float(thr) + float(budget_mean)


def _tree_split_ok(val: Any, op: str, thr: float) -> bool:
    """Match XGBoost leaf routing: compare in float32, not promoted float64."""
    v = np.float32(val)
    t = np.float32(thr)
    if op == "lt":
        return bool(v < t)
    if op == "ge":
        return bool(v >= t)
    raise ValueError(f"unknown split op {op!r}")


def _static_feasible_leaf(
    conds: list[tuple[int, str, float]],
    x_proc_row: np.ndarray,
    budget_idx: int,
) -> bool:
    for feat_idx, op, thr in conds:
        if feat_idx == budget_idx:
            continue
        if not _tree_split_ok(x_proc_row[feat_idx], op, thr):
            return False
    return True


def _raw_budget_breakpoints_from_trees(
    tree_paths: list[list[TreePath]],
    x_proc_row: np.ndarray,
    budget_idx: int,
    budget_mean: float,
    budget_scale: float,
    budget_lo: float,
    budget_hi: float,
) -> list[float]:
    """Raw ``daily_budget`` knots where the embedded tree ensemble can change."""
    pts = {float(budget_lo), float(budget_hi)}
    for paths in tree_paths:
        for path in paths:
            conds = path[0]
            if not _static_feasible_leaf(conds, x_proc_row, budget_idx):
                continue
            for feat_idx, op, thr in conds:
                if feat_idx != budget_idx:
                    continue
                raw = _processed_to_raw_budget(thr, budget_mean, budget_scale)
                if budget_lo <= raw <= budget_hi:
                    pts.add(raw)
    return sorted(pts)


def _row_at_budget(
    template: pd.DataFrame,
    budget: float,
    target: str,
) -> pd.DataFrame:
    row = template.iloc[0:1].copy()
    row["daily_budget"] = float(budget)
    if target not in row.columns:
        row[target] = 0.0
    return row


def _strictly_increasing_pwl_points(
    x_pts: list[float],
    y_pts: list[float],
) -> tuple[list[float], list[float]]:
    """Drop duplicate x; Gurobi PWL requires strictly increasing x."""
    if not x_pts:
        return [], []
    pairs = sorted(zip(x_pts, y_pts), key=lambda t: t[0])
    xs: list[float] = []
    ys: list[float] = []
    for x, y in pairs:
        if xs and x <= xs[-1] + 1e-12:
            ys[-1] = float(y)
            continue
        xs.append(float(x))
        ys.append(float(y))
    if len(xs) < 2:
        raise ValueError("PWL requires at least two breakpoints")
    return xs, ys


def _pwl_points_for_sklearn_pipeline(
    pipeline,
    row_template: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    budget_lo: float,
    budget_hi: float,
    *,
    tree_paths: list[list[TreePath]] | None = None,
    budget_idx: int | None = None,
    budget_mean: float | None = None,
    budget_scale: float | None = None,
    x_proc_row: np.ndarray | None = None,
) -> tuple[list[float], list[float]]:
    """
  Build (x, y) breakpoints so Gurobi PWL matches ``pipeline.predict`` in budget.

  Uses sklearn at interval interiors and step boundaries so piecewise-constant
  tree ensembles are reproduced exactly on [budget_lo, budget_hi].
    """
    lo, hi = float(budget_lo), float(budget_hi)
    if hi < lo:
        lo, hi = hi, lo
    inner: list[float] = []
    if (
        tree_paths is not None
        and budget_idx is not None
        and budget_mean is not None
        and budget_scale is not None
        and x_proc_row is not None
    ):
        inner = _raw_budget_breakpoints_from_trees(
            tree_paths, x_proc_row, budget_idx, budget_mean, budget_scale, lo, hi
        )

    candidates = {lo, hi, *inner}
    span = max(hi - lo, 1.0)
    eps = max(1e-6, span * 1e-9)
    eval_pts: set[float] = set()
    for b in sorted(candidates):
        eval_pts.add(b)
        if lo < b < hi:
            eval_pts.add(max(lo, b - eps))
            eval_pts.add(min(hi, b + eps))
    for i in range(len(inner) - 1):
        eval_pts.add((inner[i] + inner[i + 1]) / 2.0)

    x_pts: list[float] = []
    y_pts: list[float] = []
    for b in sorted(eval_pts):
        row = _row_at_budget(row_template, b, target)
        X, _ = prep_xy(row, target, feature_cols)
        y = float(pipeline.predict(X)[0])
        x_pts.append(float(b))
        y_pts.append(y)
    return _strictly_increasing_pwl_points(x_pts, y_pts)


def embed_sklearn_pipeline_pwl(
    model: gp.Model,
    pipeline,
    row_template: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    budget_var: Any,
    budget_lo: float,
    budget_hi: float,
    name_prefix: str,
    *,
    tree_paths: list[list[TreePath]] | None = None,
    budget_idx: int | None = None,
    budget_mean: float | None = None,
    budget_scale: float | None = None,
    x_proc_row: np.ndarray | None = None,
) -> Any:
    """Budget response via ``addGenConstrPWL`` (linear between knots; see interval embed)."""
    x_pts, y_pts = _pwl_points_for_sklearn_pipeline(
        pipeline,
        row_template,
        target,
        feature_cols,
        budget_lo,
        budget_hi,
        tree_paths=tree_paths,
        budget_idx=budget_idx,
        budget_mean=budget_mean,
        budget_scale=budget_scale,
        x_proc_row=x_proc_row,
    )
    pred = model.addVar(lb=-GRB.INFINITY, name=f"{name_prefix}_pred")
    model.addGenConstrPWL(budget_var, pred, x_pts, y_pts, name=f"{name_prefix}_pwl")
    return pred


def _predict_at_budget(
    pipeline,
    row_template: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    budget: float,
) -> float:
    row = _row_at_budget(row_template, budget, target)
    X, _ = prep_xy(row, target, feature_cols)
    return float(pipeline.predict(X)[0])


def _jump_knots_from_budget_scan(
    pipeline,
    row_template: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    budget_lo: float,
    budget_hi: float,
    *,
    n_grid: int = 128,
) -> list[float]:
    """Knots where ``pipeline.predict`` changes on [budget_lo, budget_hi] (catches missed tree splits)."""
    lo, hi = float(budget_lo), float(budget_hi)
    if hi < lo:
        lo, hi = hi, lo
    span = max(hi - lo, 1.0)
    gap_eps = max(1e-6, span * 1e-9)
    knots = [lo]
    prev_y = _predict_at_budget(pipeline, row_template, target, feature_cols, lo)
    for t in np.linspace(lo, hi, max(8, int(n_grid))):
        t = float(t)
        if t <= lo + gap_eps:
            continue
        y = _predict_at_budget(pipeline, row_template, target, feature_cols, t)
        if abs(y - prev_y) > 1e-9:
            knots.append(t)
        prev_y = y
    knots.append(hi)
    return sorted(set(knots))


def _refine_piecewise_constant_knots(
    pipeline,
    row_template: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    knots: list[float],
    budget_lo: float,
    budget_hi: float,
) -> list[float]:
    """Bisect knot spans until sklearn is flat on each sub-interval."""
    lo, hi = float(budget_lo), float(budget_hi)
    span = max(hi - lo, 1.0)
    gap_eps = max(1e-6, span * 1e-9)
    knots = sorted({lo, hi, *(float(k) for k in knots if lo <= float(k) <= hi)})
    changed = True
    while changed:
        changed = False
        refined: list[float] = [knots[0]]
        for i in range(len(knots) - 1):
            a = float(knots[i])
            b = float(knots[i + 1])
            if b - a <= gap_eps:
                refined.append(b)
                continue
            left_pt = a if i == 0 else a + gap_eps
            right_pt = b if i == len(knots) - 2 else b - gap_eps
            if right_pt <= left_pt:
                left_pt = right_pt = 0.5 * (a + b)
            y_left = _predict_at_budget(pipeline, row_template, target, feature_cols, left_pt)
            y_right = _predict_at_budget(pipeline, row_template, target, feature_cols, right_pt)
            y_mid = _predict_at_budget(
                pipeline, row_template, target, feature_cols, 0.5 * (left_pt + right_pt)
            )
            if max(abs(y_left - y_mid), abs(y_mid - y_right), abs(y_left - y_right)) > 1e-8:
                mid = 0.5 * (a + b)
                if mid not in knots:
                    changed = True
                    refined.append(mid)
            refined.append(b)
        knots = sorted(set(refined))
    return knots


def _sklearn_budget_intervals(
    pipeline,
    row_template: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    budget_lo: float,
    budget_hi: float,
    *,
    tree_paths: list[list[TreePath]] | None = None,
    budget_idx: int | None = None,
    budget_mean: float | None = None,
    budget_scale: float | None = None,
    x_proc_row: np.ndarray | None = None,
) -> list[tuple[float, float, float]]:
    """
    Disjoint [a, b] intervals on raw ``daily_budget`` where ``pipeline.predict`` is constant.

    Tree split thresholds plus a budget scan and bisection ensure steps near ``budget_hi``
    are not collapsed into one interval labeled by a midpoint value.
    """
    lo, hi = float(budget_lo), float(budget_hi)
    if hi < lo:
        lo, hi = hi, lo
    inner: list[float] = []
    if (
        tree_paths is not None
        and budget_idx is not None
        and budget_mean is not None
        and budget_scale is not None
        and x_proc_row is not None
    ):
        inner = _raw_budget_breakpoints_from_trees(
            tree_paths, x_proc_row, budget_idx, budget_mean, budget_scale, lo, hi
        )
    scan_knots = _jump_knots_from_budget_scan(
        pipeline, row_template, target, feature_cols, lo, hi
    )
    knots = _refine_piecewise_constant_knots(
        pipeline,
        row_template,
        target,
        feature_cols,
        sorted({lo, hi, *inner, *scan_knots}),
        lo,
        hi,
    )
    span = max(hi - lo, 1.0)
    gap_eps = max(1e-6, span * 1e-9)
    merged: list[tuple[float, float, float]] = []
    for i in range(len(knots) - 1):
        a = float(knots[i])
        b = float(knots[i + 1])
        if b - a <= gap_eps:
            continue
        left_pt = a if i == 0 else a + gap_eps
        right_pt = b if i == len(knots) - 2 else b - gap_eps
        if right_pt <= left_pt:
            left_pt = right_pt = 0.5 * (a + b)
        y = _predict_at_budget(
            pipeline, row_template, target, feature_cols, 0.5 * (left_pt + right_pt)
        )
        if merged and abs(merged[-1][2] - y) < 1e-12:
            merged[-1] = (merged[-1][0], b, y)
        else:
            merged.append((a, b, y))
    if not merged:
        merged.append((lo, hi, _predict_at_budget(pipeline, row_template, target, feature_cols, 0.5 * (lo + hi))))
    return merged


def embed_sklearn_pipeline_interval(
    model: gp.Model,
    pipeline,
    row_template: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    budget_var: Any,
    budget_lo: float,
    budget_hi: float,
    name_prefix: str,
    *,
    tree_paths: list[list[TreePath]] | None = None,
    budget_idx: int | None = None,
    budget_mean: float | None = None,
    budget_scale: float | None = None,
    x_proc_row: np.ndarray | None = None,
) -> Any:
    """Exact piecewise-constant sklearn/XGB budget response (one binary per constant interval)."""
    intervals = _sklearn_budget_intervals(
        pipeline,
        row_template,
        target,
        feature_cols,
        budget_lo,
        budget_hi,
        tree_paths=tree_paths,
        budget_idx=budget_idx,
        budget_mean=budget_mean,
        budget_scale=budget_scale,
        x_proc_row=x_proc_row,
    )
    pred = model.addVar(lb=-GRB.INFINITY, name=f"{name_prefix}_pred")
    if len(intervals) == 1:
        _a, _b, y_const = intervals[0]
        model.addConstr(pred == y_const, name=f"{name_prefix}_const")
        return pred

    span = max(float(budget_hi) - float(budget_lo), 1.0)
    big_m = span * 1.01 + 1.0
    gap_eps = max(1e-6, span * 1e-9)
    z_vars: list[Any] = []
    expr = gp.LinExpr()
    n_iv = len(intervals)
    for idx, (a, b, y) in enumerate(intervals):
        z = model.addVar(vtype=GRB.BINARY, name=f"{name_prefix}_iz_{idx}")
        z_vars.append(z)
        expr += float(y) * z
        left = float(a) if idx == 0 else float(a) + gap_eps
        right = float(b) if idx == n_iv - 1 else float(b) - gap_eps
        model.addConstr(budget_var >= left - big_m * (1 - z), name=f"{name_prefix}_lo_{idx}")
        model.addConstr(budget_var <= right + big_m * (1 - z), name=f"{name_prefix}_hi_{idx}")
    model.addConstr(gp.quicksum(z_vars) == 1, name=f"{name_prefix}_one_interval")
    model.addConstr(pred == expr, name=f"{name_prefix}_def")
    return pred


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
) -> Any:
    """
    Embed one tree ensemble as a Gurobi expression in ``budget_var``.

    Leaf filtering uses structural checks only: ``_static_feasible_leaf`` for
    non-budget features and ``_leaf_budget_interval`` for budget-range overlap.
    These are derived directly from the parsed tree and are exact.
    """
    x_proc_row = np.asarray(x_proc_row, dtype=np.float32).ravel()
    min_lhs, max_proc = _processed_budget_bounds(
        budget_lo, budget_hi, budget_mean, budget_scale
    )
    proc_span = max(float(max_proc) - float(min_lhs), 1.0)
    big_m_floor = proc_span * 1.1 + 1.0
    max_lhs = float(max_proc) + 0.05 * proc_span
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

            feasible = True
            dynamic_conds: list[tuple[str, float]] = []
            for feat_idx, op, thr in conds:
                if feat_idx == budget_idx:
                    dynamic_conds.append((op, thr))
                elif not _tree_split_ok(x_proc_row[feat_idx], op, thr):
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
                    m_val = max(max_lhs - bound, big_m_floor)
                    model.addConstr(
                        lhs <= bound + m_val * (1 - z),
                        name=f"{name_prefix}_lt_{t_idx}_{leaf_idx}",
                    )
                elif op == "ge":
                    bound = float(thr)
                    m_val = max(bound - min_lhs, big_m_floor)
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
