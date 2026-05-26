"""Compile and summarize campaign backtest outputs (plan vs actual, LaTeX tables)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campaign_opt.decisions import region_of_segment

REGIONS = ("USA", "A", "B")


def backtest_window_dir(
    course: str,
    exp_name: str,
    start: str,
    end: str,
    *,
    base: Path | None = None,
) -> Path:
    root = base or Path("opt_results")
    return root / course / "campaign" / exp_name / "backtest" / f"{start}_{end}"


def save_backtest_config(backtest_dir: Path, payload: dict[str, Any]) -> Path:
    backtest_dir = Path(backtest_dir)
    backtest_dir.mkdir(parents=True, exist_ok=True)
    path = backtest_dir / "backtest_config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_backtest_config(backtest_dir: Path) -> dict[str, Any]:
    path = Path(backtest_dir) / "backtest_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _day_from_path(path: Path) -> str | None:
    parent = path.parent.name
    if len(parent) == 8 and parent.isdigit():
        return f"{parent[:4]}-{parent[4:6]}-{parent[6:8]}"
    return None


def collect_plan_vs_actual(backtest_dir: Path) -> pd.DataFrame:
    """Load all per-day or per-week plan_vs_actual CSVs under plans/."""
    backtest_dir = Path(backtest_dir)
    plans_dir = backtest_dir / "plans"
    if not plans_dir.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in sorted(plans_dir.rglob("plan_vs_actual*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        if "date" not in df.columns:
            day = _day_from_path(path)
            if day:
                df = df.copy()
                df["date"] = day
        if "period" not in df.columns:
            if "weekly" in path.name:
                df = df.copy()
                df["period"] = "week"
            else:
                df = df.copy()
                df["period"] = "day"
        df["source_file"] = str(path.relative_to(backtest_dir))
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
    return out


def _agg_day(group: pd.DataFrame, target: str) -> dict[str, Any]:
    obs_col = f"observed_{target}"
    row: dict[str, Any] = {
        "Day": group["date"].iloc[0] if "date" in group.columns else None,
        "n_segments": len(group),
        "pred_lift_total": float(group["pred_lift"].sum()) if "pred_lift" in group.columns else 0.0,
        "actual_model_lift_total": float(group["actual_model_lift"].sum())
        if "actual_model_lift" in group.columns
        else 0.0,
        "opt_budget_total": float(group["daily_budget"].sum()) if "daily_budget" in group.columns else 0.0,
        "act_budget_total": float(group["actual_budget"].sum())
        if "actual_budget" in group.columns
        else 0.0,
    }
    if obs_col in group.columns:
        row["observed_total"] = float(group[obs_col].sum())
    elif "observed_clicks" in group.columns:
        row["observed_total"] = float(group["observed_clicks"].sum())
    else:
        row["observed_total"] = None
    return row


def compile_evaluation_results(
    backtest_dir: Path,
    *,
    target: str = "clicks",
) -> pd.DataFrame:
    """
    Build a daily evaluation table from plan_vs_actual files.
    Falls back to daily_backtest_summary.csv / weekly_backtest_summary.csv when present.
    """
    backtest_dir = Path(backtest_dir)
    plan_df = collect_plan_vs_actual(backtest_dir)

    if not plan_df.empty and "date" in plan_df.columns:
        daily_parts = []
        for day, grp in plan_df.groupby("date", dropna=False):
            if pd.isna(day):
                continue
            row = _agg_day(grp, target)
            row["Day"] = day
            daily_parts.append(row)
        if daily_parts:
            return pd.DataFrame(daily_parts)

    daily_path = backtest_dir / "daily_backtest_summary.csv"
    if daily_path.exists():
        df = pd.read_csv(daily_path)
        rename = {
            "opt_date": "Day",
            "pred_lift_total": "pred_lift_total",
            "actual_model_lift_total": "actual_model_lift_total",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if "Day" not in df.columns and "opt_date" in df.columns:
            df["Day"] = df["opt_date"]
        return df

    weekly_path = backtest_dir / "weekly_backtest_summary.csv"
    if weekly_path.exists():
        df = pd.read_csv(weekly_path)
        if "week_start" in df.columns:
            df["Day"] = df["week_start"]
        return df

    return pd.DataFrame()


def _window_dates_from_dir(backtest_dir: Path) -> tuple[str, str]:
    """Parse start/end from folder name like 2025-10-06_2025-10-12."""
    name = Path(backtest_dir).name
    if "_" in name:
        parts = name.split("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return "", ""


def _find_missing_days(backtest_dir: Path, start: str, end: str) -> list[str]:
    cfg = load_backtest_config(backtest_dir)
    dir_start, dir_end = _window_dates_from_dir(backtest_dir)
    start_s = str(cfg.get("start_day") or start or dir_start)
    end_s = str(cfg.get("end_day") or end or dir_end)
    if not start_s or not end_s:
        return []
    start_day = pd.Timestamp(start_s)
    end_day = pd.Timestamp(end_s)
    plans_dir = Path(backtest_dir) / "plans"
    observed: set = set()
    if plans_dir.exists():
        for sub in plans_dir.iterdir():
            if sub.is_dir() and len(sub.name) == 8 and sub.name.isdigit():
                marker = sub / "campaign_plan.csv"
                if marker.exists():
                    observed.add(
                        pd.Timestamp(
                            f"{sub.name[:4]}-{sub.name[4:6]}-{sub.name[6:8]}"
                        ).date()
                    )
    missing = [
        d.date().isoformat()
        for d in pd.date_range(start_day, end_day, freq="D")
        if d.date() not in observed
    ]
    return missing


def _add_region_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "segment" not in df.columns:
        return df
    out = df.copy()
    out["region"] = out["segment"].astype(str).map(region_of_segment)
    return out


def regional_breakdown(
    plan_df: pd.DataFrame,
    *,
    target: str = "clicks",
) -> pd.DataFrame:
    """Spend / lift / observed share by region (opt vs actual)."""
    if plan_df.empty:
        return pd.DataFrame()

    df = _add_region_columns(plan_df)
    obs_col = f"observed_{target}"
    rows: list[dict[str, Any]] = []

    for label, budget_col in (("Opt", "daily_budget"), ("Act", "actual_budget")):
        totals = {
            "budget": float(df[budget_col].sum()) if budget_col in df.columns else 0.0,
            "lift": float(df["pred_lift" if label == "Opt" else "actual_model_lift"].sum())
            if ("pred_lift" if label == "Opt" else "actual_model_lift") in df.columns
            else 0.0,
            "observed": float(df[obs_col].sum())
            if obs_col in df.columns
            else float(df["observed_clicks"].sum())
            if "observed_clicks" in df.columns
            else 0.0,
        }
        row: dict[str, Any] = {"scenario": label}
        for reg in REGIONS:
            sub = df[df["region"] == reg]
            row[f"Spend {reg}"] = (
                float(sub[budget_col].sum()) / totals["budget"] if totals["budget"] > 0 else 0.0
            )
            lift_col = "pred_lift" if label == "Opt" else "actual_model_lift"
            row[f"Lift {reg}"] = (
                float(sub[lift_col].sum()) / totals["lift"] if totals["lift"] > 0 else 0.0
            )
            if obs_col in sub.columns or "observed_clicks" in sub.columns:
                obs_sum = float(sub[obs_col].sum()) if obs_col in sub.columns else float(sub["observed_clicks"].sum())
                row[f"Observed {reg}"] = obs_sum / totals["observed"] if totals["observed"] > 0 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_performance(
    eval_df: pd.DataFrame,
    *,
    target: str = "clicks",
) -> pd.DataFrame:
    """Mean ± SE performance table (opt plan vs actual decisions vs observed)."""
    if eval_df.empty:
        return pd.DataFrame()

    df = eval_df.copy()
    if "Day" in df.columns:
        df = df.drop_duplicates(subset=["Day"])

    def _mean_se(series: pd.Series) -> tuple[float, float]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return 0.0, 0.0
        return float(s.mean()), float(s.sem()) if len(s) > 1 else 0.0

    pred_m, pred_se = _mean_se(df.get("pred_lift_total", pd.Series(dtype=float)))
    act_m, act_se = _mean_se(df.get("actual_model_lift_total", pd.Series(dtype=float)))
    obs_m, obs_se = _mean_se(df.get("observed_total", pd.Series(dtype=float)))
    opt_cost_m, opt_cost_se = _mean_se(df.get("opt_budget_total", pd.Series(dtype=float)))
    act_cost_m, act_cost_se = _mean_se(df.get("act_budget_total", pd.Series(dtype=float)))

    imp_lift = (pred_m - act_m) / act_m if act_m > 0 else 0.0
    imp_obs = (pred_m - obs_m) / obs_m if obs_m and obs_m > 0 else 0.0
    clicks_per_dollar_opt = pred_m / opt_cost_m if opt_cost_m > 0 else 0.0
    clicks_per_dollar_act = act_m / act_cost_m if act_cost_m > 0 else 0.0

    row = {
        "target": target,
        "n_days": len(df),
        "avg pred_lift (opt)": pred_m,
        "se pred_lift (opt)": pred_se,
        "avg model_lift (act)": act_m,
        "se model_lift (act)": act_se,
        "avg observed": obs_m,
        "se observed": obs_se,
        "avg budget (opt)": opt_cost_m,
        "se budget (opt)": opt_cost_se,
        "avg budget (act)": act_cost_m,
        "se budget (act)": act_cost_se,
        "lift/$ (opt)": clicks_per_dollar_opt,
        "lift/$ (act)": clicks_per_dollar_act,
        "improvement pred vs act lift": imp_lift,
        "improvement pred vs observed": imp_obs,
    }
    if "rmse_pred_vs_actual_model_lift" in df.columns:
        row["mean rmse pred vs act lift"] = float(
            pd.to_numeric(df["rmse_pred_vs_actual_model_lift"], errors="coerce").mean()
        )
    return pd.DataFrame([row])


def generate_performance_latex(summary_df: pd.DataFrame) -> str:
    """LaTeX table for backtest performance (campaign-level metrics)."""
    if summary_df.empty:
        return ""

    df = summary_df.copy()

    def fmt(mean: float, se: float, decimals: int = 1) -> str:
        return f"{mean:,.{decimals}f} $\\pm$ {se:,.{decimals}f}"

    rows = []
    r = df.iloc[0]
    rows.append(
        [
            "Pred lift (opt)",
            fmt(r["avg pred_lift (opt)"], r["se pred_lift (opt)"]),
            fmt(r["avg budget (opt)"], r["se budget (opt)"], 2),
            f"{r['lift/$ (opt)']:.3f}",
        ]
    )
    rows.append(
        [
            "Model lift (act)",
            fmt(r["avg model_lift (act)"], r["se model_lift (act)"]),
            fmt(r["avg budget (act)"], r["se budget (act)"], 2),
            f"{r['lift/$ (act)']:.3f}",
        ]
    )
    if pd.notna(r.get("avg observed")):
        rows.append(
            [
                f"Observed ({r.get('target', 'target')})",
                fmt(r["avg observed"], r["se observed"]),
                "---",
                "---",
            ]
        )

    out = pd.DataFrame(rows, columns=["Metric", "Lift / Observed", "Budget", "Lift/\\$"])
    latex = out.to_latex(index=False, escape=False, column_format="lccc")
    latex = latex.replace(r"\hline", r"\toprule", 1)
    if latex.strip().endswith(r"\hline"):
        latex = latex.strip()[:-6] + r"\bottomrule"

    imp = r.get("improvement pred vs act lift")
    note = ""
    if imp is not None and pd.notna(imp):
        note = (
            f"\\scriptsize Improvement (pred vs act lift): {100 * float(imp):,.1f}\\%."
        )

    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Campaign backtest performance (mean $\\pm$ SE over days)}\n"
        f"{latex}\n"
        f"{note}\n"
        "\\end{table}\n"
    )


def analyze_backtest_run(
    backtest_dir: Path,
    *,
    target: str = "clicks",
    write_latex: bool = True,
) -> dict[str, Any]:
    """Compile evaluation CSV, summary, regional breakdown, and optional LaTeX."""
    backtest_dir = Path(backtest_dir)
    cfg = load_backtest_config(backtest_dir)
    target = cfg.get("target") or target

    eval_df = compile_evaluation_results(backtest_dir, target=target)
    plan_df = collect_plan_vs_actual(backtest_dir)
    missing_days = _find_missing_days(
        backtest_dir,
        str(cfg.get("start_day") or ""),
        str(cfg.get("end_day") or ""),
    )

    eval_path = backtest_dir / "evaluation_results.csv"
    if not eval_df.empty:
        eval_df.to_csv(eval_path, index=False)

    summary_df = summarize_performance(eval_df, target=target)
    summary_path = backtest_dir / "backtest_summary.csv"
    if not summary_df.empty:
        summary_df["n_missing_days"] = len(missing_days)
        summary_df["missing_days"] = ", ".join(missing_days)
        summary_df.to_csv(summary_path, index=False)

    regional_df = regional_breakdown(plan_df, target=target)
    regional_path = backtest_dir / "regional_breakdown.csv"
    if not regional_df.empty:
        regional_df.to_csv(regional_path, index=False)

    latex_path = backtest_dir / "backtest_summary.tex"
    latex_text = ""
    if write_latex and not summary_df.empty:
        latex_text = generate_performance_latex(summary_df)
        latex_path.write_text(latex_text, encoding="utf-8")

    return {
        "backtest_dir": str(backtest_dir),
        "evaluation_results": str(eval_path) if eval_path.exists() else None,
        "backtest_summary": str(summary_path) if summary_path.exists() else None,
        "regional_breakdown": str(regional_path) if regional_path.exists() else None,
        "backtest_summary_tex": str(latex_path) if latex_path.exists() else None,
        "n_days_evaluated": len(eval_df),
        "missing_days": missing_days,
        "latex": latex_text,
    }
