"""Compile and summarize campaign backtest outputs (plan vs actual, LaTeX tables).

``analyze_backtest_run(out_dir)``
    Inputs: backtest window dir with ``plans/*/plan_vs_actual.csv``.
    Outputs: ``evaluation_results.csv``, ``backtest_summary.csv``,
    ``regional_breakdown.csv``, ``backtest_summary.tex``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campaign_opt.decisions import region_of_segment

REGIONS = ("USA", "A", "B")


def backtest_window_dir(exp_name: str, start: str, end: str) -> Path:
    from campaign_opt.paths import backtest_window_dir as _backtest_window_dir

    return _backtest_window_dir(exp_name, start, end)


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


def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV; return empty frame if missing or zero-byte (pandas EmptyDataError)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _compile_from_campaign_plans(backtest_dir: Path) -> pd.DataFrame:
    """Minimal per-day table from plans/YYYYMMDD/campaign_plan.csv (no ensemble metrics)."""
    plans_dir = Path(backtest_dir) / "plans"
    if not plans_dir.exists():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for sub in sorted(plans_dir.iterdir()):
        if not sub.is_dir() or len(sub.name) != 8 or not sub.name.isdigit():
            continue
        plan_path = sub / "campaign_plan.csv"
        plan = _safe_read_csv(plan_path)
        if plan.empty:
            continue
        day = f"{sub.name[:4]}-{sub.name[4:6]}-{sub.name[6:8]}"
        if "opt_date" in plan.columns and pd.notna(plan["opt_date"].iloc[0]):
            day = str(plan["opt_date"].iloc[0])
        budget = pd.to_numeric(plan.get("daily_budget"), errors="coerce").fillna(0.0)
        rows.append(
            {
                "Day": day,
                "n_segments": len(plan),
                "opt_budget_total": float(budget.sum()),
                "n_segments_zero_budget": int((budget <= 0).sum()),
            }
        )
    return pd.DataFrame(rows)


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
        df = _safe_read_csv(path)
        if df.empty:
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
    if "row_kind" in group.columns:
        plan = group[group["row_kind"] == "plan"]
        market = group[group["row_kind"] == "market"]
    else:
        plan = group
        market = group.iloc[0:0]
    row: dict[str, Any] = {
        "Day": group["date"].iloc[0] if "date" in group.columns else None,
        "n_segments": len(plan) if len(plan) else len(group),
        "pred_lift_total": float(plan["pred_lift"].sum()) if "pred_lift" in plan.columns and len(plan) else 0.0,
        "actual_model_lift_total": float(market["actual_model_lift"].sum())
        if "actual_model_lift" in market.columns and len(market)
        else float(group["actual_model_lift"].sum())
        if "actual_model_lift" in group.columns
        else 0.0,
        "pred_lift_raw_total": float(plan["pred_lift_raw"].sum())
        if "pred_lift_raw" in plan.columns and len(plan)
        else None,
        "actual_model_lift_raw_total": float(market["actual_model_lift_raw"].sum())
        if "actual_model_lift_raw" in market.columns and len(market)
        else None,
        "opt_budget_total": float(plan["daily_budget"].sum()) if "daily_budget" in plan.columns and len(plan) else 0.0,
        "act_budget_total": float(market["daily_budget"].sum())
        if "daily_budget" in market.columns and len(market)
        else float(group["actual_budget"].sum())
        if "actual_budget" in group.columns
        else 0.0,
    }
    obs_src = market if len(market) and obs_col in market.columns else group
    if obs_col in obs_src.columns:
        row["observed_total"] = float(pd.to_numeric(obs_src[obs_col], errors="coerce").sum())
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
    df = _safe_read_csv(daily_path)
    if not df.empty:
        rename = {
            "opt_date": "Day",
            "pred_lift_total": "pred_lift_total",
            "actual_model_lift_total": "actual_model_lift_total",
            "plan_budget_total": "opt_budget_total",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if "Day" not in df.columns and "opt_date" in df.columns:
            df["Day"] = df["opt_date"]
        return df

    weekly_path = backtest_dir / "weekly_backtest_summary.csv"
    df = _safe_read_csv(weekly_path)
    if not df.empty:
        if "week_start" in df.columns:
            df["Day"] = df["week_start"]
        return df

    return _compile_from_campaign_plans(backtest_dir)


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


def _regional_lift_column(df: pd.DataFrame, clipped_col: str, raw_col: str) -> str:
    if raw_col in df.columns and pd.to_numeric(df[raw_col], errors="coerce").notna().any():
        return raw_col
    return clipped_col


def _scenario_slice(plan_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Plan rows for Model; market (panel campaign) rows for Actual."""
    if "row_kind" not in plan_df.columns:
        return plan_df
    kind = "plan" if scenario == "Model" else "market"
    return plan_df[plan_df["row_kind"] == kind].copy()


def regional_breakdown(
    plan_df: pd.DataFrame,
    *,
    target: str = "clicks",
) -> pd.DataFrame:
    """
    Regional **shares** of spend, model incremental lift, and observed target.

    ``Conversions {region}`` is each region's fraction of total incremental lift
    (not absolute conversions). Model uses optimizer plan rows; Actual uses panel
    market rows only, with raw lift when available (same as ``backtest_summary``).
    """
    if plan_df.empty:
        return pd.DataFrame()

    plan_df = _add_region_columns(plan_df)
    obs_col = f"observed_{target}"
    market_obs = _scenario_slice(plan_df, "Actual")
    rows: list[dict[str, Any]] = []

    scenario_specs = (
        (
            "Model",
            _scenario_slice(plan_df, "Model"),
            "daily_budget",
            _regional_lift_column(plan_df, "pred_lift", "pred_lift_raw"),
        ),
        (
            "Actual",
            _scenario_slice(plan_df, "Actual"),
            "daily_budget",
            _regional_lift_column(plan_df, "actual_model_lift", "actual_model_lift_raw"),
        ),
    )

    obs_total = 0.0
    if obs_col in market_obs.columns:
        obs_total = float(pd.to_numeric(market_obs[obs_col], errors="coerce").sum())
    elif "observed_clicks" in market_obs.columns:
        obs_total = float(pd.to_numeric(market_obs["observed_clicks"], errors="coerce").sum())

    for label, slice_df, budget_col, conv_col in scenario_specs:
        if slice_df.empty:
            continue
        lift = pd.to_numeric(slice_df[conv_col], errors="coerce") if conv_col in slice_df.columns else pd.Series(dtype=float)
        budget = (
            pd.to_numeric(slice_df[budget_col], errors="coerce")
            if budget_col in slice_df.columns
            else pd.Series(dtype=float)
        )
        totals = {
            "budget": float(budget.sum()),
            "conversions": float(lift.sum()),
        }
        row: dict[str, Any] = {"scenario": label}
        for reg in REGIONS:
            sub = slice_df[slice_df["region"] == reg]
            row[f"Spend {reg}"] = (
                float(pd.to_numeric(sub[budget_col], errors="coerce").sum()) / totals["budget"]
                if totals["budget"] > 0 and budget_col in sub.columns
                else 0.0
            )
            if totals["conversions"] != 0 and conv_col in sub.columns:
                row[f"Conversions {reg}"] = (
                    float(pd.to_numeric(sub[conv_col], errors="coerce").sum()) / totals["conversions"]
                )
            else:
                row[f"Conversions {reg}"] = 0.0
            obs_sub = market_obs[market_obs["region"] == reg]
            if obs_total > 0 and (obs_col in obs_sub.columns or "observed_clicks" in obs_sub.columns):
                if obs_col in obs_sub.columns:
                    obs_sum = float(pd.to_numeric(obs_sub[obs_col], errors="coerce").sum())
                else:
                    obs_sum = float(pd.to_numeric(obs_sub["observed_clicks"], errors="coerce").sum())
                row[f"Observed {reg}"] = obs_sum / obs_total
            else:
                row[f"Observed {reg}"] = 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def _daily_lift_total_column(df: pd.DataFrame, clipped_col: str, raw_col: str) -> str:
    """Prefer signed daily totals when present (clipped totals are often ~0 for market rows)."""
    if raw_col in df.columns:
        raw = pd.to_numeric(df[raw_col], errors="coerce")
        if raw.notna().any():
            return raw_col
    return clipped_col


def summarize_performance(
    eval_df: pd.DataFrame,
    *,
    target: str = "clicks",
) -> pd.DataFrame:
    """Mean ± SE table with Model and Actual rows (ad_opt-style layout)."""
    if eval_df.empty:
        return pd.DataFrame()

    df = eval_df.copy()
    if "Day" in df.columns:
        df = df.drop_duplicates(subset=["Day"])

    def _mean_se(col: str) -> tuple[float, float]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return 0.0, 0.0
        return float(s.mean()), float(s.sem()) if len(s) > 1 else 0.0

    model_lift_col = _daily_lift_total_column(df, "pred_lift_total", "pred_lift_raw_total")
    actual_lift_col = _daily_lift_total_column(
        df, "actual_model_lift_total", "actual_model_lift_raw_total"
    )

    model_conv_m, model_conv_se = _mean_se(model_lift_col)
    actual_conv_m, actual_conv_se = _mean_se(actual_lift_col)
    model_budget_m, model_budget_se = _mean_se("opt_budget_total")
    actual_budget_m, actual_budget_se = _mean_se("act_budget_total")

    model_conv_per_dollar = model_conv_m / model_budget_m if model_budget_m > 0 else 0.0
    actual_conv_per_dollar = actual_conv_m / actual_budget_m if actual_budget_m > 0 else 0.0

    if actual_conv_m != 0:
        imp_conv = (model_conv_m - actual_conv_m) / abs(actual_conv_m)
    else:
        imp_conv = np.nan
    if actual_conv_per_dollar != 0:
        imp_conv_per_dollar = (model_conv_per_dollar - actual_conv_per_dollar) / abs(actual_conv_per_dollar)
    else:
        imp_conv_per_dollar = np.nan

    n_days = len(df)
    rows = [
        {
            "scenario": "Model",
            "target": target,
            "n_days": n_days,
            "conversions": model_conv_m,
            "conversions_se": model_conv_se,
            "budget": model_budget_m,
            "budget_se": model_budget_se,
            "conversions_per_dollar": model_conv_per_dollar,
            "improvement_conversions_pct": 100.0 * imp_conv if pd.notna(imp_conv) else np.nan,
            "improvement_conversions_per_dollar_pct": 100.0 * imp_conv_per_dollar
            if pd.notna(imp_conv_per_dollar)
            else np.nan,
        },
        {
            "scenario": "Actual",
            "target": target,
            "n_days": n_days,
            "conversions": actual_conv_m,
            "conversions_se": actual_conv_se,
            "budget": actual_budget_m,
            "budget_se": actual_budget_se,
            "conversions_per_dollar": actual_conv_per_dollar,
            "improvement_conversions_pct": np.nan,
            "improvement_conversions_per_dollar_pct": np.nan,
        },
    ]
    out = pd.DataFrame(rows)
    out["lift_source"] = (
        "raw" if actual_lift_col.endswith("_raw_total") or model_lift_col.endswith("_raw_total") else "clipped"
    )
    if "rmse_pred_vs_actual_model_lift" in df.columns:
        out["mean_rmse_model_vs_actual"] = float(
            pd.to_numeric(df["rmse_pred_vs_actual_model_lift"], errors="coerce").mean()
        )
    return out


def generate_performance_latex(summary_df: pd.DataFrame) -> str:
    """LaTeX table for backtest performance (Model / Actual rows, ad_opt-style)."""
    if summary_df.empty:
        return ""

    def fmt(mean: float, se: float, decimals: int = 1) -> str:
        return f"{mean:,.{decimals}f} $\\pm$ {se:,.{decimals}f}"

    def fmt_pct(val: float) -> str:
        if val is None or not pd.notna(val):
            return "---"
        return f"{float(val):,.1f}\\%"

    rows: list[list[str]] = []
    for _, r in summary_df.iterrows():
        rows.append(
            [
                str(r["scenario"]),
                fmt(r["conversions"], r["conversions_se"]),
                fmt(r["budget"], r["budget_se"], 2),
                f"{r['conversions_per_dollar']:.3f}",
                fmt_pct(r.get("improvement_conversions_pct")),
                fmt_pct(r.get("improvement_conversions_per_dollar_pct")),
            ]
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "Scenario",
            "Conversions",
            "Budget",
            "Conv/\\$",
            "Imp. Conv.",
            "Imp. Conv/\\$",
        ],
    )
    latex = out.to_latex(index=False, escape=False, column_format="lccccc")
    latex = latex.replace(r"\hline", r"\toprule", 1)
    if latex.strip().endswith(r"\hline"):
        latex = latex.strip()[:-6] + r"\bottomrule"

    target = summary_df["target"].iloc[0] if "target" in summary_df.columns else "target"
    caption = (
        f"Campaign backtest performance (mean $\\pm$ SE over days; target: {target})"
    )

    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"{latex}\n"
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
