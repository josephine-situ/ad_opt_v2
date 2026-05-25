"""Build per-segment keyword-set candidates K_s."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.campaign_features import add_segment_column, load_campaign_summary, load_keyword_sets


def _keywords_from_panel(
    kw_day: pd.DataFrame,
    segment_row: pd.Series,
    *,
    top_n: int = 30,
) -> list[str]:
    region = segment_row["region"]
    match_types = str(segment_row["match_types"])
    allowed = {m.strip().title() for m in match_types.replace(";", " ").split()}
    sub = kw_day[kw_day["region"] == region].copy()
    if allowed:
        sub = sub[sub["match_type"].isin(allowed)]
    if sub.empty:
        return []

    agg = (
        sub.groupby("keyword")
        .agg(clicks=("clicks", "sum"), cost=("cost", "sum"))
        .reset_index()
    )
    agg["efficiency"] = agg["clicks"] / agg["cost"].clip(lower=0.01)
    top_click = set(agg.nlargest(top_n, "clicks")["keyword"].tolist())
    top_eff = set(agg.nlargest(top_n, "efficiency")["keyword"].tolist())
    return sorted(top_click | top_eff)


def build_segment_candidates(
    course: str,
    *,
    top_n: int = 30,
    synthetic_prefix: str = "synthetic",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        candidates: segment, keyword_set_id, source (historical|synthetic)
        extended_sets: keyword_set_id, positive_keywords (for new synthetic ids)
    """
    summary = load_campaign_summary(course)
    summary = add_segment_column(summary)
    keyword_sets = load_keyword_sets(course)
    positive_col = "positive_keywords"
    if positive_col not in keyword_sets.columns:
        positive_col = next(
            (c for c in keyword_sets.columns if "positive" in c.lower()),
            "positive_keywords",
        )

    kw_path = Path("data") / course / "processed" / "kw-day-panel.csv"
    kw_day = pd.read_csv(kw_path) if kw_path.exists() else pd.DataFrame()

    hist_rows = []
    for segment, grp in summary.groupby("segment", sort=False):
        for set_id in grp["keyword_set_id"].dropna().unique():
            hist_rows.append(
                {
                    "segment": segment,
                    "region": grp["region"].iloc[0],
                    "match_types": grp["match_types"].iloc[0],
                    "keyword_set_id": set_id,
                    "source": "historical",
                }
            )

    hist_df = pd.DataFrame(hist_rows)
    existing_ids = set(keyword_sets["keyword_set_id"].astype(str))
    set_lookup = keyword_sets.set_index("keyword_set_id")

    synthetic_sets = []
    synth_cand = []
    synth_idx = 0

    if not kw_day.empty:
        for segment, grp in summary.groupby("segment", sort=False):
            row = grp.iloc[0]
            kws = _keywords_from_panel(kw_day, row, top_n=top_n)
            if not kws:
                continue
            synth_idx += 1
            new_id = f"{synthetic_prefix}_{segment.replace(' / ', '_')}_{synth_idx}"
            while new_id in existing_ids:
                synth_idx += 1
                new_id = f"{synthetic_prefix}_{segment.replace(' / ', '_')}_{synth_idx}"
            existing_ids.add(new_id)
            pos = "; ".join(sorted(kws))
            synthetic_sets.append({"keyword_set_id": new_id, positive_col: pos})
            synth_cand.append(
                {
                    "segment": segment,
                    "region": row["region"],
                    "match_types": row["match_types"],
                    "keyword_set_id": new_id,
                    "source": "synthetic_top",
                }
            )

    synth_df = pd.DataFrame(synth_cand)
    candidates = pd.concat([hist_df, synth_df], ignore_index=True)
    candidates = candidates.drop_duplicates(subset=["segment", "keyword_set_id"])

    extended = keyword_sets.copy()
    if synthetic_sets:
        extended = pd.concat([extended, pd.DataFrame(synthetic_sets)], ignore_index=True)

    return candidates, extended
