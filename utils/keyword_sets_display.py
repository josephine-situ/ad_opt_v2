"""Export per-keyword-set display CSVs (Broad / Phrase / Exact columns)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils.campaign_features import MATCH_TYPE_LIST_COLS, data_paths


def _parse_keyword_list(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return sorted(k.strip() for k in str(value).split(";") if k.strip())


def keyword_set_display_frame(row: pd.Series) -> pd.DataFrame:
    """One row per keyword index; columns are match types (Broad, Phrase, Exact)."""
    by_type = {mt: _parse_keyword_list(row[col]) for mt, col in MATCH_TYPE_LIST_COLS.items()}
    n_rows = max((len(words) for words in by_type.values()), default=0)
    return pd.DataFrame({mt: (by_type[mt] + [""] * n_rows)[:n_rows] for mt in MATCH_TYPE_LIST_COLS})


def load_keyword_sets_table(course: str) -> pd.DataFrame:
    processed = data_paths(course)["processed"]
    ext_path = processed / "campaign-keyword-sets-extended.csv"
    base_path = processed / "campaign-keyword-sets.csv"
    path = ext_path if ext_path.exists() else base_path
    if not path.exists():
        raise FileNotFoundError(f"No keyword sets file at {ext_path} or {base_path}")
    return pd.read_csv(path)


def keyword_set_ids_from_plan(plan_path: Path) -> list[str]:
    plan = pd.read_csv(plan_path)
    if "keyword_set_id" not in plan.columns:
        raise ValueError(f"{plan_path} is missing keyword_set_id column")
    return sorted(plan["keyword_set_id"].dropna().astype(str).unique().tolist())


def keyword_set_ids_from_fixed(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        mapping = json.load(f)
    return sorted({str(v) for v in mapping.values() if v})


def _prune_stale_display_files(out_dir: Path, keep_set_ids: set[str]) -> None:
    """Remove display CSVs that are not in ``keep_set_ids`` (keeps ``segment_index.csv``)."""
    for path in out_dir.glob("*.csv"):
        if path.name == "segment_index.csv":
            continue
        if path.stem not in keep_set_ids:
            path.unlink()


def export_keyword_sets_display(
    course: str,
    *,
    output_dir: Path | None = None,
    keyword_set_ids: list[str] | None = None,
    segment_plan: pd.DataFrame | None = None,
    prune_stale: bool = True,
) -> Path:
    """
    Write ``data/<course>/processed/keyword-sets-display/<keyword_set_id>.csv``.

    When ``keyword_set_ids`` is None, exports every set in the extended (or base) table.
    When ``prune_stale`` is True, deletes display CSVs not in the exported id list.
    """
    sets_df = load_keyword_sets_table(course)
    missing = [col for col in MATCH_TYPE_LIST_COLS.values() if col not in sets_df.columns]
    if missing:
        raise ValueError(f"Keyword sets table missing columns: {', '.join(missing)}")

    if keyword_set_ids is not None:
        wanted = {str(k) for k in keyword_set_ids}
        sets_df = sets_df[sets_df["keyword_set_id"].astype(str).isin(wanted)].copy()
        missing_ids = wanted - set(sets_df["keyword_set_id"].astype(str))
        if missing_ids:
            raise ValueError(f"Keyword set id(s) not found in keyword sets table: {sorted(missing_ids)}")

    out_dir = output_dir or (data_paths(course)["processed"] / "keyword-sets-display")
    out_dir.mkdir(parents=True, exist_ok=True)

    export_ids = set(sets_df["keyword_set_id"].astype(str))
    if prune_stale:
        _prune_stale_display_files(out_dir, export_ids)

    index_rows: list[dict[str, str]] = []
    for _, row in sets_df.iterrows():
        set_id = str(row["keyword_set_id"])
        display_df = keyword_set_display_frame(row)
        display_path = out_dir / f"{set_id}.csv"
        display_df.to_csv(display_path, index=False)
    if segment_plan is not None and {"segment", "keyword_set_id"}.issubset(segment_plan.columns):
        index_rows = [
            {
                "segment": str(row["segment"]),
                "keyword_set_id": str(row["keyword_set_id"]),
                "source": str(row["source"]) if "source" in segment_plan.columns else "",
                "display_file": f"{row['keyword_set_id']}.csv",
            }
            for _, row in segment_plan.iterrows()
        ]
    else:
        for _, row in sets_df.iterrows():
            set_id = str(row["keyword_set_id"])
            index_rows.append(
                {"segment": "", "keyword_set_id": set_id, "source": "", "display_file": f"{set_id}.csv"}
            )

    if index_rows:
        pd.DataFrame(index_rows).to_csv(out_dir / "segment_index.csv", index=False)
    elif (out_dir / "segment_index.csv").exists():
        (out_dir / "segment_index.csv").unlink()

    return out_dir
