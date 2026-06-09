"""Campaign-day feature engineering: panel, calendar, embeddings, GKP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.paths import data_path, gkp_dir, processed_dir
from utils.data_processing import (
    _extract_region_from_campaign,
    clean_keyword_sets_dataframe,
    split_keyword_field,
)
from utils.date_features import add_calendar_features
from utils.gkp_features import (
    aggregate_gkp_to_keyword_sets,
    gkp_aggregate_column_names,
    load_gkp_keyword_stats,
)

SEGMENT_CONV_PER_CLICK_RATES_CSV = "segment-conv-per-click-rates.csv"

EMBED_MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
COURSE_ANCHORS = [
    "system thinking",
    "systems thinking",
    "MIT xPRO system thinking",
    "MIT xPRO systems thinking",
]

SEMANTIC_FEATURE_COLS = [
    "embed_cohesion",
    "embed_dispersion",
    "embed_course_sim_mean",
    "embed_course_sim_p90",
]

MATCH_TYPE_LIST_COLS = {
    "Broad": "broad_keywords",
    "Phrase": "phrase_keywords",
    "Exact": "exact_keywords",
}


def _keyword_set_row_get(row: pd.Series | dict, col: str):
    if isinstance(row, dict):
        return row.get(col)
    return row[col] if col in row.index else None


def keywords_from_keyword_set_row(
    row: pd.Series | dict,
    *,
    positive_col: str = "positive_keywords",
) -> tuple[str, ...]:
    """Unique keywords from match-type list columns, else ``positive_col`` (values already cleaned at load)."""
    seen: set[str] = set()
    ordered: list[str] = []
    has_mt = False
    for col in MATCH_TYPE_LIST_COLS.values():
        raw = _keyword_set_row_get(row, col)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)) or not str(raw).strip():
            continue
        has_mt = True
        for part in split_keyword_field(raw):
            if part and part not in seen:
                seen.add(part)
                ordered.append(part)
    if not has_mt:
        raw = _keyword_set_row_get(row, positive_col)
        if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
            for part in split_keyword_field(raw):
                if part and part not in seen:
                    seen.add(part)
                    ordered.append(part)
    return tuple(sorted(ordered))


def count_unique_keywords_in_set(
    row: pd.Series | dict,
    *,
    positive_col: str = "positive_keywords",
) -> int:
    """Count unique keywords in a keyword-set row (historical or synthetic)."""
    return len(keywords_from_keyword_set_row(row, positive_col=positive_col))


def keyword_set_content_fingerprint(
    row: pd.Series | dict,
    *,
    positive_col: str = "positive_keywords",
) -> frozenset[str]:
    return frozenset(keywords_from_keyword_set_row(row, positive_col=positive_col))


def data_paths(course: str = "sys_think") -> dict[str, Path]:
    return {
        "processed": processed_dir(course),
        "gkp": gkp_dir(course),
        "cache": data_path(course, "cache"),
    }


def load_campaign_day_panel(course: str) -> pd.DataFrame:
    path = data_paths(course)["processed"] / "campaign-day-panel.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_campaign_summary(course: str) -> pd.DataFrame:
    path = data_paths(course)["processed"] / "campaign-summary.csv"
    summary = pd.read_csv(path)
    if "region" not in summary.columns and "campaign" in summary.columns:
        summary = summary.copy()
        summary["region"] = summary["campaign"].apply(_extract_region_from_campaign)
    return summary


def resolve_positive_keyword_column(keyword_sets: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Map parser keyword-set columns to positive_keywords for downstream features."""
    sets = keyword_sets.copy()
    if "positive_keywords" in sets.columns:
        return sets, "positive_keywords"
    if "unique_keywords" in sets.columns:
        sets["positive_keywords"] = sets["unique_keywords"]
        return sets, "positive_keywords"

    match_cols = ("broad_keywords", "phrase_keywords", "exact_keywords")
    if all(col in sets.columns for col in match_cols):

        def _join_positive(row: pd.Series) -> str:
            keywords: set[str] = set()
            for col in match_cols:
                raw = row.get(col, "")
                if pd.notna(raw) and str(raw).strip():
                    keywords.update(k.strip() for k in str(raw).split(";") if k.strip())
            return "; ".join(sorted(keywords))

        sets["positive_keywords"] = sets.apply(_join_positive, axis=1)
        return sets, "positive_keywords"

    missing = ", ".join(sorted({"positive_keywords", "unique_keywords", *match_cols} - set(sets.columns)))
    raise ValueError(f"campaign-keyword-sets.csv is missing keyword list column(s): {missing}")


def load_keyword_sets(course: str) -> pd.DataFrame:
    path = data_paths(course)["processed"] / "campaign-keyword-sets.csv"
    keyword_sets, _ = resolve_positive_keyword_column(pd.read_csv(path))
    return clean_keyword_sets_dataframe(keyword_sets)


def load_keyword_sets_table(course: str) -> pd.DataFrame:
    """Load base or extended keyword-set table with keyword columns cleaned."""
    processed = data_paths(course)["processed"]
    ext_path = processed / "campaign-keyword-sets-extended.csv"
    base_path = processed / "campaign-keyword-sets.csv"
    path = ext_path if ext_path.exists() else base_path
    if not path.exists():
        raise FileNotFoundError(f"No keyword sets file at {ext_path} or {base_path}")
    keyword_sets, _ = resolve_positive_keyword_column(pd.read_csv(path))
    return clean_keyword_sets_dataframe(keyword_sets)


def _merge_keyword_set_tables(base: pd.DataFrame, extended: pd.DataFrame) -> pd.DataFrame:
    """Use extended rows where present; keep base rows for ids not in extended."""
    ext = extended.set_index("keyword_set_id", drop=False)
    ext_ids = set(ext.index.astype(str))
    rows: list[pd.Series] = []
    for _, row in base.iterrows():
        kid = str(row["keyword_set_id"])
        rows.append(ext.loc[kid] if kid in ext_ids else row)
    for kid in sorted(ext_ids - {str(x) for x in base["keyword_set_id"]}):
        rows.append(ext.loc[kid])
    return pd.DataFrame(rows).reset_index(drop=True)


def load_keyword_sets_for_features(course: str) -> pd.DataFrame:
    """Historical keyword sets plus synthetic rows from campaign-keyword-sets-extended.csv."""
    from utils.keyword_allowlist import (
        apply_allowlist_to_keyword_sets,
        load_enrollment_keyword_allowlist,
    )

    base = load_keyword_sets(course)
    ext_path = data_paths(course)["processed"] / "campaign-keyword-sets-extended.csv"
    if ext_path.exists():
        extended, _ = resolve_positive_keyword_column(pd.read_csv(ext_path))
        extended = clean_keyword_sets_dataframe(extended)
        merged = _merge_keyword_set_tables(base, extended)
    else:
        merged = base
    if load_enrollment_keyword_allowlist(course) is not None:
        merged = apply_allowlist_to_keyword_sets(merged, course)
    return merged


def add_segment_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["segment"] = out["region"].astype(str) + " / " + out["match_types"].astype(str)
    return out


def parse_match_types(match_types: object) -> set[str]:
    """Parse ``Broad; Phrase; Exact`` style match-type strings."""
    if pd.isna(match_types) or not str(match_types).strip():
        return set()
    return {m.strip().title() for m in str(match_types).replace(";", " ").split() if m.strip()}


BROAD_ONLY_MATCH_TYPES = frozenset({"Broad"})
SEGMENT_BROAD_MATCH_COL = "is_broad_match"
TREE_SEGMENT_FEATURE_COLS = ["region", SEGMENT_BROAD_MATCH_COL]


def is_broad_match_campaign(match_types: object) -> bool:
    """True when the segment campaign config is Broad-only (not Phrase; Exact or mixed)."""
    return parse_match_types(match_types) == BROAD_ONLY_MATCH_TYPES


def add_segment_match_type_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose ``segment`` into region + ``is_broad_match`` for tree / ridge models.

    ``is_broad_match`` is 1 for Broad-only configs and 0 for Phrase; Exact (and other mixes).
    """
    out = df.copy()
    if "match_types" not in out.columns and "segment" in out.columns:
        parts = out["segment"].astype(str).str.split(" / ", n=1, expand=True)
        out["region"] = parts[0]
        out["match_types"] = parts[1]
    if "region" not in out.columns and "segment" in out.columns:
        out["region"] = out["segment"].astype(str).str.split(" / ", n=1).str[0]
    out[SEGMENT_BROAD_MATCH_COL] = out["match_types"].map(
        lambda m: int(is_broad_match_campaign(m))
    )
    return out


def _pairwise_mean_cosine(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return float("nan")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.clip(norms, 1e-12, None)
    sim = normed @ normed.T
    iu = np.triu_indices(len(vectors), k=1)
    if len(iu[0]) == 0:
        return float("nan")
    return float(sim[iu].mean())


def _pairwise_mean_distance(vectors: np.ndarray) -> float:
    if len(vectors) == 0:
        return float("nan")
    centroid = vectors.mean(axis=0)
    dists = 1.0 - (vectors @ centroid) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(centroid) + 1e-12
    )
    return float(np.mean(dists))


def load_or_build_embeddings(
    keywords: list[str],
    cache_path: Path,
    model_name: str = EMBED_MODEL_DEFAULT,
) -> dict[str, np.ndarray]:
    cache_path = Path(cache_path)
    requested = sorted({k.lower().strip() for k in keywords if k and str(k).strip()})
    emb_map: dict[str, np.ndarray] = {}

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if "model_name" in cached.columns and cached["model_name"].iloc[0] == model_name:
            emb_map = {
                str(row["keyword"]).lower().strip(): np.asarray(row["embedding"], dtype=float)
                for _, row in cached.iterrows()
            }

    missing = [kw for kw in requested if kw not in emb_map]
    if missing:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        embs = model.encode(missing, normalize_embeddings=True)
        for kw, vec in zip(missing, embs):
            emb_map[kw] = vec

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "keyword": list(emb_map.keys()),
                "model_name": model_name,
                "embedding": [v.tolist() for v in emb_map.values()],
            }
        ).to_parquet(cache_path, index=False)

    return emb_map


def anchor_matrix(emb_map: dict[str, np.ndarray]) -> np.ndarray:
    """Build (n_anchors, dim) matrix of COURSE_ANCHORS embeddings, computing any missing on the fly."""
    anchor_vecs = []
    for anchor_text in COURSE_ANCHORS:
        key = anchor_text.lower().strip()
        if key not in emb_map:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(EMBED_MODEL_DEFAULT)
            emb_map[key] = model.encode([anchor_text], normalize_embeddings=True)[0]
        anchor_vecs.append(emb_map[key])
    return np.stack(anchor_vecs)


# Backward-compatible alias.
_anchor_matrix = anchor_matrix


def keyword_set_semantic_features(
    keyword_sets: pd.DataFrame,
    emb_map: dict[str, np.ndarray],
    *,
    positive_col: str = "positive_keywords",
) -> pd.DataFrame:
    anchor_matrix = _anchor_matrix(emb_map)

    rows = []
    for _, row in keyword_sets.iterrows():
        set_id = row["keyword_set_id"]
        keywords = list(keywords_from_keyword_set_row(row, positive_col=positive_col))
        matched = [emb_map[k] for k in keywords if k in emb_map]
        if not matched:
            rows.append(
                {
                    "keyword_set_id": set_id,
                    "n_positive": len(keywords),
                    **{c: np.nan for c in SEMANTIC_FEATURE_COLS},
                }
            )
            continue

        vectors = np.stack(matched)
        sims = vectors @ anchor_matrix.T
        max_anchor = sims.max(axis=1)
        rows.append(
            {
                "keyword_set_id": set_id,
                "n_positive": len(keywords),
                "embed_cohesion": _pairwise_mean_cosine(vectors),
                "embed_dispersion": _pairwise_mean_distance(vectors),
                "embed_course_sim_mean": float(max_anchor.mean()),
                "embed_course_sim_p90": float(np.percentile(max_anchor, 90)),
            }
        )
    return pd.DataFrame(rows)


def _keywords_from_list_column(raw: object) -> list[str]:
    if pd.isna(raw) or not str(raw).strip():
        return []
    return [k.strip().lower() for k in str(raw).split(";") if k.strip()]


def _gkp_stats_for_keywords(keywords: list[str], gkp_map: pd.DataFrame) -> dict[str, float]:
    from utils.gkp_features import _gkp_aggregate_row

    empty = {col: float("nan") for col in gkp_aggregate_column_names()}
    if not keywords:
        return empty
    return _gkp_aggregate_row(gkp_map.reindex(keywords))


def _gkp_means_for_keywords(keywords: list[str], gkp_map: pd.DataFrame) -> dict[str, float]:
    stats = _gkp_stats_for_keywords(keywords, gkp_map)
    return {k: v for k, v in stats.items() if k.endswith("_mean")}


def keyword_set_match_type_count_features(keyword_sets: pd.DataFrame) -> pd.DataFrame:
    """Per-set keyword counts and shares by match-type column."""
    rows: list[dict[str, object]] = []
    for _, row in keyword_sets.iterrows():
        counts = {
            f"n_{mt.lower()}": len(_keywords_from_list_column(row.get(col)))
            for mt, col in MATCH_TYPE_LIST_COLS.items()
            if col in keyword_sets.columns
        }
        for mt in MATCH_TYPE_LIST_COLS:
            counts.setdefault(f"n_{mt.lower()}", 0)
        total = sum(int(counts[f"n_{mt.lower()}"]) for mt in MATCH_TYPE_LIST_COLS)
        rec: dict[str, object] = {"keyword_set_id": row["keyword_set_id"], **counts}
        for mt in MATCH_TYPE_LIST_COLS:
            key = f"n_{mt.lower()}"
            rec[f"share_{mt.lower()}"] = (
                float(counts[key]) / total if total > 0 else float("nan")
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate_gkp_per_match_type(keyword_sets: pd.DataFrame, gkp_kw: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/p90 GKP stats per match-type list column (broad / phrase / exact)."""
    base = keyword_sets[["keyword_set_id"]].drop_duplicates().copy()
    stat_cols = gkp_aggregate_column_names()
    if gkp_kw.empty:
        for mt in MATCH_TYPE_LIST_COLS:
            for stat in stat_cols:
                base[f"{mt.lower()}_{stat}"] = np.nan
        return base

    gkp_map = gkp_kw.set_index("keyword")
    rows: list[dict[str, object]] = []
    for _, row in keyword_sets.iterrows():
        rec: dict[str, object] = {"keyword_set_id": row["keyword_set_id"]}
        for mt, col in MATCH_TYPE_LIST_COLS.items():
            keywords = _keywords_from_list_column(row.get(col))
            stats = _gkp_stats_for_keywords(keywords, gkp_map)
            for stat, val in stats.items():
                rec[f"{mt.lower()}_{stat}"] = val
        rows.append(rec)
    return pd.DataFrame(rows)


def _semantic_stats_for_vectors(
    vectors: np.ndarray,
    anchor_matrix: np.ndarray,
) -> dict[str, float]:
    """Union-aligned semantic stats for one keyword embedding pool."""
    nan = float("nan")
    if len(vectors) == 0:
        return {
            "cohesion": nan,
            "dispersion": nan,
            "course_sim_mean": nan,
            "course_sim_p90": nan,
        }
    sims = vectors @ anchor_matrix.T
    max_anchor = sims.max(axis=1)
    course_mean = float(max_anchor.mean())
    course_p90 = float(np.percentile(max_anchor, 90))
    if len(vectors) < 2:
        return {
            "cohesion": nan,
            "dispersion": nan,
            "course_sim_mean": course_mean,
            "course_sim_p90": course_p90,
        }
    return {
        "cohesion": _pairwise_mean_cosine(vectors),
        "dispersion": _pairwise_mean_distance(vectors),
        "course_sim_mean": course_mean,
        "course_sim_p90": course_p90,
    }


def keyword_set_semantic_per_match_type(
    keyword_sets: pd.DataFrame,
    emb_map: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Per-match-type embedding features (mirrors union ``SEMANTIC_FEATURE_COLS`` per type).

    Columns: ``embed_cohesion_{broad|phrase|exact}``, ``embed_dispersion_*``,
    ``embed_course_sim_mean_*``, ``embed_course_sim_p90_*``.
    """
    anchor_matrix = _anchor_matrix(emb_map)
    rows: list[dict[str, object]] = []
    for _, row in keyword_sets.iterrows():
        rec: dict[str, object] = {"keyword_set_id": row["keyword_set_id"]}
        for mt, col in MATCH_TYPE_LIST_COLS.items():
            keywords = _keywords_from_list_column(row.get(col))
            matched = [emb_map[k] for k in keywords if k in emb_map]
            vectors = np.stack(matched) if matched else np.empty((0, 0))
            stats = _semantic_stats_for_vectors(vectors, anchor_matrix)
            prefix = mt.lower()
            rec[f"embed_cohesion_{prefix}"] = stats["cohesion"]
            rec[f"embed_dispersion_{prefix}"] = stats["dispersion"]
            rec[f"embed_course_sim_mean_{prefix}"] = stats["course_sim_mean"]
            rec[f"embed_course_sim_p90_{prefix}"] = stats["course_sim_p90"]
        rows.append(rec)
    return pd.DataFrame(rows)


# Columns used in match-type feature ablations (see campaign_opt/match_type_ablation.py).
MT_COUNT_FEATURE_COLS = ["n_broad", "n_phrase", "n_exact"]
MT_SHARE_FEATURE_COLS = ["share_broad", "share_phrase", "share_exact"]
GKP_SET_MEAN_COLS = [c for c in gkp_aggregate_column_names() if c.endswith("_mean")]
UNION_GKP_MEAN_COLS = ["last_month_searches_mean", "competition_index_mean"]
GKP_SET_STD_COLS = [c for c in gkp_aggregate_column_names() if c.endswith("_std")]
GKP_SET_P90_COLS = [c for c in gkp_aggregate_column_names() if c.endswith("_p90")]
GKP_SET_ALL_COLS = list(gkp_aggregate_column_names())

MT_GKP_FEATURE_COLS = [
    f"{mt.lower()}_{stat}" for mt in MATCH_TYPE_LIST_COLS for stat in GKP_SET_ALL_COLS
]
MT_GKP_MEAN_NO_BID = [
    c for c in MT_GKP_FEATURE_COLS if c.endswith("_mean") and "bid_low" not in c
]

KEYWORD_SET_STATIC_BASELINE_COLS = list(SEMANTIC_FEATURE_COLS) + ["num_unique_keywords"]
CALENDAR_BASELINE_COLS = [
    "day_of_week",
    "season",
    "is_weekend",
    "is_public_holiday",
    "days_to_next_course_start",
]
CALENDAR_EXTENDED_COLS = CALENDAR_BASELINE_COLS + ["month_sin", "month_cos"]
MT_COHESION_FEATURE_COLS = [f"embed_cohesion_{mt.lower()}" for mt in MATCH_TYPE_LIST_COLS]
MT_DISPERSION_FEATURE_COLS = [f"embed_dispersion_{mt.lower()}" for mt in MATCH_TYPE_LIST_COLS]
MT_COURSE_SIM_FEATURE_COLS = [
    col
    for mt in MATCH_TYPE_LIST_COLS
    for col in (f"embed_course_sim_mean_{mt.lower()}", f"embed_course_sim_p90_{mt.lower()}")
]
MT_SEMANTIC_FULL_FEATURE_COLS = (
    list(MT_COHESION_FEATURE_COLS)
    + list(MT_DISPERSION_FEATURE_COLS)
    + list(MT_COURSE_SIM_FEATURE_COLS)
)

# Shipped 20-feat set minus groups with |r| > ~0.85 vs a kept representative (see ablations).
SHIPPED_DEDUPED_CALENDAR = [
    "day_of_week",
    "season",
    "days_to_next_course_start",
]
SHIPPED_DEDUPED_STATIC = ["embed_course_sim_mean", "num_unique_keywords"]
SHIPPED_DEDUPED_GKP = list(UNION_GKP_MEAN_COLS)
# MT counts dropped (correlate with num_unique_keywords); union cohesion/dispersion/p90 dropped.
SHIPPED_DEDUPED_MATCH_TYPE = list(MT_DISPERSION_FEATURE_COLS)
SHIPPED_DEDUPED_CONTEXT: dict[str, list[str]] = {
    "calendar": SHIPPED_DEDUPED_CALENDAR,
    "keyword_set_static": SHIPPED_DEDUPED_STATIC,
    "gkp_set": SHIPPED_DEDUPED_GKP,
    "match_type_set": SHIPPED_DEDUPED_MATCH_TYPE,
}


def build_match_type_set_feature_table(course: str) -> pd.DataFrame:
    """All optional per-match-type keyword-set features for ablation / experiments."""
    paths = data_paths(course)
    keyword_sets = load_keyword_sets_for_features(course)

    counts = keyword_set_match_type_count_features(keyword_sets)
    gkp_kw = load_gkp_keyword_stats(paths["gkp"])
    gkp_mt = aggregate_gkp_per_match_type(keyword_sets, gkp_kw)

    all_kw: list[str] = []
    for col in (*MATCH_TYPE_LIST_COLS.values(), "positive_keywords"):
        if col not in keyword_sets.columns:
            continue
        for raw in keyword_sets[col].dropna():
            all_kw.extend(_keywords_from_list_column(raw))
    all_kw.extend(a.lower() for a in COURSE_ANCHORS)
    cache = paths["cache"] / "keyword_embeddings.parquet"
    emb_map = load_or_build_embeddings(all_kw, cache)
    semantic_mt = keyword_set_semantic_per_match_type(keyword_sets, emb_map)

    out = counts.merge(gkp_mt, on="keyword_set_id", how="left").merge(
        semantic_mt, on="keyword_set_id", how="left"
    )
    return out


def merge_match_type_set_features(panel: pd.DataFrame, course: str) -> pd.DataFrame:
    """Attach per-match-type set features to a modeling frame (requires keyword_set_id)."""
    if "keyword_set_id" not in panel.columns:
        return panel
    mt = build_match_type_set_feature_table(course)
    extra_cols = [c for c in mt.columns if c != "keyword_set_id"]
    out = panel.drop(columns=[c for c in extra_cols if c in panel.columns], errors="ignore")
    return out.merge(mt, on="keyword_set_id", how="left")


def build_keyword_set_feature_table(course: str) -> pd.DataFrame:
    paths = data_paths(course)
    keyword_sets = load_keyword_sets_for_features(course)
    positive_col = "positive_keywords"

    all_kw: list[str] = []
    for col in (*MATCH_TYPE_LIST_COLS.values(), positive_col, "unique_keywords"):
        if col not in keyword_sets.columns:
            continue
        for raw in keyword_sets[col].dropna():
            all_kw.extend(k.strip().lower() for k in str(raw).split(";") if k.strip())
    all_kw.extend(a.lower() for a in COURSE_ANCHORS)

    cache = paths["cache"] / "keyword_embeddings.parquet"
    emb_map = load_or_build_embeddings(all_kw, cache)
    sem = keyword_set_semantic_features(keyword_sets, emb_map, positive_col=positive_col)

    gkp_kw = load_gkp_keyword_stats(paths["gkp"])
    gkp_set = aggregate_gkp_to_keyword_sets(keyword_sets, gkp_kw, positive_col=positive_col)

    counts = keyword_sets.drop_duplicates(subset=["keyword_set_id"]).copy()
    counts["num_unique_keywords"] = counts.apply(
        lambda row: count_unique_keywords_in_set(row, positive_col=positive_col),
        axis=1,
    )
    counts = counts[["keyword_set_id", "num_unique_keywords"]]

    out = (
        sem.merge(gkp_set, on="keyword_set_id", how="left")
        .merge(counts, on="keyword_set_id", how="left")
    )
    mt = build_match_type_set_feature_table(course)
    mt_cols = [c for c in mt.columns if c != "keyword_set_id"]
    if mt_cols:
        out = out.merge(mt[["keyword_set_id", *mt_cols]], on="keyword_set_id", how="left")
    return out


def attach_keyword_set_to_panel(panel: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Map campaign_version -> keyword_set_id from summary."""
    if "keyword_set_id" not in summary.columns:
        return panel
    version_map = summary[["campaign_version", "keyword_set_id"]].drop_duplicates()
    out = panel.drop(columns=["keyword_set_id"], errors="ignore")
    return out.merge(version_map, on="campaign_version", how="left")


def compute_segment_conv_per_click_rates(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Course-wide conv/click per (region, match_types) from all rows in ``panel``.

    Requires columns: region, match_types, clicks, all_conv.
    Returns columns: region, match_types, seg_clicks, seg_conv, conv_per_click.
    """
    work = panel.copy()
    for col in ("region", "match_types", "clicks", "all_conv"):
        if col not in work.columns:
            raise ValueError(f"panel is missing required column for conv/click rates: {col}")

    work["clicks"] = pd.to_numeric(work["clicks"], errors="coerce")
    bad_clicks = work["clicks"].isna()
    if bad_clicks.any():
        raise ValueError("panel has non-numeric clicks values")

    work["all_conv"] = pd.to_numeric(work["all_conv"], errors="coerce")
    bad_conv = work["all_conv"].isna()
    if bad_conv.any():
        raise ValueError("panel has non-numeric all_conv values")

    seg = (
        work.groupby(["region", "match_types"], as_index=False)
        .agg(seg_clicks=("clicks", "sum"), seg_conv=("all_conv", "sum"))
    )
    zero_click = seg["seg_clicks"] <= 0.0
    if zero_click.any():
        bad = seg.loc[zero_click, ["region", "match_types"]]
        raise ValueError(
            "conv/click rate undefined for segment(s) with zero clicks: "
            f"{bad.to_dict(orient='records')}"
        )
    seg["conv_per_click"] = seg["seg_conv"] / seg["seg_clicks"]
    return seg[["region", "match_types", "seg_clicks", "seg_conv", "conv_per_click"]]


def segment_conv_per_click_rates_path(course: str) -> Path:
    return data_paths(course)["processed"] / SEGMENT_CONV_PER_CLICK_RATES_CSV


def export_segment_conv_per_click_rates(
    panel: pd.DataFrame,
    path: str | Path,
) -> pd.DataFrame:
    """Write course-wide segment conv/click rates next to the campaign-day panel."""
    rates = compute_segment_conv_per_click_rates(panel)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rates.to_csv(output_path, index=False)
    return rates


def load_course_conv_per_click_rates(course: str) -> pd.DataFrame:
    """Fixed conv/click per segment from ``segment-conv-per-click-rates.csv``."""
    path = segment_conv_per_click_rates_path(course)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run prepare-data (or generate_campaign_day_panel) "
            "to export segment conv/click rates."
        )
    rates = pd.read_csv(path)
    required = {"region", "match_types", "conv_per_click"}
    missing = required - set(rates.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rates[list(required)].copy()


def add_conversion_scaled_clicks_target(
    panel: pd.DataFrame,
    *,
    target_col: str = "conv_scaled_clicks",
    rates: pd.DataFrame | None = None,
    course: str | None = None,
) -> pd.DataFrame:
    """
    Add ``clicks * conv_per_click`` using fixed segment rates.

    When ``rates`` is omitted, loads fixed rates from ``course``'s exported CSV if set,
    otherwise computes rates from all rows in ``panel``. Rates do not vary by date.
    """
    out = panel.copy()
    if "clicks" not in out.columns:
        out["clicks"] = 0.0
    out["clicks"] = pd.to_numeric(out["clicks"], errors="coerce").fillna(0.0)

    if "all_conv" not in out.columns:
        out[target_col] = out["clicks"].astype(float)
        return out

    if rates is None:
        rates = load_course_conv_per_click_rates(course) if course else compute_segment_conv_per_click_rates(out)

    out = out.merge(rates, on=["region", "match_types"], how="left")
    fallback = float(rates["conv_per_click"].mean()) if len(rates) else 0.0
    out["conv_per_click"] = out["conv_per_click"].fillna(fallback)
    out[target_col] = out["clicks"] * out["conv_per_click"]
    return out.drop(columns=["conv_per_click"])


def build_modeling_frame(
    course: str,
    *,
    target_col: str = "all_conv",
    include_all_conv_from_summary: bool = True,
) -> pd.DataFrame:
    """
    Build segment-day modeling dataframe with calendar + set features.
    Expects campaign-day-panel with clicks/cost and optional all_conv column.
    """
    panel = load_campaign_day_panel(course)
    panel = add_segment_column(panel)
    panel = add_segment_match_type_indicators(panel)
    summary = load_campaign_summary(course)
    panel = attach_keyword_set_to_panel(panel, summary)

    set_feats = build_keyword_set_feature_table(course)
    panel = panel.merge(set_feats, on="keyword_set_id", how="left")
    panel = add_calendar_features(panel, course=course)

    if target_col == "all_conv":
        if "all_conv" not in panel.columns and include_all_conv_from_summary:
            panel["all_conv"] = np.nan
    elif target_col == "conv_scaled_clicks":
        rates = load_course_conv_per_click_rates(course)
        panel = add_conversion_scaled_clicks_target(
            panel, target_col=target_col, rates=rates, course=course
        )

    return panel.sort_values(["date", "segment"]).reset_index(drop=True)


def get_context_feature_columns(config_context: dict[str, list[str]]) -> list[str]:
    cols: list[str] = []
    for group_cols in config_context.values():
        cols.extend(group_cols)
    return list(dict.fromkeys(cols))
