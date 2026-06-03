"""Campaign-day feature engineering: panel, calendar, embeddings, GKP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import COURSE_CONFIG
from utils.data_processing import _extract_region_from_campaign
from utils.date_features import add_calendar_features
from utils.gkp_features import aggregate_gkp_to_keyword_sets, load_gkp_keyword_stats

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


def data_paths(course: str) -> dict[str, Path]:
    base = Path("data") / course
    return {
        "processed": base / "processed",
        "gkp": base / "gkp",
        "cache": base / "cache",
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
    return keyword_sets


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


def _anchor_matrix(emb_map: dict[str, np.ndarray]) -> np.ndarray:
    anchor_vecs = []
    for anchor in COURSE_ANCHORS:
        key = anchor.lower().strip()
        if key not in emb_map:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(EMBED_MODEL_DEFAULT)
            emb_map[key] = model.encode([anchor], normalize_embeddings=True)[0]
        anchor_vecs.append(emb_map[key])
    return np.stack(anchor_vecs)


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
        raw = row.get(positive_col, "")
        keywords = (
            [k.strip().lower() for k in str(raw).split(";") if k.strip()]
            if pd.notna(raw)
            else []
        )
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


def _gkp_means_for_keywords(keywords: list[str], gkp_map: pd.DataFrame) -> dict[str, float]:
    if not keywords:
        return {
            "last_month_searches_mean": float("nan"),
            "competition_index_mean": float("nan"),
            "bid_low_mean": float("nan"),
        }
    sub = gkp_map.reindex(keywords)
    return {
        "last_month_searches_mean": float(sub["last_month_searches"].mean()),
        "competition_index_mean": float(sub["competition_index"].mean()),
        "bid_low_mean": float(sub["bid_low"].mean()),
    }


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
    """Mean GKP stats per match-type list column (broad / phrase / exact)."""
    base = keyword_sets[["keyword_set_id"]].drop_duplicates().copy()
    if gkp_kw.empty:
        for mt in MATCH_TYPE_LIST_COLS:
            for stat in ("last_month_searches_mean", "competition_index_mean", "bid_low_mean"):
                base[f"{mt.lower()}_{stat}"] = np.nan
        return base

    gkp_map = gkp_kw.set_index("keyword")
    rows: list[dict[str, object]] = []
    for _, row in keyword_sets.iterrows():
        rec: dict[str, object] = {"keyword_set_id": row["keyword_set_id"]}
        for mt, col in MATCH_TYPE_LIST_COLS.items():
            keywords = _keywords_from_list_column(row.get(col))
            means = _gkp_means_for_keywords(keywords, gkp_map)
            for stat, val in means.items():
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
MT_GKP_FEATURE_COLS = [
    f"{mt.lower()}_{stat}"
    for mt in MATCH_TYPE_LIST_COLS
    for stat in ("last_month_searches_mean", "competition_index_mean", "bid_low_mean")
]
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

    summary = load_campaign_summary(course)
    if "num_unique_keywords" in summary.columns:
        counts = summary.groupby("keyword_set_id")["num_unique_keywords"].max().reset_index()
    else:
        counts = sem[["keyword_set_id", "n_positive"]].rename(
            columns={"n_positive": "num_unique_keywords"}
        )

    out = (
        sem.merge(gkp_set, on="keyword_set_id", how="left")
        .merge(counts, on="keyword_set_id", how="left")
    )
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

    Returns columns: region, match_types, conv_per_click.
    """
    work = panel.copy()
    work["clicks"] = pd.to_numeric(work.get("clicks", 0), errors="coerce").fillna(0.0)
    if "all_conv" not in work.columns:
        work["conv_per_click"] = 0.0
        return work[["region", "match_types", "conv_per_click"]].drop_duplicates()

    work["all_conv"] = pd.to_numeric(work["all_conv"], errors="coerce").fillna(0.0)
    total_clicks = float(work["clicks"].sum())
    total_conv = float(work["all_conv"].sum())
    global_rate = total_conv / total_clicks if total_clicks > 0 else 0.0

    seg = (
        work.groupby(["region", "match_types"], as_index=False)
        .agg(seg_clicks=("clicks", "sum"), seg_conv=("all_conv", "sum"))
    )
    seg["conv_per_click"] = np.where(
        seg["seg_clicks"] > 0.0,
        seg["seg_conv"] / seg["seg_clicks"],
        global_rate,
    )
    return seg[["region", "match_types", "conv_per_click"]]


def load_course_conv_per_click_rates(course: str) -> pd.DataFrame:
    """Fixed conv/click per segment from the full processed campaign-day panel."""
    panel = load_campaign_day_panel(course)
    return compute_segment_conv_per_click_rates(panel)


_version_start_cache: dict[str, dict[object, pd.Timestamp]] = {}


def version_start_dates(summary: pd.DataFrame) -> dict[object, pd.Timestamp]:
    """``campaign_version`` -> version ``start_date``."""
    if "campaign_version" not in summary.columns or "start_date" not in summary.columns:
        return {}
    starts = summary[["campaign_version", "start_date"]].drop_duplicates()
    return {
        row["campaign_version"]: pd.Timestamp(row["start_date"])
        for _, row in starts.iterrows()
    }


def version_start_dates_for_course(course: str) -> dict[object, pd.Timestamp]:
    if course not in _version_start_cache:
        _version_start_cache[course] = version_start_dates(load_campaign_summary(course))
    return _version_start_cache[course]


def version_for_segment_on_date(
    panel: pd.DataFrame,
    segment: str,
    on_date: pd.Timestamp,
) -> object | None:
    """``campaign_version`` active for ``segment`` on ``on_date`` (latest row on or before date)."""
    if panel.empty or "campaign_version" not in panel.columns:
        return None
    sub = panel.copy()
    sub["date"] = pd.to_datetime(sub["date"])
    seg_mask = sub["segment"].astype(str) == str(segment)
    sub = sub.loc[seg_mask]
    if sub.empty:
        return None
    d = pd.Timestamp(on_date).normalize()
    on_day = sub[sub["date"] == d]
    row = on_day.iloc[-1] if not on_day.empty else sub[sub["date"] <= d].sort_values("date").iloc[-1]
    ver = row.get("campaign_version")
    return None if pd.isna(ver) else ver


def days_since_version_start_value(
    on_date: pd.Timestamp,
    campaign_version: object | None,
    *,
    course: str | None = None,
    version_starts: dict[object, pd.Timestamp] | None = None,
) -> float:
    if campaign_version is None or pd.isna(campaign_version):
        return float("nan")
    starts = version_starts if version_starts is not None else version_start_dates_for_course(course or "")
    start = starts.get(campaign_version)
    if start is None:
        return float("nan")
    return float((pd.Timestamp(on_date).normalize() - pd.Timestamp(start).normalize()).days)


def version_run_vector_for_date(
    planning_date: pd.Timestamp,
    *,
    course: str,
    campaign_version: object | None = None,
    segment: str | None = None,
    panel: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Regime features tied to campaign version start (for scoring / MILP rows)."""
    ver = campaign_version
    if (ver is None or pd.isna(ver)) and segment is not None and panel is not None:
        ver = version_for_segment_on_date(panel, segment, planning_date)
    return {
        "days_since_version_start": days_since_version_start_value(
            planning_date, ver, course=course
        )
    }


def add_version_run_features(panel: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Add ``days_since_version_start`` from summary version ``start_date``."""
    if "campaign_version" not in panel.columns:
        out = panel.copy()
        out["days_since_version_start"] = np.nan
        return out
    starts = version_start_dates(summary)
    if not starts:
        out = panel.copy()
        out["days_since_version_start"] = np.nan
        return out

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    start_s = out["campaign_version"].map(starts)
    out["days_since_version_start"] = (out["date"] - start_s).dt.days.astype(float)
    return out


def add_conversion_scaled_clicks_target(
    panel: pd.DataFrame,
    *,
    target_col: str = "conv_scaled_clicks",
    rates: pd.DataFrame | None = None,
    course: str | None = None,
) -> pd.DataFrame:
    """
    Add ``clicks * conv_per_click`` using fixed segment rates.

    When ``rates`` is omitted, uses ``course`` (full campaign-day panel) if set,
    otherwise rates from all rows in ``panel``. Rates do not vary by date.
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
    panel = add_version_run_features(panel, summary)

    set_feats = build_keyword_set_feature_table(course)
    panel = panel.merge(set_feats, on="keyword_set_id", how="left")
    panel = add_calendar_features(panel, course=course)

    if target_col == "all_conv":
        if "all_conv" in panel.columns:
            panel["all_conv"] = panel["all_conv"].fillna(0.0)
        elif include_all_conv_from_summary:
            panel["all_conv"] = np.nan
    elif target_col == "conv_scaled_clicks":
        rates = load_course_conv_per_click_rates(course)
        panel = add_conversion_scaled_clicks_target(
            panel, target_col=target_col, rates=rates, course=course
        )

    if "clicks" not in panel.columns:
        panel["clicks"] = 0

    return panel.sort_values(["date", "segment"]).reset_index(drop=True)


def get_context_feature_columns(config_context: dict[str, list[str]]) -> list[str]:
    cols: list[str] = []
    for group_cols in config_context.values():
        cols.extend(group_cols)
    return list(dict.fromkeys(cols))
