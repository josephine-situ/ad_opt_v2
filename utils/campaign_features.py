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


def load_keyword_sets_for_features(course: str) -> pd.DataFrame:
    """Historical keyword sets plus synthetic rows from campaign-keyword-sets-extended.csv."""
    base = load_keyword_sets(course)
    ext_path = data_paths(course)["processed"] / "campaign-keyword-sets-extended.csv"
    if not ext_path.exists():
        return base
    extended, _ = resolve_positive_keyword_column(pd.read_csv(ext_path))
    new_ids = set(extended["keyword_set_id"].astype(str)) - set(base["keyword_set_id"].astype(str))
    if not new_ids:
        return base
    extra = extended[extended["keyword_set_id"].astype(str).isin(new_ids)]
    return pd.concat([base, extra], ignore_index=True)


def add_segment_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["segment"] = out["region"].astype(str) + " / " + out["match_types"].astype(str)
    return out


def parse_match_types(match_types: object) -> set[str]:
    """Parse ``Broad; Phrase; Exact`` style match-type strings."""
    if pd.isna(match_types) or not str(match_types).strip():
        return set()
    return {m.strip().title() for m in str(match_types).replace(";", " ").split() if m.strip()}


TREE_SEGMENT_FEATURE_COLS = ["region", "has_broad", "has_phrase", "has_exact"]


def add_segment_match_type_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose ``segment`` into region + multi-hot match-type flags for tree models.

    ``has_broad``, ``has_phrase``, and ``has_exact`` are 0/1 indicators (one column per type).
    """
    out = df.copy()
    if "match_types" not in out.columns and "segment" in out.columns:
        parts = out["segment"].astype(str).str.split(" / ", n=1, expand=True)
        out["region"] = parts[0]
        out["match_types"] = parts[1]
    if "region" not in out.columns and "segment" in out.columns:
        out["region"] = out["segment"].astype(str).str.split(" / ", n=1).str[0]
    parsed = out["match_types"].map(parse_match_types)
    out["has_broad"] = parsed.map(lambda s: int("Broad" in s))
    out["has_phrase"] = parsed.map(lambda s: int("Phrase" in s))
    out["has_exact"] = parsed.map(lambda s: int("Exact" in s))
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


def add_conversion_scaled_clicks_target(
    panel: pd.DataFrame,
    *,
    target_col: str = "conv_scaled_clicks",
) -> pd.DataFrame:
    """
    Add a clicks target scaled by conversion rate per (region, match_types).

    The segment scale is:
      sum(all_conv) / sum(clicks)
    and target = clicks * segment_scale (with global-rate fallback on zero-click segments).
    """
    out = panel.copy()
    if "clicks" not in out.columns:
        out["clicks"] = 0.0
    out["clicks"] = pd.to_numeric(out["clicks"], errors="coerce").fillna(0.0)

    if "all_conv" not in out.columns:
        out[target_col] = out["clicks"].astype(float)
        return out

    out["all_conv"] = pd.to_numeric(out["all_conv"], errors="coerce").fillna(0.0)
    total_clicks = float(out["clicks"].sum())
    total_conv = float(out["all_conv"].sum())
    global_rate = total_conv / total_clicks if total_clicks > 0 else 0.0

    seg = (
        out.groupby(["region", "match_types"], as_index=False)
        .agg(seg_clicks=("clicks", "sum"), seg_conv=("all_conv", "sum"))
    )
    seg["conv_per_click"] = np.where(
        seg["seg_clicks"] > 0.0,
        seg["seg_conv"] / seg["seg_clicks"],
        global_rate,
    )
    out = out.merge(seg[["region", "match_types", "conv_per_click"]], on=["region", "match_types"], how="left")
    out["conv_per_click"] = out["conv_per_click"].fillna(global_rate)
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
        if "all_conv" in panel.columns:
            panel["all_conv"] = panel["all_conv"].fillna(0.0)
        elif include_all_conv_from_summary:
            panel["all_conv"] = np.nan
    elif target_col == "conv_scaled_clicks":
        panel = add_conversion_scaled_clicks_target(panel, target_col=target_col)

    if "clicks" not in panel.columns:
        panel["clicks"] = 0

    return panel.sort_values(["date", "segment"]).reset_index(drop=True)


def get_context_feature_columns(config_context: dict[str, list[str]]) -> list[str]:
    cols: list[str] = []
    for group_cols in config_context.values():
        cols.extend(group_cols)
    return list(dict.fromkeys(cols))
