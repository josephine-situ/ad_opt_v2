"""Campaign-day feature engineering: panel, calendar, embeddings, GKP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import COURSE_CONFIG
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
    return pd.read_csv(path)


def load_keyword_sets(course: str) -> pd.DataFrame:
    path = data_paths(course)["processed"] / "campaign-keyword-sets.csv"
    return pd.read_csv(path)


def add_segment_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["segment"] = out["region"].astype(str) + " / " + out["match_types"].astype(str)
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
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if "model_name" in cached.columns and cached["model_name"].iloc[0] == model_name:
            return {
                row["keyword"]: np.asarray(row["embedding"], dtype=float)
                for _, row in cached.iterrows()
            }

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    unique = sorted(set(keywords))
    embs = model.encode(unique, normalize_embeddings=True)
    emb_map = {kw: embs[i] for i, kw in enumerate(unique)}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "keyword": list(emb_map.keys()),
            "model_name": model_name,
            "embedding": [v.tolist() for v in emb_map.values()],
        }
    ).to_parquet(cache_path, index=False)
    return emb_map


def keyword_set_semantic_features(
    keyword_sets: pd.DataFrame,
    emb_map: dict[str, np.ndarray],
    *,
    positive_col: str = "positive_keywords",
) -> pd.DataFrame:
    anchor_vecs = []
    for anchor in COURSE_ANCHORS:
        key = anchor.lower().strip()
        if key in emb_map:
            anchor_vecs.append(emb_map[key])
        else:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(EMBED_MODEL_DEFAULT)
            vec = model.encode([anchor], normalize_embeddings=True)[0]
            emb_map[key] = vec
            anchor_vecs.append(vec)

    anchor_matrix = np.stack(anchor_vecs)

    rows = []
    for _, row in keyword_sets.iterrows():
        set_id = row["keyword_set_id"]
        raw = row.get(positive_col, "")
        keywords = (
            [k.strip().lower() for k in str(raw).split(";") if k.strip()]
            if pd.notna(raw)
            else []
        )
        vectors = np.stack([emb_map[k] for k in keywords if k in emb_map]) if keywords else None
        if vectors is None or len(vectors) == 0:
            rows.append(
                {
                    "keyword_set_id": set_id,
                    "n_positive": 0,
                    **{c: np.nan for c in SEMANTIC_FEATURE_COLS},
                }
            )
            continue

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
    keyword_sets = load_keyword_sets(course)
    positive_col = "positive_keywords"
    if positive_col not in keyword_sets.columns:
        for alt in ("keywords_positive", "positive_keyword_list"):
            if alt in keyword_sets.columns:
                positive_col = alt
                break

    all_kw: list[str] = []
    for raw in keyword_sets[positive_col].dropna():
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

    out = sem.merge(gkp_set, on="keyword_set_id", how="left").merge(
        counts, on="keyword_set_id", how="left"
    )
    return out


def attach_keyword_set_to_panel(panel: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Map campaign_version -> keyword_set_id from summary."""
    if "keyword_set_id" not in summary.columns:
        return panel
    version_map = summary[["campaign_version", "keyword_set_id"]].drop_duplicates()
    out = panel.drop(columns=["keyword_set_id"], errors="ignore")
    return out.merge(version_map, on="campaign_version", how="left")


def build_modeling_frame(
    course: str,
    *,
    target_col: str = "all_conv",
    include_all_conv_from_summary: bool = True,
) -> pd.DataFrame:
    """
    Build segment-day modeling dataframe with calendar + set features.
    Expects campaign-day-panel; merges all_conv if conv file exists.
    """
    panel = load_campaign_day_panel(course)
    panel = add_segment_column(panel)
    summary = load_campaign_summary(course)
    panel = attach_keyword_set_to_panel(panel, summary)

    set_feats = build_keyword_set_feature_table(course)
    panel = panel.merge(set_feats, on="keyword_set_id", how="left")
    panel = add_calendar_features(panel, course=course)

    conv_path = data_paths(course)["processed"] / "campaign-day-conv.csv"
    if target_col == "all_conv" and conv_path.exists():
        conv = pd.read_csv(conv_path)
        conv["date"] = pd.to_datetime(conv["date"])
        panel = panel.merge(
            conv[["date", "campaign_version", "all_conv"]],
            on=["date", "campaign_version"],
            how="left",
        )
        panel["all_conv"] = panel["all_conv"].fillna(0.0)
    elif target_col == "all_conv" and include_all_conv_from_summary:
        panel["all_conv"] = np.nan

    if "clicks" not in panel.columns:
        panel["clicks"] = 0

    return panel.sort_values(["date", "segment"]).reset_index(drop=True)


def get_context_feature_columns(config_context: dict[str, list[str]]) -> list[str]:
    cols: list[str] = []
    for group_cols in config_context.values():
        cols.extend(group_cols)
    return list(dict.fromkeys(cols))
