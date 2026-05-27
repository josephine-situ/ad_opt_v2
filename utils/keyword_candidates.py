"""Build per-segment keyword-set candidates K_s."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.campaign_features import (
    COURSE_ANCHORS,
    _pairwise_mean_distance,
    add_segment_column,
    data_paths,
    load_campaign_summary,
    load_keyword_sets,
    load_or_build_embeddings,
)
from utils.keyword_allowlist import (
    filter_keyword_list,
    filter_keyword_sets_dataframe,
    load_enrollment_keyword_allowlist,
)

MATCH_TYPE_COLS = {
    "Broad": "broad_keywords",
    "Phrase": "phrase_keywords",
    "Exact": "exact_keywords",
}


def _parse_segment_match_types(match_types: str) -> list[str]:
    return [m.strip().title() for m in str(match_types).replace(";", " ").split() if m.strip()]


def _keywords_from_panel(
    kw_day: pd.DataFrame,
    segment_row: pd.Series,
    *,
    top_n: int = 30,
    volume_col: str = "clicks",
    allowed_keywords: set[str] | None = None,
) -> list[str]:
    """Top keywords by volume and volume/cost efficiency for the segment panel slice."""
    if volume_col not in kw_day.columns:
        return []

    region = segment_row["region"]
    allowed = set(_parse_segment_match_types(segment_row["match_types"]))
    sub = kw_day[kw_day["region"] == region].copy()
    if allowed:
        sub = sub[sub["match_type"].isin(allowed)]
    if allowed_keywords:
        sub = sub[sub["keyword"].astype(str).str.lower().str.strip().isin(allowed_keywords)]
    if sub.empty:
        return []

    agg = (
        sub.groupby("keyword")
        .agg(volume=(volume_col, "sum"), cost=("cost", "sum"))
        .reset_index()
    )
    agg = agg[agg["volume"] > 0]
    if agg.empty:
        return []

    agg["efficiency"] = agg["volume"] / agg["cost"].clip(lower=0.01)
    top_vol = set(agg.nlargest(top_n, "volume")["keyword"].tolist())
    top_eff = set(agg.nlargest(top_n, "efficiency")["keyword"].tolist())
    return sorted(top_vol | top_eff)


def _assign_keywords_by_match_type(
    kw_day: pd.DataFrame,
    keywords: list[str],
    segment_row: pd.Series,
    *,
    rank_col: str = "clicks",
) -> dict[str, str]:
    """Split a flat keyword list into broad/phrase/exact columns using panel rank_col."""
    region = segment_row["region"]
    allowed = _parse_segment_match_types(segment_row["match_types"])
    kw_lower = {k.lower(): k for k in keywords}

    sub = kw_day[kw_day["region"] == region]
    if allowed:
        sub = sub[sub["match_type"].isin(allowed)]

    by_mt: dict[str, list[str]] = {mt: [] for mt in allowed}
    if not sub.empty and kw_lower and rank_col in sub.columns:
        agg = (
            sub[sub["keyword"].str.lower().isin(kw_lower.keys())]
            .groupby(["keyword", "match_type"], as_index=False)
            .agg(rank_metric=(rank_col, "sum"))
        )
        for kw, grp in agg.groupby("keyword"):
            best_mt = grp.loc[grp["rank_metric"].idxmax(), "match_type"]
            canon = kw_lower.get(str(kw).lower(), str(kw))
            if best_mt in by_mt:
                by_mt[best_mt].append(canon)

    seen = {k.lower() for ks in by_mt.values() for k in ks}
    fallback = allowed[0] if allowed else "Broad"
    for kw in keywords:
        if kw.lower() not in seen:
            by_mt.setdefault(fallback, []).append(kw)

    return {
        MATCH_TYPE_COLS[mt]: "; ".join(sorted(set(by_mt.get(mt, []))))
        for mt in ("Broad", "Phrase", "Exact")
    }


def _anchor_matrix(emb_map: dict[str, np.ndarray]) -> np.ndarray:
    vecs = []
    for anchor in COURSE_ANCHORS:
        key = anchor.lower().strip()
        if key not in emb_map:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            emb_map[key] = model.encode([anchor], normalize_embeddings=True)[0]
        vecs.append(emb_map[key])
    return np.stack(vecs)


def _top_keywords_by_course_sim(
    pool: list[str],
    emb_map: dict[str, np.ndarray],
    anchor_matrix: np.ndarray,
    *,
    set_size: int,
) -> list[str]:
    """Select keywords with highest per-keyword max-anchor similarity (EDA: embed_course_sim_mean)."""
    scored: list[tuple[str, float]] = []
    for kw in {k.lower() for k in pool}:
        if kw not in emb_map:
            continue
        vec = emb_map[kw]
        scored.append((kw, float((anchor_matrix @ vec).max())))
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [kw for kw, _ in scored[: max(1, min(set_size, len(scored)))]]


def _set_vectors(keywords: list[str], emb_map: dict[str, np.ndarray]) -> np.ndarray | None:
    vecs = [emb_map[k] for k in keywords if k in emb_map]
    if not vecs:
        return None
    return np.stack(vecs)


def _set_course_sim_mean(
    keywords: list[str],
    emb_map: dict[str, np.ndarray],
    anchor_matrix: np.ndarray,
) -> float:
    vecs = _set_vectors(keywords, emb_map)
    if vecs is None:
        return float("-inf")
    sims = vecs @ anchor_matrix.T
    return float(sims.max(axis=1).mean())


def _set_dispersion(keywords: list[str], emb_map: dict[str, np.ndarray]) -> float:
    vecs = _set_vectors(keywords, emb_map)
    if vecs is None or len(vecs) < 2:
        return float("-inf")
    return _pairwise_mean_distance(vecs)


def _greedy_select_by_set_score(
    pool: list[str],
    emb_map: dict[str, np.ndarray],
    anchor_matrix: np.ndarray,
    *,
    set_size: int,
    score_fn,
) -> list[str]:
    eligible = sorted({k.lower() for k in pool if k.lower() in emb_map})
    if not eligible:
        return []
    set_size = max(1, min(set_size, len(eligible)))
    selected = [max(eligible, key=lambda k: score_fn([k]))]
    while len(selected) < set_size:
        best_kw = None
        best_score = float("-inf")
        for kw in eligible:
            if kw in selected:
                continue
            score = score_fn(selected + [kw])
            if score > best_score:
                best_score = score
                best_kw = kw
        if best_kw is None:
            break
        selected.append(best_kw)
    return selected


def _top_keywords_by_dispersion(
    pool: list[str],
    emb_map: dict[str, np.ndarray],
    anchor_matrix: np.ndarray,
    *,
    set_size: int,
) -> list[str]:
    """Greedy set maximizing embed_dispersion (mean distance to centroid)."""
    return _greedy_select_by_set_score(
        pool,
        emb_map,
        anchor_matrix,
        set_size=set_size,
        score_fn=lambda kws: _set_dispersion(kws, emb_map),
    )


def _top_keywords_by_composite(
    pool: list[str],
    emb_map: dict[str, np.ndarray],
    anchor_matrix: np.ndarray,
    *,
    set_size: int,
) -> list[str]:
    """Greedy set maximizing z(course_sim_mean) + z(dispersion), Model C-style."""
    eligible = sorted({k.lower() for k in pool if k.lower() in emb_map})
    if not eligible:
        return []

    per_kw_cs = [_set_course_sim_mean([kw], emb_map, anchor_matrix) for kw in eligible]
    mu_cs = float(np.mean(per_kw_cs))
    sd_cs = float(np.std(per_kw_cs)) or 1.0

    pair_disps: list[float] = []
    for i, k1 in enumerate(eligible):
        for k2 in eligible[i + 1 :]:
            d = _set_dispersion([k1, k2], emb_map)
            if np.isfinite(d):
                pair_disps.append(d)
    mu_disp = float(np.mean(pair_disps)) if pair_disps else 0.0
    sd_disp = float(np.std(pair_disps)) if pair_disps else 1.0
    if sd_disp == 0.0:
        sd_disp = 1.0

    def composite_score(keywords: list[str]) -> float:
        cs = _set_course_sim_mean(keywords, emb_map, anchor_matrix)
        z_cs = (cs - mu_cs) / sd_cs
        if len(keywords) < 2:
            return z_cs
        disp = _set_dispersion(keywords, emb_map)
        z_disp = (disp - mu_disp) / sd_disp
        return z_cs + z_disp

    return _greedy_select_by_set_score(
        pool,
        emb_map,
        anchor_matrix,
        set_size=set_size,
        score_fn=composite_score,
    )


def _ensure_embeddings(
    course: str,
    summary: pd.DataFrame,
    kw_day: pd.DataFrame,
    *,
    top_n: int,
    pool: list[str],
    emb_map: dict[str, np.ndarray] | None,
    anchors: np.ndarray | None,
    allowed_keywords: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if emb_map is not None and anchors is not None:
        return emb_map, anchors
    paths = data_paths(course)
    all_kw = [k.lower() for k in pool]
    for _, g in summary.groupby("segment", sort=False):
        all_kw.extend(
            k.lower()
            for k in _keywords_from_panel(
                kw_day, g.iloc[0], top_n=top_n, allowed_keywords=allowed_keywords
            )
        )
    all_kw.extend(a.lower() for a in COURSE_ANCHORS)
    cache = paths["cache"] / "keyword_embeddings.parquet"
    emb_map = load_or_build_embeddings(all_kw, cache)
    return emb_map, _anchor_matrix(emb_map)


def _is_distinct_variant(keywords: list[str], pool: list[str], seen: set[frozenset[str]]) -> bool:
    kw_set = frozenset(keywords)
    if kw_set == frozenset(k.lower() for k in pool):
        return False
    if kw_set in seen:
        return False
    seen.add(kw_set)
    return True


def _target_set_size(summary: pd.DataFrame, segment: str, *, top_n: int, fallback: int) -> int:
    seg = summary[summary["segment"] == segment]
    if "num_unique_keywords" in seg.columns and seg["num_unique_keywords"].notna().any():
        size = int(seg["num_unique_keywords"].median())
        return max(5, min(size, top_n))
    return fallback


def _next_synthetic_id(
    segment: str,
    suffix: str,
    *,
    synthetic_prefix: str,
    synth_idx: int,
    existing_ids: set[str],
) -> tuple[str, int]:
    seg_token = segment.replace(" / ", "_")
    while True:
        new_id = f"{synthetic_prefix}_{seg_token}_{suffix}_{synth_idx}"
        if new_id not in existing_ids:
            existing_ids.add(new_id)
            return new_id, synth_idx
        synth_idx += 1


def _append_synthetic_set(
    *,
    synthetic_sets: list[dict],
    synth_cand: list[dict],
    segment: str,
    row: pd.Series,
    new_id: str,
    keywords: list[str],
    source: str,
    kw_day: pd.DataFrame,
    positive_col: str,
    match_type_rank_col: str = "clicks",
) -> None:
    if not keywords:
        return
    pos = "; ".join(sorted(keywords))
    record: dict = {"keyword_set_id": new_id, positive_col: pos}
    if not kw_day.empty:
        record.update(
            _assign_keywords_by_match_type(kw_day, keywords, row, rank_col=match_type_rank_col)
        )
    synthetic_sets.append(record)
    synth_cand.append(
        {
            "segment": segment,
            "region": row["region"],
            "match_types": row["match_types"],
            "keyword_set_id": new_id,
            "source": source,
        }
    )


def build_segment_candidates(
    course: str,
    *,
    top_n: int = 30,
    set_size: int | None = None,
    synthetic_prefix: str = "synthetic",
    allowed_match_types: list[str] | None = None,
    excluded_regions: list[str] | None = None,
    allowed_keywords: set[str] | None = None,
    include_performance_synthetic: bool = True,
    include_semantic_synthetic: bool = True,
    include_dispersion_synthetic: bool = True,
    include_composite_synthetic: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        candidates: segment, keyword_set_id, source
        extended_sets: keyword_set_id, positive_keywords (+ match-type columns when known)

    When ``data/<course>/gkp/*Keywords*Enrollments*.xlsx`` exists, only those keywords
    are kept in historical and synthetic sets; empty sets are dropped from candidates.

    Synthetic sources:
        synthetic_top — union of top-click and top click-efficiency keywords from kw-day-panel
        synthetic_top_conv — union of top all_conv and top conversion-efficiency keywords
        synthetic_semantic — top keywords by per-keyword course-anchor similarity
        synthetic_dispersion — greedy set maximizing embed_dispersion
        synthetic_composite — greedy set maximizing z(course_sim_mean) + z(dispersion)
    """
    summary = load_campaign_summary(course)
    summary = add_segment_column(summary)
    if excluded_regions:
        summary = summary[~summary["region"].isin(excluded_regions)]
    if allowed_match_types:
        summary = summary[summary["match_types"].isin(allowed_match_types)]
    if allowed_keywords is None:
        allowed_keywords = load_enrollment_keyword_allowlist(course)
    keyword_sets = load_keyword_sets(course)
    if allowed_keywords:
        keyword_sets = filter_keyword_sets_dataframe(keyword_sets, allowed_keywords)
    positive_col = "positive_keywords"

    kw_path = Path("data") / course / "processed" / "kw-day-panel.csv"
    kw_day = pd.read_csv(kw_path) if kw_path.exists() else pd.DataFrame()

    allowed_set_ids = set(keyword_sets["keyword_set_id"].astype(str))
    hist_rows = []
    for segment, grp in summary.groupby("segment", sort=False):
        for set_id in grp["keyword_set_id"].dropna().unique():
            if str(set_id) not in allowed_set_ids:
                continue
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

    synthetic_sets: list[dict] = []
    synth_cand: list[dict] = []
    synth_idx = 0
    emb_map: dict[str, np.ndarray] | None = None
    anchors: np.ndarray | None = None
    need_embeddings = include_semantic_synthetic or include_dispersion_synthetic or include_composite_synthetic

    if not kw_day.empty:
        for segment, grp in summary.groupby("segment", sort=False):
            row = grp.iloc[0]
            pool = _keywords_from_panel(
                kw_day, row, top_n=top_n, allowed_keywords=allowed_keywords
            )
            if not pool:
                continue
            if allowed_keywords:
                pool = filter_keyword_list(pool, allowed_keywords)

            target_size = set_size or _target_set_size(summary, segment, top_n=top_n, fallback=top_n)
            seen_variants: set[frozenset[str]] = set()

            if include_performance_synthetic:
                synth_idx += 1
                new_id, synth_idx = _next_synthetic_id(
                    segment, "top", synthetic_prefix=synthetic_prefix, synth_idx=synth_idx, existing_ids=existing_ids
                )
                _append_synthetic_set(
                    synthetic_sets=synthetic_sets,
                    synth_cand=synth_cand,
                    segment=segment,
                    row=row,
                    new_id=new_id,
                    keywords=pool,
                    source="synthetic_top",
                    kw_day=kw_day,
                    positive_col=positive_col,
                    match_type_rank_col="clicks",
                )
                seen_variants.add(frozenset(k.lower() for k in pool))

                pool_conv = _keywords_from_panel(
                    kw_day,
                    row,
                    top_n=top_n,
                    volume_col="all_conv",
                    allowed_keywords=allowed_keywords,
                )
                if pool_conv:
                    synth_idx += 1
                    new_id, synth_idx = _next_synthetic_id(
                        segment,
                        "top_conv",
                        synthetic_prefix=synthetic_prefix,
                        synth_idx=synth_idx,
                        existing_ids=existing_ids,
                    )
                    _append_synthetic_set(
                        synthetic_sets=synthetic_sets,
                        synth_cand=synth_cand,
                        segment=segment,
                        row=row,
                        new_id=new_id,
                        keywords=pool_conv,
                        source="synthetic_top_conv",
                        kw_day=kw_day,
                        positive_col=positive_col,
                        match_type_rank_col="all_conv",
                    )
                    seen_variants.add(frozenset(k.lower() for k in pool_conv))

            if need_embeddings:
                emb_map, anchors = _ensure_embeddings(
                    course,
                    summary,
                    kw_day,
                    top_n=top_n,
                    pool=pool,
                    emb_map=emb_map,
                    anchors=anchors,
                    allowed_keywords=allowed_keywords,
                )
                pool_lower = [k.lower() for k in pool]

                semantic_variants: list[tuple[str, str, list[str]]] = []
                if include_semantic_synthetic:
                    semantic_variants.append(
                        (
                            "semantic",
                            "synthetic_semantic",
                            _top_keywords_by_course_sim(pool_lower, emb_map, anchors, set_size=target_size),
                        )
                    )
                if include_dispersion_synthetic:
                    semantic_variants.append(
                        (
                            "dispersion",
                            "synthetic_dispersion",
                            _top_keywords_by_dispersion(pool_lower, emb_map, anchors, set_size=target_size),
                        )
                    )
                if include_composite_synthetic:
                    semantic_variants.append(
                        (
                            "composite",
                            "synthetic_composite",
                            _top_keywords_by_composite(pool_lower, emb_map, anchors, set_size=target_size),
                        )
                    )

                for suffix, source, keywords in semantic_variants:
                    if not _is_distinct_variant(keywords, pool, seen_variants):
                        continue
                    synth_idx += 1
                    new_id, synth_idx = _next_synthetic_id(
                        segment,
                        suffix,
                        synthetic_prefix=synthetic_prefix,
                        synth_idx=synth_idx,
                        existing_ids=existing_ids,
                    )
                    _append_synthetic_set(
                        synthetic_sets=synthetic_sets,
                        synth_cand=synth_cand,
                        segment=segment,
                        row=row,
                        new_id=new_id,
                        keywords=keywords,
                        source=source,
                        kw_day=kw_day,
                        positive_col=positive_col,
                    )

    synth_df = pd.DataFrame(synth_cand)
    candidates = pd.concat([hist_df, synth_df], ignore_index=True)
    candidates = candidates.drop_duplicates(subset=["segment", "keyword_set_id"])

    extended = keyword_sets.copy()
    if synthetic_sets:
        extended = pd.concat([extended, pd.DataFrame(synthetic_sets)], ignore_index=True)

    return candidates, extended


def ensure_segment_keyword_candidates(
    course: str,
    *,
    allowed_match_types: list[str] | None = None,
    excluded_regions: list[str] | None = None,
) -> Path:
    """Write segment-keyword-candidates and extended sets when missing or allowlist is newer."""
    from utils.keyword_allowlist import should_refresh_keyword_candidates

    processed = Path("data") / course / "processed"
    cand_path = processed / "segment-keyword-candidates.csv"
    ext_path = processed / "campaign-keyword-sets-extended.csv"
    if not should_refresh_keyword_candidates(course, cand_path):
        return cand_path

    candidates, extended = build_segment_candidates(
        course,
        allowed_match_types=allowed_match_types,
        excluded_regions=excluded_regions,
    )
    processed.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(cand_path, index=False)
    extended.to_csv(ext_path, index=False)
    return cand_path
