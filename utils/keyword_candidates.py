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
    clean_keyword_text,
    enrollment_allowlist_keywords,
    filter_keyword_list,
    filter_keyword_sets_dataframe,
    load_enrollment_keyword_allowlist,
    load_enrollment_keyword_allowlist_ordered,
    normalize_keyword,
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
    require_positive_volume: bool = False,
) -> list[str]:
    """Top keywords by volume and volume/cost efficiency for the segment panel slice.

    When ``allowed_keywords`` is set, only those keywords are ranked (allowlist-first).
    With ``require_positive_volume=True``, keywords with zero ``volume_col`` are dropped;
    ``build_segment_candidates`` may pad to ``top_n`` from the ranked enrollment allowlist.
    """
    if volume_col not in kw_day.columns:
        return []

    region = segment_row["region"]
    allowed = set(_parse_segment_match_types(segment_row["match_types"]))
    sub = kw_day[kw_day["region"] == region].copy()
    if allowed:
        sub = sub[sub["match_type"].isin(allowed)]
    if allowed_keywords:
        sub["_kw_key"] = sub["keyword"].astype(str).map(normalize_keyword)
        sub = sub[sub["_kw_key"].isin(allowed_keywords)]
    if sub.empty:
        return []

    key_col = "_kw_key" if allowed_keywords else "keyword"
    if key_col == "keyword":
        sub = sub.copy()
        sub["_kw_key"] = sub["keyword"].astype(str)

    agg = (
        sub.groupby("_kw_key", as_index=False)
        .agg(
            volume=(volume_col, "sum"),
            cost=("cost", "sum"),
            keyword=("keyword", "first"),
        )
    )
    if require_positive_volume:
        agg = agg[agg["volume"] > 0]
    if agg.empty:
        return []

    agg["efficiency"] = agg["volume"] / agg["cost"].clip(lower=0.01)
    top_vol = {
        clean_keyword_text(k)
        for k in agg.nlargest(top_n, "volume")["keyword"].astype(str).tolist()
        if clean_keyword_text(k)
    }
    top_eff = {
        clean_keyword_text(k)
        for k in agg.nlargest(top_n, "efficiency")["keyword"].astype(str).tolist()
        if clean_keyword_text(k)
    }
    return sorted(top_vol | top_eff)


def _fill_pool_to_top_n(
    pool: list[str],
    *,
    top_n: int,
    allowlist_ranked: list[str],
    segment_allowlist: list[str],
) -> list[str]:
    """Append top allowlisted keywords (by enrollment priority) until ``top_n`` keywords."""
    if len(pool) >= top_n:
        return pool[:top_n]

    seen = {normalize_keyword(k) for k in pool}
    canonical = {normalize_keyword(k): k for k in segment_allowlist}
    out = list(pool)
    for key in allowlist_ranked:
        if key in seen:
            continue
        out.append(canonical.get(key, key))
        seen.add(key)
        if len(out) >= top_n:
            break
    return out


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
    kw_by_key = {normalize_keyword(k): clean_keyword_text(k) for k in keywords}

    sub = kw_day[kw_day["region"] == region]
    if allowed:
        sub = sub[sub["match_type"].isin(allowed)]

    by_mt: dict[str, list[str]] = {mt: [] for mt in allowed}
    if not sub.empty and kw_by_key and rank_col in sub.columns:
        sub = sub.copy()
        sub["_kw_key"] = sub["keyword"].astype(str).map(normalize_keyword)
        agg = (
            sub[sub["_kw_key"].isin(kw_by_key.keys())]
            .groupby(["_kw_key", "match_type"], as_index=False)
            .agg(rank_metric=(rank_col, "sum"))
        )
        for kw_key, grp in agg.groupby("_kw_key"):
            best_mt = grp.loc[grp["rank_metric"].idxmax(), "match_type"]
            canon = kw_by_key.get(str(kw_key), clean_keyword_text(str(kw_key)))
            if best_mt in by_mt:
                by_mt[best_mt].append(canon)

    seen = {normalize_keyword(k) for ks in by_mt.values() for k in ks}
    fallback = allowed[0] if allowed else "Broad"
    for kw in keywords:
        if normalize_keyword(kw) not in seen:
            by_mt.setdefault(fallback, []).append(clean_keyword_text(kw))

    return {
        MATCH_TYPE_COLS[mt]: "; ".join(sorted(dict.fromkeys(by_mt.get(mt, []))))
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
    keywords = [clean_keyword_text(k) for k in keywords if clean_keyword_text(k)]
    if not keywords:
        return
    pos = "; ".join(sorted(dict.fromkeys(keywords)))
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


def _top_n_labels(base_suffix: str, base_source: str, n: int, *, multi_top_n: bool) -> tuple[str, str]:
    if multi_top_n:
        return f"{base_suffix}_n{n}", f"{base_source}_n{n}"
    return base_suffix, base_source


def build_segment_candidates(
    course: str,
    *,
    top_n: int = 30,
    top_n_values: list[int] | None = None,
    set_size: int | None = None,
    synthetic_prefix: str = "synthetic",
    allowed_match_types: list[str] | None = None,
    excluded_regions: list[str] | None = None,
    allowed_keywords: set[str] | None = None,
    include_top_conv_synthetic: bool = True,
    include_allowlist_synthetic: bool = True,
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
        synthetic_top_conv — union of top all_conv and top conversion-efficiency keywords from kw-day-panel
        synthetic_top_conv_n{N} — same, when ``top_n_values`` lists multiple N (e.g. 10, 20, 40)
        synthetic_allowlist — full enrollment allowlist (when ``*Keywords*Enrollments*.xlsx`` exists)
        synthetic_allowlist_n{N} — first N allowlist keywords by enrollment priority (with ``top_n_values``)
        synthetic_semantic — top keywords by per-keyword course-anchor similarity
        synthetic_dispersion — greedy set maximizing embed_dispersion
        synthetic_composite — greedy set maximizing z(course_sim_mean) + z(dispersion)

    Pass ``top_n_values=[10, 20, 40]`` to emit separate performance/embedding sets per cap.
    """
    summary = load_campaign_summary(course)
    summary = add_segment_column(summary)
    if excluded_regions:
        summary = summary[~summary["region"].isin(excluded_regions)]
    if allowed_match_types:
        summary = summary[summary["match_types"].isin(allowed_match_types)]
    if allowed_keywords is None:
        allowed_keywords = load_enrollment_keyword_allowlist(course)
    allowlist_ordered = load_enrollment_keyword_allowlist_ordered(course)
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

    top_n_list = sorted({int(n) for n in (top_n_values or [top_n]) if int(n) > 0})
    multi_top_n = len(top_n_list) > 1
    rank_col = "all_conv" if not kw_day.empty and "all_conv" in kw_day.columns else "clicks"

    if not kw_day.empty:
        for segment, grp in summary.groupby("segment", sort=False):
            row = grp.iloc[0]
            allowlist_pool: list[str] = []
            if allowed_keywords:
                allowlist_pool = enrollment_allowlist_keywords(
                    allowed_keywords,
                    kw_day,
                    row,
                    allowlist_order=allowlist_ordered,
                )

            segment_seen: set[frozenset[str]] = set()
            any_pool = bool(allowlist_pool)

            if include_allowlist_synthetic and allowlist_pool and not multi_top_n:
                allowlist_key = frozenset(k.lower() for k in allowlist_pool)
                if allowlist_key not in segment_seen:
                    segment_seen.add(allowlist_key)
                    synth_idx += 1
                    new_id, synth_idx = _next_synthetic_id(
                        segment,
                        "allowlist",
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
                        keywords=allowlist_pool,
                        source="synthetic_allowlist",
                        kw_day=kw_day,
                        positive_col=positive_col,
                        match_type_rank_col=rank_col,
                    )

            for n in top_n_list:
                pool = _keywords_from_panel(
                    kw_day,
                    row,
                    top_n=n,
                    volume_col="all_conv",
                    allowed_keywords=allowed_keywords,
                    require_positive_volume=True,
                )
                if allowed_keywords and allowlist_ordered:
                    pool = _fill_pool_to_top_n(
                        pool,
                        top_n=n,
                        allowlist_ranked=allowlist_ordered,
                        segment_allowlist=allowlist_pool,
                    )
                if not pool:
                    pool_empty = True
                else:
                    pool_empty = False
                    any_pool = True

                if include_allowlist_synthetic and allowlist_pool and multi_top_n:
                    capped_allowlist = allowlist_pool[:n]
                    if capped_allowlist:
                        any_pool = True
                        allow_suffix, allow_source = _top_n_labels(
                            "allowlist", "synthetic_allowlist", n, multi_top_n=True
                        )
                        allow_key = frozenset(k.lower() for k in capped_allowlist)
                        if allow_key not in segment_seen:
                            segment_seen.add(allow_key)
                            synth_idx += 1
                            new_id, synth_idx = _next_synthetic_id(
                                segment,
                                allow_suffix,
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
                                keywords=capped_allowlist,
                                source=allow_source,
                                kw_day=kw_day,
                                positive_col=positive_col,
                                match_type_rank_col=rank_col,
                            )

                if pool_empty:
                    continue

                target_size = set_size or _target_set_size(summary, segment, top_n=n, fallback=n)
                seen_variants: set[frozenset[str]] = set()
                conv_suffix, conv_source = _top_n_labels(
                    "top_conv", "synthetic_top_conv", n, multi_top_n=multi_top_n
                )

                if include_top_conv_synthetic:
                    synth_idx += 1
                    new_id, synth_idx = _next_synthetic_id(
                        segment,
                        conv_suffix,
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
                        keywords=pool,
                        source=conv_source,
                        kw_day=kw_day,
                        positive_col=positive_col,
                        match_type_rank_col="all_conv",
                    )
                    seen_variants.add(frozenset(k.lower() for k in pool))

                if need_embeddings:
                    emb_map, anchors = _ensure_embeddings(
                        course,
                        summary,
                        kw_day,
                        top_n=n,
                        pool=pool,
                        emb_map=emb_map,
                        anchors=anchors,
                        allowed_keywords=allowed_keywords,
                    )
                    pool_lower = [k.lower() for k in pool]

                    semantic_variants: list[tuple[str, str, list[str]]] = []
                    if include_semantic_synthetic:
                        sem_suffix, sem_source = _top_n_labels(
                            "semantic", "synthetic_semantic", n, multi_top_n=multi_top_n
                        )
                        semantic_variants.append(
                            (
                                sem_suffix,
                                sem_source,
                                _top_keywords_by_course_sim(
                                    pool_lower, emb_map, anchors, set_size=target_size
                                ),
                            )
                        )
                    if include_dispersion_synthetic:
                        disp_suffix, disp_source = _top_n_labels(
                            "dispersion", "synthetic_dispersion", n, multi_top_n=multi_top_n
                        )
                        semantic_variants.append(
                            (
                                disp_suffix,
                                disp_source,
                                _top_keywords_by_dispersion(
                                    pool_lower, emb_map, anchors, set_size=target_size
                                ),
                            )
                        )
                    if include_composite_synthetic:
                        comp_suffix, comp_source = _top_n_labels(
                            "composite", "synthetic_composite", n, multi_top_n=multi_top_n
                        )
                        semantic_variants.append(
                            (
                                comp_suffix,
                                comp_source,
                                _top_keywords_by_composite(
                                    pool_lower, emb_map, anchors, set_size=target_size
                                ),
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

            if not any_pool and not allowlist_pool:
                continue

    synth_df = pd.DataFrame(synth_cand)
    candidates = pd.concat([hist_df, synth_df], ignore_index=True)
    candidates = candidates.drop_duplicates(subset=["segment", "keyword_set_id"])

    extended = keyword_sets.copy()
    if synthetic_sets:
        extended = pd.concat([extended, pd.DataFrame(synthetic_sets)], ignore_index=True)

    return candidates, extended


def write_segment_keyword_candidate_files(
    course: str,
    candidates: pd.DataFrame,
    extended: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    """
    Persist candidate tables and refresh ``keyword-sets-display`` for candidate sets only.

    Returns ``(candidates_path, extended_path, display_dir)``.
    """
    from utils.keyword_sets_display import export_keyword_sets_display

    processed = Path("data") / course / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    cand_path = processed / "segment-keyword-candidates.csv"
    ext_path = processed / "campaign-keyword-sets-extended.csv"
    candidates.to_csv(cand_path, index=False)
    extended.to_csv(ext_path, index=False)
    display_dir = export_keyword_sets_display(
        course,
        keyword_set_ids=candidates["keyword_set_id"].astype(str).tolist(),
        segment_plan=candidates,
    )
    return cand_path, ext_path, display_dir


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
    write_segment_keyword_candidate_files(course, candidates, extended)
    return cand_path
