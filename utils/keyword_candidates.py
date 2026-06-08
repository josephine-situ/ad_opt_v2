"""Build per-segment keyword-set candidates K_s.

See ``docs/keyword_sets.md`` for historical vs synthetic construction and match-type columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.paths import data_path
from utils.campaign_features import (
    COURSE_ANCHORS,
    anchor_matrix,
    _pairwise_mean_distance,
    add_segment_column,
    data_paths,
    load_campaign_summary,
    load_keyword_sets,
    load_or_build_embeddings,
)
from utils.keyword_allowlist import (
    allowlist_keys_in_order,
    clean_keyword_text,
    enrollment_allowlist_keywords,
    filter_keyword_list,
    filter_keyword_sets_dataframe,
    load_enrollment_keyword_allowlist,
    load_enrollment_keyword_allowlist_ordered,
    normalize_keyword,
    require_enrollment_allowlist,
)

MATCH_TYPE_COLS = {
    "Broad": "broad_keywords",
    "Phrase": "phrase_keywords",
    "Exact": "exact_keywords",
}

# Shipped caps for synthetic sets (matches two-stage backtest / presentation).
DEFAULT_TOP_N_VALUES: tuple[int, ...] = (10, 20, 40)

_KEYWORD_LIST_COLS = (
    "positive_keywords",
    "broad_keywords",
    "phrase_keywords",
    "exact_keywords",
)


def _parse_segment_match_types(match_types: str) -> list[str]:
    return [m.strip().title() for m in str(match_types).replace(";", " ").split() if m.strip()]


def _segment_allowed_match_types(segment_row: pd.Series) -> list[str]:
    allowed = _parse_segment_match_types(segment_row["match_types"])
    return allowed if allowed else ["Broad", "Phrase", "Exact"]


# Re-export for backward compatibility; canonical definition in keyword_allowlist.
_allowlist_keys_in_order = allowlist_keys_in_order


def _build_segment_panel_maps(
    kw_day: pd.DataFrame,
    segment_row: pd.Series,
    allowlist: set[str] | None,
) -> tuple[dict[str, set[str]], dict[str, str], set[str]]:
    """
    Panel keys per match type for this segment's region.

    Pools are **region + match_type** only (not tied to one campaign ``match_types`` config).
    For ``USA / Exact``, ``panel_keys_by_mt['Exact']`` includes every Exact keyword observed
    in the USA region (e.g. from USA Exact, USA Phrase; Exact, USA Broad; Phrase; Exact).

    ``keys_in_segment`` is the union of those per-type pools (used for logging / padding checks).
    """
    allowed = _segment_allowed_match_types(segment_row)
    region = segment_row["region"]
    panel_keys_by_mt: dict[str, set[str]] = {mt: set() for mt in allowed}
    canonical: dict[str, str] = {}
    keys_in_segment: set[str] = set()

    if kw_day.empty or not pd.notna(region) or "keyword" not in kw_day.columns:
        return panel_keys_by_mt, canonical, keys_in_segment

    for mt in allowed:
        sub = kw_day[(kw_day["region"] == region) & (kw_day["match_type"] == mt)]
        if sub.empty:
            continue
        for kw in sub["keyword"].dropna().astype(str):
            key = normalize_keyword(kw)
            if not key:
                continue
            if allowlist is not None and key not in allowlist:
                continue
            panel_keys_by_mt[mt].add(key)
            keys_in_segment.add(key)
            canonical[key] = clean_keyword_text(kw) or key

    return panel_keys_by_mt, canonical, keys_in_segment


def _segment_intersection_keyword_list(
    keys_in_segment: set[str],
    allowlist_keys_ordered: list[str],
    canonical: dict[str, str],
    *,
    allowlist: set[str] | None,
) -> list[str]:
    """Enrollment order, restricted to allowlist∩region panel (or all segment panel keys)."""
    if allowlist is not None:
        ordered = [k for k in allowlist_keys_ordered if k in keys_in_segment]
    else:
        ordered = sorted(keys_in_segment)
    return [canonical.get(k, k) for k in ordered]


def _intersection_keywords_for_match_type(
    match_type: str,
    panel_keys_by_mt: dict[str, set[str]],
    allowlist_keys_ordered: list[str],
    canonical: dict[str, str],
    *,
    allowlist: set[str] | None,
) -> list[str]:
    """Allowlist∩region panel keywords for one match type (enrollment order)."""
    mt_keys = panel_keys_by_mt.get(match_type, set())
    if allowlist is not None:
        ordered = [k for k in allowlist_keys_ordered if k in mt_keys]
    else:
        ordered = sorted(mt_keys)
    return [canonical.get(k, k) for k in ordered]


def _log_keyword_pool(
    segment: str,
    source: str,
    label: str,
    *,
    intersection_count: int,
    ranked_count: int,
    top_n: int,
    padded_count: int = 0,
) -> None:
    if padded_count > 0:
        print(
            f"[keyword_candidates] {segment} | {source} | {label}: "
            f"{intersection_count} allowlist∩region panel, {ranked_count} after rank/select, "
            f"padded {padded_count} from enrollment allowlist → {ranked_count + padded_count}/{top_n}"
        )
    elif ranked_count >= top_n and ranked_count > intersection_count:
        print(
            f"[keyword_candidates] {segment} | {source} | {label}: "
            f"{intersection_count} allowlist∩region panel, {ranked_count} after rank/select (cap top_n={top_n})"
        )
    else:
        print(
            f"[keyword_candidates] {segment} | {source} | {label}: "
            f"{intersection_count} allowlist∩region panel, {ranked_count} after rank/select (top_n={top_n})"
        )


def _rank_scores(scores: dict[str, float], *, higher_is_better: bool) -> dict[str, int]:
    """Rank 1 = best among ``scores`` (ties get the same rank via min method)."""
    if not scores:
        return {}
    worst = float("-inf") if higher_is_better else float("inf")
    series = pd.Series({k: scores.get(k, worst) for k in scores}, dtype=float)
    ascending = not higher_is_better
    return series.rank(method="min", ascending=ascending).astype(int).to_dict()


def _top_keywords_by_rank_sum(
    keys: list[str],
    metric_a: dict[str, float],
    metric_b: dict[str, float],
    *,
    top_n: int,
    label_by_key: dict[str, str] | None = None,
) -> list[str]:
    """Score each keyword 1..K on both metrics (1 = best), sum ranks, take lowest ``top_n``."""
    unique = list(dict.fromkeys(keys))
    if not unique:
        return []

    rank_a = _rank_scores({k: metric_a.get(k, float("-inf")) for k in unique}, higher_is_better=True)
    rank_b = _rank_scores({k: metric_b.get(k, float("-inf")) for k in unique}, higher_is_better=True)
    combined = sorted(unique, key=lambda k: (rank_a[k] + rank_b[k], k))
    picked = combined[:top_n]
    if label_by_key:
        return [label_by_key.get(k, k) for k in picked]
    return picked


def _top_keywords_for_match_type_slice(
    sub: pd.DataFrame,
    *,
    top_n: int,
    volume_col: str,
    require_positive_volume: bool,
) -> list[str]:
    """Top ``top_n`` by summed rank on volume and conversion efficiency (1 = best on each)."""
    if sub.empty or volume_col not in sub.columns:
        return []
    sub = sub.copy()
    sub["_kw_key"] = sub["keyword"].astype(str).map(normalize_keyword)
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
    keys = agg["_kw_key"].astype(str).tolist()
    volume_scores = dict(zip(agg["_kw_key"].astype(str), agg["volume"]))
    efficiency_scores = dict(zip(agg["_kw_key"].astype(str), agg["efficiency"]))
    labels = {
        str(row["_kw_key"]): clean_keyword_text(str(row["keyword"]))
        for _, row in agg.iterrows()
        if clean_keyword_text(str(row["keyword"]))
    }
    return _top_keywords_by_rank_sum(
        keys,
        volume_scores,
        efficiency_scores,
        top_n=top_n,
        label_by_key=labels,
    )


def _keywords_from_panel_by_match_type(
    kw_day: pd.DataFrame,
    segment_row: pd.Series,
    *,
    top_n: int = 30,
    volume_col: str = "clicks",
    allowed_keywords: set[str] | None,
    panel_keys_by_mt: dict[str, set[str]],
    require_positive_volume: bool = False,
    segment: str = "",
    source: str = "",
    allowlist_keys_ordered: list[str] | None = None,
    enrollment_canonical: list[str] | None = None,
) -> dict[str, list[str]]:
    """Rank top performers per (region, match_type) panel slice; pad to top_n from allowlist."""
    if volume_col not in kw_day.columns:
        return {}

    region = segment_row["region"]
    allowed = _segment_allowed_match_types(segment_row)
    by_mt: dict[str, list[str]] = {mt: [] for mt in allowed}
    pad_keys = allowlist_keys_ordered or []
    pad_canon = {normalize_keyword(k): k for k in (enrollment_canonical or [])}

    for mt in allowed:
        sub = kw_day[(kw_day["region"] == region) & (kw_day["match_type"] == mt)].copy()
        if allowed_keywords:
            sub["_kw_key"] = sub["keyword"].astype(str).map(normalize_keyword)
            sub = sub[sub["_kw_key"].isin(allowed_keywords)]
        ranked = _top_keywords_for_match_type_slice(
            sub,
            top_n=top_n,
            volume_col=volume_col,
            require_positive_volume=require_positive_volume,
        )
        n_ranked = len(ranked)
        padded = 0
        if pad_keys and n_ranked < top_n:
            seen = {normalize_keyword(k) for k in ranked}
            for key in pad_keys:
                if key in seen:
                    continue
                ranked.append(pad_canon.get(key, key))
                seen.add(key)
                padded += 1
                if len(ranked) >= top_n:
                    break
        by_mt[mt] = ranked[:top_n]
        if segment and source:
            mt_ix = len(panel_keys_by_mt.get(mt, set()))
            _log_keyword_pool(
                segment,
                source,
                f"match_type={mt}",
                intersection_count=mt_ix,
                ranked_count=n_ranked,
                top_n=top_n,
                padded_count=padded,
            )
    return by_mt


def _fill_pool_to_top_n(
    pool: list[str],
    *,
    top_n: int,
    allowlist_ranked: list[str],
    enrollment_canonical: list[str],
    segment: str = "",
    source: str = "",
    label: str = "",
    intersection_count: int | None = None,
) -> list[str]:
    """Append enrollment-allowlist keywords (by priority) until ``top_n``."""
    n_ranked = len(pool)
    if n_ranked >= top_n:
        return pool[:top_n]

    seen = {normalize_keyword(k) for k in pool}
    canonical = {normalize_keyword(k): k for k in enrollment_canonical}
    out = list(pool)
    padded = 0
    for key in allowlist_ranked:
        if key in seen:
            continue
        out.append(canonical.get(key, key))
        seen.add(key)
        padded += 1
        if len(out) >= top_n:
            break
    if segment and source and padded > 0:
        _log_keyword_pool(
            segment,
            source,
            label,
            intersection_count=intersection_count if intersection_count is not None else n_ranked,
            ranked_count=n_ranked,
            top_n=top_n,
            padded_count=padded,
        )
    return out


def _fill_pools_by_match_type_to_top_n(
    pools: dict[str, list[str]],
    *,
    top_n: int,
    allowlist_ranked: list[str],
    enrollment_canonical: list[str],
    segment: str = "",
    source: str = "",
    panel_keys_by_mt: dict[str, set[str]] | None = None,
) -> dict[str, list[str]]:
    """Pad each match-type list independently to ``top_n`` from the enrollment allowlist."""
    out: dict[str, list[str]] = {}
    for mt, pool in pools.items():
        mt_ix = len((panel_keys_by_mt or {}).get(mt, set()))
        out[mt] = _fill_pool_to_top_n(
            pool,
            top_n=top_n,
            allowlist_ranked=allowlist_ranked,
            enrollment_canonical=enrollment_canonical,
            segment=segment,
            source=source,
            label=f"match_type={mt}",
            intersection_count=mt_ix,
        )
    return out


def _allowlist_keywords_by_match_type(
    segment_row: pd.Series,
    *,
    top_n: int,
    allowlist_keys_ordered: list[str],
    panel_keys_by_mt: dict[str, set[str]],
    canonical: dict[str, str],
    enrollment_canonical: list[str],
    segment: str = "",
    source: str = "",
) -> dict[str, list[str]]:
    """``top_n`` allowlist∩region panel keywords per match type (enrollment order), then pad."""
    allowed = _segment_allowed_match_types(segment_row)
    by_mt: dict[str, list[str]] = {mt: [] for mt in allowed}
    for key in allowlist_keys_ordered:
        canon = canonical.get(key, key)
        for mt in allowed:
            if len(by_mt[mt]) >= top_n:
                continue
            if key in panel_keys_by_mt.get(mt, set()):
                by_mt[mt].append(canon)

    for mt in allowed:
        n_from_ix = len(by_mt[mt])
        if segment and source and n_from_ix > 0:
            mt_ix = len(panel_keys_by_mt.get(mt, set()))
            print(
                f"[keyword_candidates] {segment} | {source} | match_type={mt}: "
                f"{mt_ix} allowlist∩region panel, {n_from_ix} selected from intersection (top_n={top_n})"
            )

    return _fill_pools_by_match_type_to_top_n(
        by_mt,
        top_n=top_n,
        allowlist_ranked=allowlist_keys_ordered,
        enrollment_canonical=enrollment_canonical,
        segment=segment,
        source=source,
        panel_keys_by_mt=panel_keys_by_mt,
    )


def _union_keywords_from_match_types(by_mt: dict[str, list[str]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for mt in ("Broad", "Phrase", "Exact"):
        for kw in by_mt.get(mt, []):
            key = normalize_keyword(kw)
            if key not in seen:
                seen.add(key)
                out.append(clean_keyword_text(kw))
    return sorted(out)


def _match_type_lists_to_columns(by_mt: dict[str, list[str]], *, positive_col: str) -> dict[str, str]:
    cols = {
        MATCH_TYPE_COLS[mt]: "; ".join(sorted(dict.fromkeys(by_mt.get(mt, []))))
        for mt in ("Broad", "Phrase", "Exact")
    }
    cols[positive_col] = "; ".join(_union_keywords_from_match_types(by_mt))
    return cols


def _frozenset_union(by_mt: dict[str, list[str]]) -> frozenset[str]:
    return frozenset(normalize_keyword(k) for ks in by_mt.values() for k in ks if k)


# Re-export for backward compatibility; canonical definition in campaign_features.
_anchor_matrix = anchor_matrix


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


def _per_keyword_mean_distance_to_pool(
    keyword: str,
    pool: list[str],
    emb_map: dict[str, np.ndarray],
) -> float:
    """Mean embedding distance from ``keyword`` to other keywords in ``pool``."""
    if keyword not in emb_map:
        return float("-inf")
    vec = emb_map[keyword]
    others = [emb_map[k] for k in pool if k != keyword and k in emb_map]
    if not others:
        return 0.0
    dists = [float(np.linalg.norm(vec - other)) for other in others]
    return float(np.mean(dists))


def _top_keywords_by_composite(
    pool: list[str],
    emb_map: dict[str, np.ndarray],
    anchor_matrix: np.ndarray,
    *,
    set_size: int,
) -> list[str]:
    """
    Top keywords by summed rank on course-anchor similarity and mean distance to pool.

    Same selection rule as top_conv: rank 1 = best on each metric among K candidates, sum ranks, take top N.
    """
    eligible = sorted({k.lower() for k in pool if k.lower() in emb_map})
    if not eligible:
        return []

    course_sim = {
        kw: float((anchor_matrix @ emb_map[kw]).max()) for kw in eligible if kw in emb_map
    }
    dispersion = {
        kw: _per_keyword_mean_distance_to_pool(kw, eligible, emb_map) for kw in eligible
    }
    return _top_keywords_by_rank_sum(eligible, course_sim, dispersion, top_n=set_size)


def _ensure_embeddings(
    course: str,
    summary: pd.DataFrame,
    kw_day: pd.DataFrame,
    *,
    rank_pool: list[str],
    emb_map: dict[str, np.ndarray] | None,
    anchors: np.ndarray | None,
    allowed_keywords: set[str] | None,
    allowlist_ordered: list[str] | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if emb_map is not None and anchors is not None:
        return emb_map, anchors
    paths = data_paths(course)
    all_kw = [k.lower() for k in rank_pool]
    for _, g in summary.groupby("segment", sort=False):
        row = g.iloc[0]
        panel_keys_by_mt, canonical, keys_in_segment = _build_segment_panel_maps(
            kw_day, row, allowed_keywords
        )
        ordered = _allowlist_keys_in_order(allowed_keywords, allowlist_ordered)
        for mt in _segment_allowed_match_types(row):
            pool = _intersection_keywords_for_match_type(
                mt,
                panel_keys_by_mt,
                ordered,
                canonical,
                allowlist=allowed_keywords,
            )
            all_kw.extend(k.lower() for k in pool)
    all_kw.extend(a.lower() for a in COURSE_ANCHORS)
    cache = paths["cache"] / "keyword_embeddings.parquet"
    emb_map = load_or_build_embeddings(all_kw, cache)
    return emb_map, _anchor_matrix(emb_map)


def _is_distinct_variant_mt(
    variant_by_mt: dict[str, list[str]],
    pool_by_mt: dict[str, list[str]],
    seen: set[frozenset[str]],
) -> bool:
    kw_set = _frozenset_union(variant_by_mt)
    if kw_set == _frozenset_union(pool_by_mt):
        return False
    if kw_set in seen:
        return False
    seen.add(kw_set)
    return True


def _select_embedding_keywords_for_pool(
    intersection_pool: list[str],
    emb_map: dict[str, np.ndarray],
    anchor_matrix: np.ndarray,
    *,
    top_n: int,
    selector,
) -> list[str]:
    canon: dict[str, str] = {}
    pool_lower: list[str] = []
    for k in intersection_pool:
        kl = normalize_keyword(k)
        if kl in emb_map:
            canon[kl] = clean_keyword_text(k)
            pool_lower.append(kl)
    if not pool_lower:
        return []
    selected_lower = selector(
        pool_lower,
        emb_map,
        anchor_matrix,
        set_size=min(top_n, len(pool_lower)),
    )
    return [canon[kw] for kw in selected_lower]


def _apply_embedding_variant_by_match_type(
    segment_row: pd.Series,
    panel_keys_by_mt: dict[str, set[str]],
    canonical: dict[str, str],
    emb_map: dict[str, np.ndarray],
    anchor_matrix: np.ndarray,
    *,
    top_n: int,
    selector,
    allowlist_keys_ordered: list[str],
    enrollment_canonical: list[str],
    segment: str,
    source: str,
    variant_label: str,
    allowlist: set[str] | None,
) -> dict[str, list[str]]:
    """Per match type: rank allowlist∩region panel for that type, top ``top_n``, pad from allowlist."""
    if not any(panel_keys_by_mt.values()) and not allowlist_keys_ordered:
        return {}

    allowed = _segment_allowed_match_types(segment_row)
    by_mt: dict[str, list[str]] = {}
    for mt in allowed:
        ix_pool = _intersection_keywords_for_match_type(
            mt,
            panel_keys_by_mt,
            allowlist_keys_ordered,
            canonical,
            allowlist=allowlist,
        )
        n_ix = len(ix_pool)
        selected = _select_embedding_keywords_for_pool(
            ix_pool,
            emb_map,
            anchor_matrix,
            top_n=top_n,
            selector=selector,
        )
        n_ranked = len(selected)
        selected = _fill_pool_to_top_n(
            selected,
            top_n=top_n,
            allowlist_ranked=allowlist_keys_ordered,
            enrollment_canonical=enrollment_canonical,
            segment=segment,
            source=source,
            label=f"{variant_label} match_type={mt}",
            intersection_count=n_ix,
        )
        if selected:
            by_mt[mt] = selected
        elif segment and source:
            print(
                f"[keyword_candidates] {segment} | {source} | {variant_label} match_type={mt}: "
                f"0 keywords after rank/select (intersection={n_ix}, top_n={top_n})"
            )
    return by_mt


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
    by_match_type: dict[str, list[str]],
    source: str,
    positive_col: str,
) -> None:
    allowed = _segment_allowed_match_types(row)
    cleaned: dict[str, list[str]] = {
        mt: [clean_keyword_text(k) for k in by_match_type.get(mt, []) if clean_keyword_text(k)]
        for mt in allowed
    }
    if not _union_keywords_from_match_types(cleaned):
        return
    record: dict = {"keyword_set_id": new_id}
    record.update(_match_type_lists_to_columns(cleaned, positive_col=positive_col))
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

    Synthetic sources (ranked **per match type** within segment region):
        synthetic_top_conv — top all_conv + conversion-efficiency keywords per match type from kw-day-panel
        synthetic_top_conv_n{N} — same, when ``top_n_values`` lists multiple N (e.g. 10, 20, 40)
        synthetic_allowlist — enrollment allowlist per match type (panel-observed first, then pad)
        synthetic_allowlist_n{N} — first N per match type (with ``top_n_values``)
        synthetic_semantic / dispersion / composite — top ``N`` per match type from allowlist∩region (for that type) by embedding score; pad from allowlist

    Pass ``top_n_values=[10, 20, 40]`` to emit separate performance/embedding sets per cap.
    """
    summary = load_campaign_summary(course)
    summary = add_segment_column(summary)
    if excluded_regions:
        summary = summary[~summary["region"].isin(excluded_regions)]
    if allowed_match_types:
        summary = summary[summary["match_types"].isin(allowed_match_types)]
    require_enrollment_allowlist(course)
    if allowed_keywords is None:
        allowed_keywords = load_enrollment_keyword_allowlist(course)
    allowlist_ordered = load_enrollment_keyword_allowlist_ordered(course)
    keyword_sets = load_keyword_sets(course)
    if allowed_keywords:
        keyword_sets = filter_keyword_sets_dataframe(keyword_sets, allowed_keywords)
    positive_col = "positive_keywords"

    kw_path = data_path(course, "processed", "kw-day-panel.csv")
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

    if not kw_day.empty:
        for segment, grp in summary.groupby("segment", sort=False):
            row = grp.iloc[0]
            panel_keys_by_mt, canonical, keys_in_segment = _build_segment_panel_maps(
                kw_day, row, allowed_keywords
            )
            allowlist_keys_ordered = _allowlist_keys_in_order(allowed_keywords, allowlist_ordered)
            enrollment_canonical = enrollment_allowlist_keywords(
                allowed_keywords,
                kw_day,
                row,
                allowlist_order=allowlist_ordered,
            )
            intersection_list = _segment_intersection_keyword_list(
                keys_in_segment,
                allowlist_keys_ordered,
                canonical,
                allowlist=allowed_keywords,
            )
            region = row["region"]
            print(
                f"[keyword_candidates] {segment}: {len(keys_in_segment)} keywords in "
                f"allowlist∩region panel across match types (region={region}, "
                f"allowlist size {len(allowed_keywords)}; pools from all campaigns in region)"
            )

            segment_seen: set[frozenset[str]] = set()
            any_pool = bool(keys_in_segment) or bool(allowlist_keys_ordered)

            def _emit_allowlist(by_mt: dict[str, list[str]], suffix: str, source: str) -> None:
                nonlocal synth_idx, any_pool
                key = _frozenset_union(by_mt)
                if not key or key in segment_seen:
                    return
                segment_seen.add(key)
                any_pool = True
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
                    by_match_type=by_mt,
                    source=source,
                    positive_col=positive_col,
                )

            if include_allowlist_synthetic and allowed_keywords and not multi_top_n:
                allow_cap = len(allowlist_keys_ordered) or 1
                by_mt = _allowlist_keywords_by_match_type(
                    row,
                    top_n=allow_cap,
                    allowlist_keys_ordered=allowlist_keys_ordered,
                    panel_keys_by_mt=panel_keys_by_mt,
                    canonical=canonical,
                    enrollment_canonical=enrollment_canonical,
                    segment=segment,
                    source="synthetic_allowlist",
                )
                _emit_allowlist(by_mt, "allowlist", "synthetic_allowlist")

            for n in top_n_list:
                pool_by_mt = _keywords_from_panel_by_match_type(
                    kw_day,
                    row,
                    top_n=n,
                    volume_col="all_conv",
                    allowed_keywords=allowed_keywords,
                    panel_keys_by_mt=panel_keys_by_mt,
                    require_positive_volume=True,
                    segment=segment,
                    source=_top_n_labels("top_conv", "synthetic_top_conv", n, multi_top_n=multi_top_n)[1],
                    allowlist_keys_ordered=allowlist_keys_ordered,
                    enrollment_canonical=enrollment_canonical,
                )
                pool_empty = not _union_keywords_from_match_types(pool_by_mt)
                if not pool_empty:
                    any_pool = True

                if include_allowlist_synthetic and allowed_keywords and multi_top_n:
                    allow_suffix, allow_source = _top_n_labels(
                        "allowlist", "synthetic_allowlist", n, multi_top_n=True
                    )
                    by_mt_allow = _allowlist_keywords_by_match_type(
                        row,
                        top_n=n,
                        allowlist_keys_ordered=allowlist_keys_ordered,
                        panel_keys_by_mt=panel_keys_by_mt,
                        canonical=canonical,
                        enrollment_canonical=enrollment_canonical,
                        segment=segment,
                        source=allow_source,
                    )
                    _emit_allowlist(by_mt_allow, allow_suffix, allow_source)

                if pool_empty and not (need_embeddings and (keys_in_segment or allowlist_keys_ordered)):
                    continue

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
                        by_match_type=pool_by_mt,
                        source=conv_source,
                        positive_col=positive_col,
                    )
                    seen_variants.add(_frozenset_union(pool_by_mt))

                if need_embeddings and (keys_in_segment or allowlist_keys_ordered):
                    emb_map, anchors = _ensure_embeddings(
                        course,
                        summary,
                        kw_day,
                        rank_pool=intersection_list or enrollment_canonical,
                        emb_map=emb_map,
                        anchors=anchors,
                        allowed_keywords=allowed_keywords,
                        allowlist_ordered=allowlist_ordered,
                    )

                    semantic_variants: list[tuple[str, str, dict[str, list[str]]]] = []
                    if include_semantic_synthetic:
                        sem_suffix, sem_source = _top_n_labels(
                            "semantic", "synthetic_semantic", n, multi_top_n=multi_top_n
                        )
                        semantic_variants.append(
                            (
                                sem_suffix,
                                sem_source,
                                _apply_embedding_variant_by_match_type(
                                    row,
                                    panel_keys_by_mt,
                                    canonical,
                                    emb_map,
                                    anchors,
                                    top_n=n,
                                    selector=_top_keywords_by_course_sim,
                                    allowlist_keys_ordered=allowlist_keys_ordered,
                                    enrollment_canonical=enrollment_canonical,
                                    segment=segment,
                                    source=sem_source,
                                    variant_label="semantic",
                                    allowlist=allowed_keywords,
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
                                _apply_embedding_variant_by_match_type(
                                    row,
                                    panel_keys_by_mt,
                                    canonical,
                                    emb_map,
                                    anchors,
                                    top_n=n,
                                    selector=_top_keywords_by_dispersion,
                                    allowlist_keys_ordered=allowlist_keys_ordered,
                                    enrollment_canonical=enrollment_canonical,
                                    segment=segment,
                                    source=disp_source,
                                    variant_label="dispersion",
                                    allowlist=allowed_keywords,
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
                                _apply_embedding_variant_by_match_type(
                                    row,
                                    panel_keys_by_mt,
                                    canonical,
                                    emb_map,
                                    anchors,
                                    top_n=n,
                                    selector=_top_keywords_by_composite,
                                    allowlist_keys_ordered=allowlist_keys_ordered,
                                    enrollment_canonical=enrollment_canonical,
                                    segment=segment,
                                    source=comp_source,
                                    variant_label="composite",
                                    allowlist=allowed_keywords,
                                ),
                            )
                        )

                    for suffix, source, variant_by_mt in semantic_variants:
                        if not _is_distinct_variant_mt(variant_by_mt, pool_by_mt, seen_variants):
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
                            by_match_type=variant_by_mt,
                            source=source,
                            positive_col=positive_col,
                        )

            if not any_pool:
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

    processed = data_path(course, "processed")
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


def verify_segment_keyword_candidates(
    course: str,
    candidates: pd.DataFrame | None = None,
    extended: pd.DataFrame | None = None,
    *,
    top_n_values: tuple[int, ...] = DEFAULT_TOP_N_VALUES,
) -> list[str]:
    """
    Validate ``segment-keyword-candidates.csv`` against backtest expectations.

    Checks allowlist filtering, ``top_conv`` multi-cap sources, and per-segment coverage.
    Returns a list of human-readable issue strings (empty when OK).
    """
    processed = data_path(course, "processed")
    cand_path = processed / "segment-keyword-candidates.csv"
    ext_path = processed / "campaign-keyword-sets-extended.csv"
    if candidates is None:
        if not cand_path.is_file():
            return [f"Missing {cand_path}"]
        candidates = pd.read_csv(cand_path)
    if extended is None:
        if not ext_path.is_file():
            return [f"Missing {ext_path}"]
        extended = pd.read_csv(ext_path)

    issues: list[str] = []
    allowlist = load_enrollment_keyword_allowlist(course)
    segments = sorted(candidates["segment"].dropna().unique())

    for n in top_n_values:
        for base in ("synthetic_top_conv", "synthetic_allowlist"):
            source = f"{base}_n{n}"
            present = set(candidates.loc[candidates["source"] == source, "segment"])
            missing = [s for s in segments if s not in present]
            if missing:
                issues.append(
                    f"Missing {source} for segment(s): {', '.join(missing)}"
                )

    ext_by_id = extended.set_index("keyword_set_id", drop=False)
    for _, row in candidates.iterrows():
        set_id = str(row["keyword_set_id"])
        if set_id not in ext_by_id.index:
            issues.append(f"Candidate {set_id!r} missing from extended sets")
            continue
        ext_row = ext_by_id.loc[set_id]
        if isinstance(ext_row, pd.DataFrame):
            ext_row = ext_row.iloc[0]
        for col in _KEYWORD_LIST_COLS:
            if col not in ext_row or pd.isna(ext_row[col]) or not str(ext_row[col]).strip():
                continue
            for kw in str(ext_row[col]).split(";"):
                kw = kw.strip()
                if kw and normalize_keyword(kw) not in allowlist:
                    issues.append(
                        f"Allowlist violation in {set_id!r} ({col}): {kw!r}"
                    )

    return issues


def ensure_segment_keyword_candidates(
    course: str,
    *,
    allowed_match_types: list[str] | None = None,
    excluded_regions: list[str] | None = None,
    top_n_values: tuple[int, ...] | None = None,
) -> Path:
    """Write segment-keyword-candidates and extended sets when missing or allowlist is newer."""
    from utils.keyword_allowlist import should_refresh_keyword_candidates

    require_enrollment_allowlist(course)
    processed = data_path(course, "processed")
    cand_path = processed / "segment-keyword-candidates.csv"
    if not should_refresh_keyword_candidates(course, cand_path):
        return cand_path

    caps = top_n_values if top_n_values is not None else DEFAULT_TOP_N_VALUES
    candidates, extended = build_segment_candidates(
        course,
        top_n_values=list(caps),
        allowed_match_types=allowed_match_types,
        excluded_regions=excluded_regions,
    )
    write_segment_keyword_candidate_files(course, candidates, extended)
    return cand_path
