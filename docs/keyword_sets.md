# Keyword sets and segment candidates

This document describes how keyword sets are defined, how per-segment candidate pools are built, and how synthetic sets differ from historical ones. For MILP/backtest wiring, see [`campaign_opt/README.md`](../campaign_opt/README.md).

## Concepts

| Term | Meaning |
|------|---------|
| **Keyword set** | A named list of positive keywords (`keyword_set_id`), with `broad_keywords`, `phrase_keywords`, and `exact_keywords` columns. |
| **Segment** | `(region, match_types)` from campaign summary — e.g. `USA / Phrase; Exact`. |
| **Allowlist∩region (per match type)** | Enrollment-approved keywords in `kw-day-panel.csv` for the segment’s **region** and that **match type**, from any campaign config in the region (e.g. USA Exact includes Exact rows from USA Exact, USA Phrase; Exact, USA Broad; Phrase; Exact). |
| **`positive_keywords`** | Sorted union across match-type columns. |

Build logs use the prefix `[keyword_candidates]` and report intersection counts, ranked/selected counts, and allowlist padding.

## Historical keyword sets

**Source:** `scripts/parse_change_history_html.py` → `campaign-keyword-sets.csv`.

Parsed per match type from change history; overlap across columns is allowed. **Candidates:** every `keyword_set_id` seen in that segment (allowlist-filtered when the enrollments file exists).

## Synthetic keyword sets

**Source:** `build_segment_candidates()` → `campaign-keyword-sets-extended.csv`, `segment-keyword-candidates.csv`.

**Script:** `scripts/build_keyword_candidates.py`

### Shared rule (all synthetics)

When an enrollment allowlist exists:

1. **Base pool** = allowlist ∩ keywords in the panel for **region + match type** (`_build_segment_panel_maps`), not limited to one campaign `match_types` string.
2. **Rank / select** from that intersection (rules differ by variant).
3. If fewer than `N` keywords, **pad** from the full enrollment allowlist in enrollment priority order (may add keywords not in the segment panel).

Without an allowlist file, only panel keywords in the segment are used (no enrollment padding).

### Variants

| Source | Rank / select | Match-type columns |
|--------|----------------|-------------------|
| `synthetic_top_conv` | Per match type: rank each allowlist∩region keyword 1..K on `all_conv` volume and on `all_conv`/cost; **sum ranks**, take top `N` (1 = best on each metric) | Different per type; pad each column to `N` |
| `synthetic_allowlist` | Per match type: enrollment order among allowlist∩region keys seen under that type | Different per type; pad each column to `N` |
| `synthetic_semantic` / `dispersion` | Per match type: greedy embedding objective on allowlist∩region pool; top `N` | Different per type; pad to `N` from allowlist |
| `synthetic_composite` | Per match type: rank 1..K on course similarity and mean embedding distance to pool; **sum ranks**, top `N` (same rule as top_conv) | Different per type; pad to `N` from allowlist |

Caps: `--top-n-values 10,20,40` → separate sets per `N`.

### Logging

Example lines:

```text
[keyword_candidates] USA / Broad: 42 keywords in allowlist∩region panel (course allowlist size 120)
[keyword_candidates] USA / Broad | synthetic_top_conv_n20 | match_type=Broad: 8 allowlist∩region panel, 5 after rank/select, padded 15 from enrollment allowlist → 20/20
```

## Outputs

| File | Role |
|------|------|
| `campaign-keyword-sets-extended.csv` | Historical + synthetic sets |
| `segment-keyword-candidates.csv` | segment → `keyword_set_id`, `source` |
| `keyword-set-features.csv` | Features on `positive_keywords` union |

## Commands

```powershell
uv run python scripts/build_keyword_candidates.py --course sys_think --top-n-values 10,20,40 --verify
uv run python scripts/build_gkp_set_features.py --course sys_think
```
