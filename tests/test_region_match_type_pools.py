"""Region + match-type panel pools span campaign configurations."""

from __future__ import annotations

import pandas as pd

from utils.keyword_candidates import _build_segment_panel_maps


def test_exact_pool_includes_keywords_from_other_us_campaign_configs():
    kw_day = pd.DataFrame(
        [
            {"region": "USA", "keyword": "only exact cfg", "match_type": "Exact"},
            {"region": "USA", "keyword": "from phrase exact cfg", "match_type": "Exact"},
            {"region": "USA", "keyword": "phrase only", "match_type": "Phrase"},
        ]
    )
    row = pd.Series({"region": "USA", "match_types": "Exact"})
    panel_by_mt, _, keys_union = _build_segment_panel_maps(kw_day, row, None)

    assert panel_by_mt["Exact"] == {"only exact cfg", "from phrase exact cfg"}
    assert "phrase only" not in panel_by_mt["Exact"]
    assert keys_union == panel_by_mt["Exact"]
