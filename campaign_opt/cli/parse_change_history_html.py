#!/usr/bin/env python3
"""Parse a saved Google Ads change history HTML page into budget and keyword CSVs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path

from campaign_opt.paths import DATA_DIR, PROCESSED_DIR, data_path
from config import COURSE, COURSE_CONFIG
from utils.campaign_metadata import read_keyword_day_index

BUDGET_CHANGE_RE = re.compile(r"\bbudget\b", re.IGNORECASE)
BUDGET_AMOUNT_CHANGE_RE = re.compile(
    r"\bbudget amount\b.*\b(?:increased|decreased|changed|is)\b",
    re.IGNORECASE,
)
BUDGET_DETAIL_CAMPAIGN_RE = re.compile(
    r"(?P<campaign>Course\s+-\s+.+?):\s+Budget amount\b",
    re.IGNORECASE,
)
CURRENCY_RE = re.compile(
    r"(?:US\$|\$|USD\s*)\s*([0-9][0-9,]*(?:\.[0-9]+)?)|"
    r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:USD|dollars?)",
    re.IGNORECASE,
)
DATE_RE = re.compile(r"\b\d{1,2} [A-Za-z]{3} \d{4}, \d{2}:\d{2}:\d{2}\b")
CHANGE_HISTORY_ASSIGNMENT_RE = re.compile(
    r"change_history_data\['CHANGE_HISTORY_TABLE'\]\s*=\s*'((?:\\.|[^'])*)';",
    re.DOTALL,
)
CAMPAIGN_COUNT_RE = re.compile(r"^\d+\s+campaigns?$", re.IGNORECASE)
KEYWORD_TEXT_RE = re.compile(r"^(?P<negative>-)?(?P<keyword>\[[^\]]+\]|\"[^\"]+\"|[^:]+)")
KEYWORD_ACTION_RE = re.compile(r"\bkeyword\b", re.IGNORECASE)
CAMPAIGN_NAME_CHANGE_RE = re.compile(r"Campaign name changed from\s+(.+?)\s+to\s+(.+)$")


@dataclass(frozen=True)
class ChangeAction:
    """One action from the decoded Google Ads change history payload."""

    order: int
    date_text: str
    date: str
    summary: str
    campaign: str
    details: list[tuple[str, str]]


@dataclass
class VisibleChangeRow:
    """Visible cells from one rendered Google Ads change-history table row."""

    user_date: list[str] = field(default_factory=list)
    summary_change: list[str] = field(default_factory=list)
    campaign: list[str] = field(default_factory=list)

    def date_text(self) -> str:
        for value in self.user_date:
            match = DATE_RE.search(value)
            if match:
                return match.group(0)
        return ""

    def change_text(self) -> str:
        return " ".join(self.summary_change)

    def campaign_text(self) -> str:
        return " ".join(value for value in self.campaign if value != "expand_more").strip()


class VisibleChangeHistoryParser(HTMLParser):
    """Small DOM parser for saved Google Ads table rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[VisibleChangeRow] = []
        self._row_stack: list[tuple[VisibleChangeRow, int]] = []
        self._cell_stack: list[str | None] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_by_name = dict(attrs)

        if tag in {"script", "style"}:
            self._skip_depth += 1
            return

        if tag == "div" and attrs_by_name.get("role") == "row":
            if self._row_stack:
                row, depth = self._row_stack[-1]
                self._row_stack[-1] = (row, depth + 1)
            self._row_stack.append((VisibleChangeRow(), 1))
            return

        if self._row_stack:
            row, depth = self._row_stack[-1]
            self._row_stack[-1] = (row, depth + 1)

        if tag == "ess-cell" and self._row_stack:
            self._cell_stack.append(attrs_by_name.get("essfield"))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        if tag == "ess-cell" and self._cell_stack:
            self._cell_stack.pop()

        if not self._row_stack:
            return

        row, depth = self._row_stack[-1]
        depth -= 1
        if depth > 0:
            self._row_stack[-1] = (row, depth)
            return

        row, _ = self._row_stack.pop()
        if self._row_stack:
            parent_row, parent_depth = self._row_stack[-1]
            self._row_stack[-1] = (parent_row, parent_depth - 1)
            if not row.user_date:
                row.user_date = parent_row.user_date.copy()

        if row.summary_change or row.campaign or row.user_date:
            self.rows.append(row)

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._row_stack or not self._cell_stack:
            return

        text = " ".join(data.split())
        if not text:
            return

        cell = self._cell_stack[-1]
        row = self._row_stack[-1][0]
        if cell == "user_date":
            row.user_date.append(text)
        elif cell == "summary_change":
            row.summary_change.append(text)
        elif cell == "campaign":
            row.campaign.append(text)


@dataclass(frozen=True)
class KeywordEvent:
    """Keyword status change for a campaign."""

    order: int
    date: str
    campaign: str
    keyword: str
    match_type: str
    action: str


@dataclass(frozen=True)
class CampaignRename:
    """Campaign rename event from change history."""

    order: int
    date: str
    previous_name: str
    current_name: str


@dataclass(frozen=True)
class CampaignStatusEvent:
    """Campaign enabled/paused event from change history."""

    order: int
    date: str
    campaign: str
    status: str
    summary: str


@dataclass(frozen=True)
class CampaignSnapshot:
    """Campaign state after a budget or keyword change."""

    order: int
    date: str
    campaign: str
    end_date: str
    status: str
    daily_budget: str
    keywords_by_match_type: dict[str, set[str]]
    negative_keywords_by_match_type: dict[str, set[str]]
    change_type: str
    change_summary: str


def parse_google_ads_date(value: str) -> str:
    """Return ISO date from values like '21 Apr 2026, 14:01:16'."""
    try:
        parsed = datetime.strptime(value, "%d %b %Y, %H:%M:%S")
    except ValueError:
        return value
    return parsed.date().isoformat()


def extract_daily_budget(change_text: str) -> str:
    """Extract the final currency amount from a budget-change summary."""
    matches = CURRENCY_RE.findall(change_text)
    if not matches:
        return ""

    amount = next(value for value in matches[-1] if value)
    return amount.replace(",", "")


def extract_previous_daily_budget(change_text: str) -> str:
    """Extract the previous amount from a budget change like 'from X to Y'."""
    if " from " not in f" {change_text.lower()} " or " to " not in f" {change_text.lower()} ":
        return ""
    matches = CURRENCY_RE.findall(change_text)
    if len(matches) < 2:
        return ""

    amount = next(value for value in matches[0] if value)
    return amount.replace(",", "")


def extract_budget_campaign(change_text: str) -> str:
    match = BUDGET_DETAIL_CAMPAIGN_RE.search(change_text)
    if not match:
        return ""
    return clean_campaign_name(match.group("campaign").removeprefix("expand_more "))


def decode_embedded_change_history_actions(html: str) -> list[ChangeAction]:
    """Decode the saved-page JSON payload used by the current Google Ads UI."""
    match = CHANGE_HISTORY_ASSIGNMENT_RE.search(html)
    if not match:
        return []

    # JS string literals may contain escapes like \/ that Python rejects.
    js_payload = match.group(1).replace("\\/", "/")
    outer_payload = ast.literal_eval("'" + js_payload + "'")
    outer = json.loads(outer_payload)
    table = json.loads(outer["2"])

    actions: list[ChangeAction] = []
    for row_index, row in enumerate(table.get("1", [])):
        date_text = row.get("8", {}).get("2", "")
        date = parse_google_ads_date(date_text)
        for action_index, action in enumerate(row.get("9", [])):
            summary = action.get("1", "")
            campaign = action.get("2", {}).get("3", "")
            details: list[tuple[str, str]] = []
            for detail in action.get("5", []):
                detail_campaign = detail.get("2", {}).get("3", "") or campaign
                for text in detail.get("1", []):
                    clean_text = " ".join(text.split())
                    if clean_text:
                        details.append((detail_campaign, clean_text))

            actions.append(
                ChangeAction(
                    order=row_index * 1000 + action_index,
                    date_text=date_text,
                    date=date,
                    summary=summary,
                    campaign=campaign,
                    details=details,
                )
            )

    return actions


def parse_visible_change_history_actions(html: str) -> list[ChangeAction]:
    parser = VisibleChangeHistoryParser()
    parser.feed(html)

    actions: list[ChangeAction] = []
    current_date_text = ""
    current_date = ""
    current_detail_summary = ""
    current_detail_order = 0
    for row_index, row in enumerate(parser.rows):
        row_date_text = row.date_text()
        if row_date_text:
            current_date_text = row_date_text
            current_date = parse_google_ads_date(row_date_text)
            current_detail_summary = ""
            current_detail_order = 0

        date_text = current_date_text
        date = current_date
        campaign = row.campaign_text()
        change_text = row.change_text()
        if not date_text or not change_text:
            continue

        if action_keyword_status(change_text):
            current_detail_summary = change_text
            current_detail_order = row_index * 1000 + 500
            continue

        parsed_keyword = parse_keyword_detail(change_text)
        if current_detail_summary and parsed_keyword and campaign:
            actions.append(
                ChangeAction(
                    order=current_detail_order,
                    date_text=date_text,
                    date=date,
                    summary=current_detail_summary,
                    campaign=campaign,
                    details=[(campaign, change_text)],
                )
            )
            continue

        actions.append(
            ChangeAction(
                # Keep visible rows after the embedded rows when duplicate
                # order values tie; both sources are rendered newest-first.
                order=row_index * 1000 + 500,
                date_text=date_text,
                date=date,
                summary=change_text,
                campaign=campaign,
                details=[(campaign, change_text)] if campaign else [("", change_text)],
            )
        )

    return actions


def dedupe_actions(actions: list[ChangeAction]) -> list[ChangeAction]:
    deduped: list[ChangeAction] = []
    seen: set[tuple[str, str, str]] = set()
    for action in sorted(actions, key=lambda item: item.order):
        details_text = " ".join(detail for _campaign, detail in action.details)
        combined_text = " ".join([action.summary, details_text])
        key = (action.date_text, action.campaign, " ".join(combined_text.split()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped


def decode_change_history_actions(path: Path) -> list[ChangeAction]:
    html = path.read_text(encoding="utf-8", errors="replace")
    actions = [
        *decode_embedded_change_history_actions(html),
        *parse_visible_change_history_actions(html),
    ]
    if not actions:
        raise ValueError("Could not find change history rows in HTML.")
    return dedupe_actions(actions)


def is_aggregate_campaign(value: str) -> bool:
    return bool(CAMPAIGN_COUNT_RE.match(value.strip()))


def is_search_campaign(value: str) -> bool:
    return bool(value) and not is_aggregate_campaign(value) and "Search" in value and "Experiment" not in value


def is_raw_search_campaign(value: str) -> bool:
    return bool(value) and not is_aggregate_campaign(value)


def clean_campaign_name(value: str) -> str:
    return value.strip().strip(" ?\uFFFD\"'\u2018\u2019\u201C\u201D")


def extract_campaign_renames(actions: list[ChangeAction]) -> list[CampaignRename]:
    renames: list[CampaignRename] = []
    for action in actions:
        for _detail_campaign, detail_text in action.details:
            match = CAMPAIGN_NAME_CHANGE_RE.search(detail_text)
            if not match:
                continue
            previous_name = clean_campaign_name(match.group(1))
            current_name = clean_campaign_name(match.group(2))
            if is_search_campaign(previous_name) and is_search_campaign(current_name):
                renames.append(
                    CampaignRename(
                        order=action.order,
                        date=action.date,
                        previous_name=previous_name,
                        current_name=current_name,
                    )
                )
    return renames


def historical_campaign_name(
    campaign: str,
    order: int,
    renames: list[CampaignRename],
) -> str:
    """Return the campaign name that applied at this point in the change timeline."""
    historical_name = campaign
    for rename in sorted(renames, key=lambda item: item.order):
        if order > rename.order and historical_name == rename.current_name:
            historical_name = rename.previous_name
    return historical_name


def canonical_campaign_name(campaign: str, renames: list[CampaignRename]) -> str:
    canonical_name = campaign
    changed = True
    while changed:
        changed = False
        for rename in sorted(renames, key=lambda item: item.order, reverse=True):
            if canonical_name == rename.previous_name:
                canonical_name = rename.current_name
                changed = True
    return canonical_name


def parse_keyword_detail(detail: str) -> tuple[str, str, bool] | None:
    detail = detail.strip()
    detail = detail.removeprefix("expand_more ").strip()
    detail_lower = detail.lower()
    if (
        " keyword added" in detail_lower
        or " keyword enabled" in detail_lower
        or " keyword removed" in detail_lower
        or " keyword paused" in detail_lower
    ):
        return None

    match = KEYWORD_TEXT_RE.match(detail)
    if not match:
        return None

    raw_keyword = match.group("keyword").strip()
    negative = bool(match.group("negative"))
    if raw_keyword.startswith("[") and raw_keyword.endswith("]"):
        match_type = "Exact"
        keyword = raw_keyword[1:-1]
    elif raw_keyword.startswith('"') and raw_keyword.endswith('"'):
        match_type = "Phrase"
        keyword = raw_keyword[1:-1]
    else:
        match_type = "Broad"
        keyword = raw_keyword

    keyword = " ".join(keyword.lower().split())
    if not keyword:
        return None
    return keyword, match_type, negative


def action_keyword_status(summary: str) -> str | None:
    summary_lower = summary.lower()
    if "negative" in summary_lower or not KEYWORD_ACTION_RE.search(summary):
        return None
    if any(token in summary_lower for token in ("added", "enabled", "active")):
        return "add"
    if any(token in summary_lower for token in ("removed", "paused")):
        return "remove"
    return None


def negative_keyword_status(summary: str) -> str | None:
    summary_lower = summary.lower()
    if "negative" not in summary_lower or not KEYWORD_ACTION_RE.search(summary):
        return None
    if any(token in summary_lower for token in ("added", "enabled", "active")):
        return "add"
    if any(token in summary_lower for token in ("removed", "paused")):
        return "remove"
    return None


def summary_match_type(summary: str) -> str | None:
    summary_lower = summary.lower()
    if "exact match keyword" in summary_lower:
        return "Exact"
    if "phrase match keyword" in summary_lower:
        return "Phrase"
    if "broad match keyword" in summary_lower:
        return "Broad"
    return None


def extract_keyword_events(
    actions: list[ChangeAction],
    renames: list[CampaignRename],
    *,
    negative: bool = False,
) -> list[KeywordEvent]:
    events: list[KeywordEvent] = []
    for action in actions:
        status = negative_keyword_status(action.summary) if negative else action_keyword_status(action.summary)
        if status is None:
            continue
        action_match_type = summary_match_type(action.summary)

        for detail_campaign, detail_text in action.details:
            campaign = historical_campaign_name(detail_campaign or action.campaign, action.order, renames)
            if not is_search_campaign(campaign):
                continue

            parsed = parse_keyword_detail(detail_text)
            if parsed is None:
                continue
            keyword, match_type, parsed_negative = parsed
            if parsed_negative != negative:
                continue
            match_type = action_match_type or match_type

            events.append(
                KeywordEvent(
                    order=action.order,
                    date=action.date,
                    campaign=campaign,
                    keyword=keyword,
                    match_type=match_type,
                    action=status,
                )
            )

    return events


def extract_budget_events(
    actions: list[ChangeAction],
    renames: list[CampaignRename],
) -> list[tuple[int, str, str, str, str]]:
    events: list[tuple[int, str, str, str, str]] = []
    for action in actions:
        change_text = " ".join([action.summary, *(detail for _, detail in action.details)])
        if not BUDGET_AMOUNT_CHANGE_RE.search(change_text):
            continue
        campaign = extract_budget_campaign(change_text) or action.campaign
        if "Prospecting" in campaign or not extract_budget_campaign(change_text):
            campaign = historical_campaign_name(campaign, action.order, renames)
        if not is_search_campaign(campaign):
            continue
        daily_budget = extract_daily_budget(change_text)
        if daily_budget:
            events.append((action.order, action.date, campaign, daily_budget, action.summary))

    return events


def extract_campaign_status_events(
    actions: list[ChangeAction],
    renames: list[CampaignRename],
) -> list[CampaignStatusEvent]:
    events: list[CampaignStatusEvent] = []
    for action in actions:
        summary_lower = action.summary.lower()
        if "campaign" not in summary_lower or not (
            "active" in summary_lower or "paused" in summary_lower
        ):
            continue

        for detail_campaign, detail_text in action.details:
            detail_lower = detail_text.lower()
            if "status changed" not in detail_lower:
                continue
            campaign = historical_campaign_name(detail_campaign or action.campaign, action.order, renames)
            if not is_search_campaign(campaign):
                continue
            if "to active" in detail_lower:
                status = "active"
            elif "to paused" in detail_lower:
                status = "paused"
            else:
                continue
            events.append(
                CampaignStatusEvent(
                    order=action.order,
                    date=action.date,
                    campaign=campaign,
                    status=status,
                    summary=action.summary,
                )
            )

    return events


def group_keyword_events(events: list[KeywordEvent]) -> list[tuple[int, str, str, list[KeywordEvent]]]:
    grouped: dict[tuple[int, str, str], list[KeywordEvent]] = defaultdict(list)
    for event in events:
        grouped[(event.order, event.date, event.campaign)].append(event)

    return [
        (order, date, campaign, grouped_events)
        for (order, date, campaign), grouped_events in grouped.items()
    ]


def describe_keyword_change(events: list[KeywordEvent], *, negative: bool = False) -> tuple[str, str]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for event in events:
        counts[(event.match_type, event.action)] += 1

    keyword_label = "negative keyword" if negative else "keyword"
    change_type_prefix = "negative_" if negative else ""
    parts = [
        f"{count} {match_type.lower()} {keyword_label} {'added/enabled' if action == 'add' else 'removed/paused'}"
        for (match_type, action), count in sorted(counts.items())
    ]
    change_types = sorted(
        {f"{change_type_prefix}{event.match_type.lower()}_keyword_{event.action}" for event in events}
    )
    return ";".join(change_types), "; ".join(parts)


def copy_keywords_by_match_type(source: dict[str, set[str]]) -> dict[str, set[str]]:
    return {match_type: set(source.get(match_type, set())) for match_type in ["Broad", "Phrase", "Exact"]}


def campaign_change_dates(
    actions: list[ChangeAction],
    renames: list[CampaignRename],
) -> dict[str, list[str]]:
    dates_by_campaign: dict[str, set[str]] = defaultdict(set)
    for action in actions:
        campaigns = {action.campaign, *(detail_campaign for detail_campaign, _detail in action.details)}
        for campaign in campaigns:
            historical_campaign = historical_campaign_name(campaign, action.order, renames)
            if is_search_campaign(historical_campaign) and action.date:
                dates_by_campaign[historical_campaign].add(action.date)

    return {
        campaign: sorted(dates)
        for campaign, dates in dates_by_campaign.items()
    }


def next_campaign_change_date(
    snapshot: CampaignSnapshot,
    dates_by_campaign: dict[str, list[str]],
) -> str:
    for change_date in dates_by_campaign.get(snapshot.campaign, []):
        if change_date > snapshot.date:
            return change_date
    return ""


def build_campaign_snapshots(actions: list[ChangeAction]) -> list[CampaignSnapshot]:
    renames = extract_campaign_renames(actions)
    dates_by_campaign = campaign_change_dates(actions, renames)
    keyword_events = extract_keyword_events(actions, renames)
    negative_keyword_events = extract_keyword_events(actions, renames, negative=True)
    budget_events = extract_budget_events(actions, renames)
    status_events = extract_campaign_status_events(actions, renames)
    timeline = [
        ("keyword", order, (order, date, campaign, events))
        for order, date, campaign, events in group_keyword_events(keyword_events)
    ] + [
        ("negative_keyword", order, (order, date, campaign, events))
        for order, date, campaign, events in group_keyword_events(negative_keyword_events)
    ] + [
        ("budget", order, (order, date, campaign, daily_budget, summary))
        for order, date, campaign, daily_budget, summary in budget_events
    ] + [
        ("status", event.order, event)
        for event in status_events
    ] + [
        ("rename", rename.order, rename)
        for rename in renames
    ]

    active_keywords: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"Broad": set(), "Phrase": set(), "Exact": set()}
    )
    active_negative_keywords: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"Broad": set(), "Phrase": set(), "Exact": set()}
    )
    active_budgets: dict[str, str] = {}
    active_statuses: dict[str, str] = {}
    snapshots: list[CampaignSnapshot] = []

    for event_type, _order, payload in sorted(timeline, key=lambda item: item[1], reverse=True):
        if event_type == "rename":
            rename = payload
            assert isinstance(rename, CampaignRename)
            if rename.previous_name in active_keywords and rename.current_name not in active_keywords:
                active_keywords[rename.current_name] = active_keywords[rename.previous_name]
            if (
                rename.previous_name in active_negative_keywords
                and rename.current_name not in active_negative_keywords
            ):
                active_negative_keywords[rename.current_name] = active_negative_keywords[rename.previous_name]
            if rename.previous_name in active_budgets and rename.current_name not in active_budgets:
                active_budgets[rename.current_name] = active_budgets[rename.previous_name]
            if rename.previous_name in active_statuses and rename.current_name not in active_statuses:
                active_statuses[rename.current_name] = active_statuses[rename.previous_name]
            continue

        if event_type in {"keyword", "negative_keyword"}:
            order, date, campaign, events = payload
            is_negative_keyword_event = event_type == "negative_keyword"
            change_type, change_summary = describe_keyword_change(events, negative=is_negative_keyword_event)
            active_keyword_store = active_negative_keywords if is_negative_keyword_event else active_keywords
            snapshot_keywords_by_match_type = copy_keywords_by_match_type(active_keywords[campaign])
            snapshot_negative_keywords_by_match_type = copy_keywords_by_match_type(
                active_negative_keywords[campaign]
            )
            for event in events:
                keyword_set = active_keyword_store[campaign][event.match_type]
                snapshot_store = (
                    snapshot_negative_keywords_by_match_type
                    if is_negative_keyword_event
                    else snapshot_keywords_by_match_type
                )
                if event.action == "add":
                    keyword_set.add(event.keyword)
                    snapshot_store[event.match_type].add(event.keyword)
                elif event.action == "remove":
                    # If the visible history starts with a remove/pause event,
                    # the removed keyword itself is evidence that it belonged
                    # to the immediately preceding setup.
                    snapshot_store[event.match_type].add(event.keyword)
                    keyword_set.discard(event.keyword)

            snapshots.append(
                CampaignSnapshot(
                    order=order,
                    date=date,
                    campaign=campaign,
                    end_date="",
                    status=active_statuses.get(campaign, ""),
                    daily_budget=active_budgets.get(campaign, ""),
                    keywords_by_match_type=snapshot_keywords_by_match_type,
                    negative_keywords_by_match_type=snapshot_negative_keywords_by_match_type,
                    change_type=change_type,
                    change_summary=change_summary,
                )
            )
            continue

        if event_type == "status":
            event = payload
            assert isinstance(event, CampaignStatusEvent)
            active_statuses[event.campaign] = event.status
            snapshots.append(
                CampaignSnapshot(
                    order=event.order,
                    date=event.date,
                    campaign=event.campaign,
                    end_date="",
                    status=event.status,
                    daily_budget=active_budgets.get(event.campaign, ""),
                    keywords_by_match_type=copy_keywords_by_match_type(active_keywords[event.campaign]),
                    negative_keywords_by_match_type=copy_keywords_by_match_type(
                        active_negative_keywords[event.campaign]
                    ),
                    change_type=f"campaign_{event.status}",
                    change_summary=event.summary,
                )
            )
            continue

        order, date, campaign, daily_budget, summary = payload
        active_budgets[campaign] = daily_budget
        snapshots.append(
            CampaignSnapshot(
                order=order,
                date=date,
                campaign=campaign,
                end_date="",
                status=active_statuses.get(campaign, ""),
                daily_budget=daily_budget,
                keywords_by_match_type=copy_keywords_by_match_type(active_keywords[campaign]),
                negative_keywords_by_match_type=copy_keywords_by_match_type(active_negative_keywords[campaign]),
                change_type="daily_budget",
                change_summary=summary,
            )
        )

    sorted_snapshots = sorted(snapshots, key=lambda snapshot: (snapshot.date, snapshot.order, snapshot.campaign))
    next_start_by_campaign: dict[str, str] = {
        rename.previous_name: rename.date
        for rename in renames
    }
    with_end_dates: list[CampaignSnapshot] = []
    for snapshot in reversed(sorted_snapshots):
        end_date = next_start_by_campaign.get(snapshot.campaign, "") or next_campaign_change_date(
            snapshot,
            dates_by_campaign,
        )
        with_end_dates.append(
            CampaignSnapshot(
                order=snapshot.order,
                date=snapshot.date,
                campaign=snapshot.campaign,
                end_date=end_date,
                status=snapshot.status,
                daily_budget=snapshot.daily_budget,
                keywords_by_match_type=snapshot.keywords_by_match_type,
                negative_keywords_by_match_type=snapshot.negative_keywords_by_match_type,
                change_type=snapshot.change_type,
                change_summary=snapshot.change_summary,
            )
        )
        next_start_by_campaign[snapshot.campaign] = snapshot.date

    return backfill_empty_keyword_sets(list(reversed(with_end_dates)), renames)


def backfill_empty_keyword_sets(
    snapshots: list[CampaignSnapshot],
    renames: list[CampaignRename],
) -> list[CampaignSnapshot]:
    """Fill empty states from the nearest known setup for the same campaign.

    Collapsed rows in saved Google Ads pages often show budget/status changes
    without repeating the keyword list. If a later or earlier row for the same
    campaign has an inferred keyword set, use that as the campaign setup for
    otherwise-empty rows.
    """
    future_keywords_by_campaign: dict[str, dict[str, set[str]]] = {}
    backfilled_reversed: list[CampaignSnapshot] = []
    for snapshot in reversed(snapshots):
        campaign_key = canonical_campaign_name(snapshot.campaign, renames)
        keywords = snapshot_keywords(snapshot)
        replacement = snapshot.keywords_by_match_type
        if not keywords and campaign_key in future_keywords_by_campaign:
            replacement = future_keywords_by_campaign[campaign_key]
        elif keywords:
            future_keywords_by_campaign[campaign_key] = snapshot.keywords_by_match_type

        backfilled_reversed.append(
            CampaignSnapshot(
                order=snapshot.order,
                date=snapshot.date,
                campaign=snapshot.campaign,
                end_date=snapshot.end_date,
                status=snapshot.status,
                daily_budget=snapshot.daily_budget,
                keywords_by_match_type=copy_keywords_by_match_type(replacement),
                negative_keywords_by_match_type=copy_keywords_by_match_type(
                    snapshot.negative_keywords_by_match_type
                ),
                change_type=snapshot.change_type,
                change_summary=snapshot.change_summary,
            )
        )

    backfilled = list(reversed(backfilled_reversed))
    previous_keywords_by_campaign: dict[str, dict[str, set[str]]] = {}
    final_snapshots: list[CampaignSnapshot] = []
    for snapshot in backfilled:
        campaign_key = canonical_campaign_name(snapshot.campaign, renames)
        keywords = snapshot_keywords(snapshot)
        replacement = snapshot.keywords_by_match_type
        if not keywords and campaign_key in previous_keywords_by_campaign:
            replacement = previous_keywords_by_campaign[campaign_key]
        elif keywords:
            previous_keywords_by_campaign[campaign_key] = snapshot.keywords_by_match_type

        final_snapshots.append(
            CampaignSnapshot(
                order=snapshot.order,
                date=snapshot.date,
                campaign=snapshot.campaign,
                end_date=snapshot.end_date,
                status=snapshot.status,
                daily_budget=snapshot.daily_budget,
                keywords_by_match_type=copy_keywords_by_match_type(replacement),
                negative_keywords_by_match_type=copy_keywords_by_match_type(
                    snapshot.negative_keywords_by_match_type
                ),
                change_type=snapshot.change_type,
                change_summary=snapshot.change_summary,
            )
        )

    return final_snapshots


def join_keywords(keywords: set[str]) -> str:
    return "; ".join(sorted(keywords))


def snapshot_keywords(snapshot: CampaignSnapshot) -> set[str]:
    keywords: set[str] = set()
    for match_keywords in snapshot.keywords_by_match_type.values():
        keywords.update(match_keywords)
    return keywords


def snapshot_negative_keywords(snapshot: CampaignSnapshot) -> set[str]:
    keywords: set[str] = set()
    for match_keywords in snapshot.negative_keywords_by_match_type.values():
        keywords.update(match_keywords)
    return keywords


def snapshot_match_types(snapshot: CampaignSnapshot) -> str:
    return "; ".join(
        match_type
        for match_type in ["Broad", "Phrase", "Exact"]
        if snapshot.keywords_by_match_type[match_type]
    )


def clean_change_summary(value: str) -> str:
    return " ".join(value.removeprefix("expand_more ").split())


def make_keyword_set_id(index: int) -> str:
    return f"ks_{index:04d}"


def campaign_summary_rows(actions: list[ChangeAction]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for version, snapshot in enumerate(build_campaign_snapshots(actions), start=1):
        keywords = snapshot_keywords(snapshot)
        negative_keywords = snapshot_negative_keywords(snapshot)
        rows.append(
            {
                "campaign_version": str(version),
                "campaign": snapshot.campaign,
                "start_date": snapshot.date,
                "end_date": snapshot.end_date,
                "status": snapshot.status,
                "change_type": snapshot.change_type,
                "change_summary": clean_change_summary(snapshot.change_summary),
                "daily_budget": snapshot.daily_budget,
                "match_types": snapshot_match_types(snapshot),
                "num_unique_keywords": str(len(keywords)),
                "broad_keyword_count": str(len(snapshot.keywords_by_match_type["Broad"])),
                "phrase_keyword_count": str(len(snapshot.keywords_by_match_type["Phrase"])),
                "exact_keyword_count": str(len(snapshot.keywords_by_match_type["Exact"])),
                "num_negative_keywords": str(len(negative_keywords)),
                "_broad_keywords": join_keywords(snapshot.keywords_by_match_type["Broad"]),
                "_phrase_keywords": join_keywords(snapshot.keywords_by_match_type["Phrase"]),
                "_exact_keywords": join_keywords(snapshot.keywords_by_match_type["Exact"]),
                "_unique_keywords": join_keywords(keywords),
                "_negative_broad_keywords": join_keywords(
                    snapshot.negative_keywords_by_match_type["Broad"]
                ),
                "_negative_phrase_keywords": join_keywords(
                    snapshot.negative_keywords_by_match_type["Phrase"]
                ),
                "_negative_exact_keywords": join_keywords(
                    snapshot.negative_keywords_by_match_type["Exact"]
                ),
                "_negative_keywords": join_keywords(negative_keywords),
            }
        )
    return rows


def split_keyword_list(value: str) -> set[str]:
    return {keyword.strip() for keyword in value.split(";") if keyword.strip()}


def clean_keyword_text(value: str) -> str:
    return " ".join(value.strip().strip('"[]').lower().split())


def next_iso_date(value: str) -> str:
    return (datetime.strptime(value, "%Y-%m-%d").date() + timedelta(days=1)).isoformat()


def raw_keywords_for_window(
    records: list[tuple[str, str, str]],
    start_date: str,
    end_date: str,
) -> tuple[dict[str, set[str]], int]:
    keywords_by_match_type = {"Broad": set(), "Phrase": set(), "Exact": set()}
    row_count = 0
    for day, keyword, match_type in records:
        if start_date <= day < end_date:
            row_count += 1
            keywords_by_match_type.setdefault(match_type, set()).add(keyword)
    return keywords_by_match_type, row_count


def all_keywords(keywords_by_match_type: dict[str, set[str]]) -> set[str]:
    keywords: set[str] = set()
    for values in keywords_by_match_type.values():
        keywords.update(values)
    return keywords


def match_types_for_keywords(keywords_by_match_type: dict[str, set[str]]) -> str:
    return "; ".join(
        match_type
        for match_type in ["Broad", "Phrase", "Exact"]
        if keywords_by_match_type[match_type]
    )


def snapshot_lookup_by_canonical_campaign(
    snapshots: list[CampaignSnapshot],
    renames: list[CampaignRename],
) -> dict[str, list[CampaignSnapshot]]:
    snapshots_by_campaign: dict[str, list[CampaignSnapshot]] = defaultdict(list)
    for snapshot in snapshots:
        snapshots_by_campaign[canonical_campaign_name(snapshot.campaign, renames)].append(snapshot)

    return {
        campaign: sorted(campaign_snapshots, key=lambda item: (item.date, item.order))
        for campaign, campaign_snapshots in snapshots_by_campaign.items()
    }


def latest_snapshot_on_or_before(
    snapshots: list[CampaignSnapshot],
    date: str,
) -> CampaignSnapshot | None:
    latest_snapshot = None
    for snapshot in snapshots:
        if snapshot.date <= date:
            latest_snapshot = snapshot
        else:
            break
    return latest_snapshot


def inferred_previous_budget_from_next_change(
    snapshots: list[CampaignSnapshot],
    start_date: str,
    end_date: str,
) -> str:
    """Use the pre-change budget for raw intervals before the first budget snapshot."""
    for snapshot in snapshots:
        if snapshot.date < start_date:
            continue
        if end_date and snapshot.date > end_date:
            break
        if snapshot.change_type != "daily_budget":
            continue
        previous_budget = extract_previous_daily_budget(snapshot.change_summary)
        if previous_budget:
            return previous_budget
    return ""


def raw_campaign_summary_rows(
    actions: list[ChangeAction],
    keyword_day_panel: Path,
) -> list[dict[str, str]]:
    """Build campaign intervals from keyword-day panel coverage, then attach change history."""
    renames = extract_campaign_renames(actions)
    snapshots = build_campaign_snapshots(actions)
    snapshots_by_campaign = snapshot_lookup_by_canonical_campaign(snapshots, renames)
    keywords_by_campaign = read_keyword_day_index(keyword_day_panel)

    rows: list[dict[str, str]] = []
    version = 1
    for campaign, records in sorted(keywords_by_campaign.items()):
        if not is_raw_search_campaign(campaign):
            continue
        records = sorted(records, key=lambda item: item[0])
        raw_start_date = records[0][0]
        raw_end_date = next_iso_date(records[-1][0])
        canonical_campaign = canonical_campaign_name(campaign, renames)
        campaign_snapshots = snapshots_by_campaign.get(canonical_campaign, [])
        change_dates = {
            snapshot.date
            for snapshot in campaign_snapshots
            if raw_start_date < snapshot.date < raw_end_date
        }
        boundaries = [raw_start_date, *sorted(change_dates), raw_end_date]
        previous_positive_keywords_by_match_type: dict[str, set[str]] | None = None

        for start_date, end_date in zip(boundaries, boundaries[1:]):
            raw_keywords_by_match_type, raw_report_row_count = raw_keywords_for_window(
                records,
                start_date,
                end_date,
            )
            if raw_report_row_count == 0:
                continue

            raw_keywords = all_keywords(raw_keywords_by_match_type)
            attached_snapshot = latest_snapshot_on_or_before(campaign_snapshots, start_date)
            daily_budget = attached_snapshot.daily_budget if attached_snapshot else ""
            if not daily_budget:
                daily_budget = inferred_previous_budget_from_next_change(
                    campaign_snapshots,
                    start_date,
                    end_date,
                )
            if not daily_budget:
                continue
            negative_keywords_by_match_type = (
                attached_snapshot.negative_keywords_by_match_type
                if attached_snapshot
                else {"Broad": set(), "Phrase": set(), "Exact": set()}
            )
            negative_keywords = all_keywords(negative_keywords_by_match_type)
            change_history_keywords = snapshot_keywords(attached_snapshot) if attached_snapshot else set()
            exact_change_types = sorted(
                {
                    snapshot.change_type
                    for snapshot in campaign_snapshots
                    if snapshot.date == start_date
                }
            )
            change_type = ";".join(exact_change_types) or "raw_keyword_coverage"
            positive_keywords_by_match_type = copy_keywords_by_match_type(raw_keywords_by_match_type)
            positive_keywords = all_keywords(positive_keywords_by_match_type)

            rows.append(
                {
                    "campaign_version": str(version),
                    "campaign": campaign,
                    "start_date": start_date,
                    "end_date": "" if end_date == raw_end_date else end_date,
                    "status": attached_snapshot.status if attached_snapshot else "",
                    "change_type": change_type,
                    "change_summary": attached_snapshot.change_summary if attached_snapshot else "",
                    "daily_budget": daily_budget,
                    "match_types": match_types_for_keywords(positive_keywords_by_match_type),
                    "num_unique_keywords": str(len(positive_keywords)),
                    "broad_keyword_count": str(len(positive_keywords_by_match_type["Broad"])),
                    "phrase_keyword_count": str(len(positive_keywords_by_match_type["Phrase"])),
                    "exact_keyword_count": str(len(positive_keywords_by_match_type["Exact"])),
                    "num_negative_keywords": str(len(negative_keywords)),
                    "_raw_report_row_count": str(raw_report_row_count),
                    "_raw_broad_keywords": join_keywords(raw_keywords_by_match_type["Broad"]),
                    "_raw_phrase_keywords": join_keywords(raw_keywords_by_match_type["Phrase"]),
                    "_raw_exact_keywords": join_keywords(raw_keywords_by_match_type["Exact"]),
                    "_broad_keywords": join_keywords(positive_keywords_by_match_type["Broad"]),
                    "_phrase_keywords": join_keywords(positive_keywords_by_match_type["Phrase"]),
                    "_exact_keywords": join_keywords(positive_keywords_by_match_type["Exact"]),
                    "_unique_keywords": join_keywords(positive_keywords),
                    "_negative_broad_keywords": join_keywords(negative_keywords_by_match_type["Broad"]),
                    "_negative_phrase_keywords": join_keywords(negative_keywords_by_match_type["Phrase"]),
                    "_negative_exact_keywords": join_keywords(negative_keywords_by_match_type["Exact"]),
                    "_negative_keywords": join_keywords(negative_keywords),
                    "_raw_unique_keywords": join_keywords(raw_keywords),
                    "_change_history_unique_keywords": join_keywords(change_history_keywords),
                    "_raw_keywords_not_in_change_history": join_keywords(
                        raw_keywords - change_history_keywords
                    ),
                    "_change_history_keywords_not_in_raw": join_keywords(
                        change_history_keywords - raw_keywords
                    ),
                }
            )
            version += 1

    return union_daily_budget_keyword_runs(rows)


def union_daily_budget_keyword_runs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Share a unioned positive keyword set across budget-only splits."""
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["campaign"]].append(row)

    for campaign_rows in grouped_rows.values():
        current_run: list[dict[str, str]] = []
        for row in campaign_rows:
            if current_run and row["change_type"] != "daily_budget":
                apply_union_keyword_set(current_run)
                current_run = []
            current_run.append(row)
        if current_run:
            apply_union_keyword_set(current_run)

    return rows


def apply_union_keyword_set(rows: list[dict[str, str]]) -> None:
    keywords_by_match_type = {
        "Broad": set(),
        "Phrase": set(),
        "Exact": set(),
    }
    for row in rows:
        keywords_by_match_type["Broad"].update(split_keyword_list(row.get("_raw_broad_keywords", "")))
        keywords_by_match_type["Phrase"].update(split_keyword_list(row.get("_raw_phrase_keywords", "")))
        keywords_by_match_type["Exact"].update(split_keyword_list(row.get("_raw_exact_keywords", "")))

    keywords = all_keywords(keywords_by_match_type)
    for row in rows:
        row["match_types"] = match_types_for_keywords(keywords_by_match_type)
        row["num_unique_keywords"] = str(len(keywords))
        row["broad_keyword_count"] = str(len(keywords_by_match_type["Broad"]))
        row["phrase_keyword_count"] = str(len(keywords_by_match_type["Phrase"]))
        row["exact_keyword_count"] = str(len(keywords_by_match_type["Exact"]))
        row["_broad_keywords"] = join_keywords(keywords_by_match_type["Broad"])
        row["_phrase_keywords"] = join_keywords(keywords_by_match_type["Phrase"])
        row["_exact_keywords"] = join_keywords(keywords_by_match_type["Exact"])
        row["_unique_keywords"] = join_keywords(keywords)


def enrich_campaign_summary_with_search_keywords(
    rows: list[dict[str, str]],
    keyword_day_panel: Path,
) -> list[dict[str, str]]:
    """Use keyword-day panel rows for positive keyword sets and cross-checks."""
    keywords_by_campaign = read_keyword_day_index(keyword_day_panel)
    enriched_rows: list[dict[str, str]] = []
    for row in rows:
        start_date = row["start_date"]
        end_date = row["end_date"] or "9999-12-31"
        raw_report_row_count = 0
        raw_keywords_by_match_type = {"Broad": set(), "Phrase": set(), "Exact": set()}
        for day, keyword, match_type in keywords_by_campaign.get(row["campaign"], []):
            if start_date <= day < end_date:
                raw_report_row_count += 1
                raw_keywords_by_match_type.setdefault(match_type, set()).add(keyword)
        raw_keywords = set().union(*raw_keywords_by_match_type.values())
        change_history_keywords = split_keyword_list(row.get("_unique_keywords", ""))

        enriched_row = row.copy()
        enriched_row["_raw_report_row_count"] = str(raw_report_row_count)
        enriched_row["match_types"] = "; ".join(
            match_type
            for match_type in ["Broad", "Phrase", "Exact"]
            if raw_keywords_by_match_type[match_type]
        )
        enriched_row["num_unique_keywords"] = str(len(raw_keywords))
        enriched_row["broad_keyword_count"] = str(len(raw_keywords_by_match_type["Broad"]))
        enriched_row["phrase_keyword_count"] = str(len(raw_keywords_by_match_type["Phrase"]))
        enriched_row["exact_keyword_count"] = str(len(raw_keywords_by_match_type["Exact"]))
        enriched_row["raw_num_unique_keywords"] = str(len(raw_keywords))
        enriched_row["raw_keyword_count_delta"] = str(
            len(raw_keywords) - len(change_history_keywords)
        )
        enriched_row["_change_history_unique_keywords"] = row.get("_unique_keywords", "")
        enriched_row["_broad_keywords"] = join_keywords(raw_keywords_by_match_type["Broad"])
        enriched_row["_phrase_keywords"] = join_keywords(raw_keywords_by_match_type["Phrase"])
        enriched_row["_exact_keywords"] = join_keywords(raw_keywords_by_match_type["Exact"])
        enriched_row["_unique_keywords"] = join_keywords(raw_keywords)
        enriched_row["_raw_unique_keywords"] = join_keywords(raw_keywords)
        enriched_row["_raw_keywords_not_in_change_history"] = join_keywords(raw_keywords - change_history_keywords)
        enriched_row["_change_history_keywords_not_in_raw"] = join_keywords(change_history_keywords - raw_keywords)
        enriched_rows.append(enriched_row)

    return enriched_rows


def keyword_set_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("_broad_keywords", ""),
        row.get("_phrase_keywords", ""),
        row.get("_exact_keywords", ""),
        row.get("_negative_broad_keywords", ""),
        row.get("_negative_phrase_keywords", ""),
        row.get("_negative_exact_keywords", ""),
    )


def assign_keyword_set_ids(
    summary_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    keyword_set_ids: dict[tuple[str, ...], str] = {}
    keyword_set_rows: dict[str, dict[str, str]] = {}
    keyed_summary_rows: list[dict[str, str]] = []

    for row in summary_rows:
        key = keyword_set_key(row)
        if key not in keyword_set_ids:
            keyword_set_ids[key] = make_keyword_set_id(len(keyword_set_ids) + 1)
            keyword_set_id = keyword_set_ids[key]
            keyword_set_rows[keyword_set_id] = {
                "keyword_set_id": keyword_set_id,
                "broad_keywords": row.get("_broad_keywords", ""),
                "phrase_keywords": row.get("_phrase_keywords", ""),
                "exact_keywords": row.get("_exact_keywords", ""),
                "unique_keywords": row.get("_unique_keywords", ""),
                "negative_broad_keywords": row.get("_negative_broad_keywords", ""),
                "negative_phrase_keywords": row.get("_negative_phrase_keywords", ""),
                "negative_exact_keywords": row.get("_negative_exact_keywords", ""),
                "negative_keywords": row.get("_negative_keywords", ""),
                "used_by_campaign_versions": row["campaign_version"],
            }
        else:
            keyword_set_id = keyword_set_ids[key]
            keyword_set_rows[keyword_set_id]["used_by_campaign_versions"] += f"; {row['campaign_version']}"

        keyed_row = row.copy()
        keyed_row["keyword_set_id"] = keyword_set_id
        keyed_summary_rows.append(keyed_row)

    return keyed_summary_rows, list(keyword_set_rows.values())


def campaign_keyword_check_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in summary_rows:
        rows.append(
            {
                "keyword_set_id": row["keyword_set_id"],
                "campaign_version": row["campaign_version"],
                "campaign": row["campaign"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "raw_unique_keywords": row.get("_raw_unique_keywords", ""),
                "change_history_unique_keywords": row.get("_change_history_unique_keywords", ""),
                "raw_keywords_not_in_change_history": row.get(
                    "_raw_keywords_not_in_change_history", ""
                ),
                "change_history_keywords_not_in_raw": row.get(
                    "_change_history_keywords_not_in_raw", ""
                ),
            }
        )
    return rows


def keyword_overlap_rows(actions: list[ChangeAction]) -> list[dict[str, str]]:
    snapshots = build_campaign_snapshots(actions)
    rows: list[dict[str, str]] = []
    for left_index, left in enumerate(snapshots):
        left_keywords = snapshot_keywords(left)
        if not left_keywords:
            continue
        for right_index, right in enumerate(snapshots[left_index + 1 :], start=left_index + 1):
            right_keywords = snapshot_keywords(right)
            if not right_keywords:
                continue
            intersection = left_keywords & right_keywords
            union = left_keywords | right_keywords
            rows.append(
                {
                    "campaign_version_a": str(left_index + 1),
                    "campaign_a": left.campaign,
                    "start_date_a": left.date,
                    "change_type_a": left.change_type,
                    "daily_budget_a": left.daily_budget,
                    "num_keywords_a": str(len(left_keywords)),
                    "campaign_version_b": str(right_index + 1),
                    "campaign_b": right.campaign,
                    "start_date_b": right.date,
                    "change_type_b": right.change_type,
                    "daily_budget_b": right.daily_budget,
                    "num_keywords_b": str(len(right_keywords)),
                    "shared_keywords": join_keywords(intersection),
                    "num_shared_keywords": str(len(intersection)),
                    "num_union_keywords": str(len(union)),
                    "jaccard_similarity": f"{len(intersection) / len(union):.4f}" if union else "",
                    "overlap_pct_a": (
                        f"{len(intersection) / len(left_keywords):.4f}" if left_keywords else ""
                    ),
                    "overlap_pct_b": (
                        f"{len(intersection) / len(right_keywords):.4f}" if right_keywords else ""
                    ),
                }
            )

    return rows


def keep_final_budget_per_day(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep the latest visible change for each campaign/date pair.

    Google Ads change history is rendered newest-first, so the first row for a
    campaign/date is the final amount for that day.
    """
    seen: set[tuple[str, str]] = set()
    final_rows: list[dict[str, str]] = []
    for row in rows:
        key = (row["date"], row["campaign"])
        if key in seen:
            continue
        seen.add(key)
        final_rows.append(row)

    return final_rows


def write_csv(
    rows: list[dict[str, str]],
    output_path: Path | None,
    fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    if output_path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_change_history_html(html_file: Path | None = None) -> Path:
    """Locate the saved change-history HTML under sys_think/data/."""
    if html_file is not None:
        if not html_file.exists():
            raise FileNotFoundError(f"Change history HTML not found: {html_file}")
        return html_file

    data_dir = DATA_DIR
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Course data directory not found: {data_dir}. "
            "Save the Google Ads change history HTML there or pass --html-file."
        )

    candidates = sorted(data_dir.glob("*.html"))
    preferred = [path for path in candidates if "change history" in path.name.lower()]
    pool = preferred or candidates
    if len(pool) == 1:
        return pool[0]
    if not pool:
        raise FileNotFoundError(
            f"No change history HTML in {data_dir}. "
            "Save the Google Ads export there or pass --html-file."
        )
    names = ", ".join(path.name for path in pool)
    raise FileNotFoundError(
        f"Multiple HTML files in {data_dir}: {names}. Pass --html-file explicitly."
    )


def resolve_keyword_panel(kw_day_panel: Path | None) -> Path | None:
    if kw_day_panel:
        return kw_day_panel
    processed = data_path("processed", "kw-day-panel.csv")
    return processed if processed.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse saved Google Ads change history HTML into budget and keyword CSVs."
    )
    parser.add_argument(
        "--html-file",
        type=Path,
        default=None,
        help="Saved Google Ads change history HTML under sys_think/data/.",
    )
    parser.add_argument(
        "--campaign-summary-output",
        type=Path,
        default=None,
        help="Campaign-summary CSV. Defaults to sys_think/data/processed/campaign-summary.csv.",
    )
    parser.add_argument(
        "--keyword-sets-output",
        type=Path,
        default=None,
        help="Keyword-set CSV. Defaults to sys_think/data/processed/campaign-keyword-sets.csv.",
    )
    parser.add_argument(
        "--kw-day-panel",
        type=Path,
        default=None,
        help="Processed kw-day-panel CSV.",
    )
    args = parser.parse_args()

    processed_dir = PROCESSED_DIR

    html_file = find_change_history_html(args.html_file)
    campaign_summary_output = args.campaign_summary_output or (
        processed_dir / "campaign-summary.csv"
    )
    keyword_sets_output = args.keyword_sets_output or (
        processed_dir / "campaign-keyword-sets.csv"
    )
    keyword_panel = resolve_keyword_panel(args.kw_day_panel)

    print(f"HTML: {html_file}", file=sys.stderr)
    print(f"Campaign summary: {campaign_summary_output}", file=sys.stderr)
    print(f"Keyword sets: {keyword_sets_output}", file=sys.stderr)
    if keyword_panel:
        print(f"Keyword panel: {keyword_panel}", file=sys.stderr)

    actions = decode_change_history_actions(html_file)

    if keyword_panel:
        summary_rows = raw_campaign_summary_rows(actions, keyword_panel)
    else:
        print(
            "[Warn] processed/kw-day-panel.csv not found; run process_input_data.py first "
            "to fill keyword inventory from the API panel.",
            file=sys.stderr,
        )
        summary_rows = campaign_summary_rows(actions)
    summary_rows, keyword_set_rows = assign_keyword_set_ids(summary_rows)
    write_csv(
        summary_rows,
        campaign_summary_output,
        [
            "campaign_version",
            "keyword_set_id",
            "campaign",
            "start_date",
            "end_date",
            "change_type",
            "daily_budget",
            "match_types",
            "num_unique_keywords",
            "broad_keyword_count",
            "phrase_keyword_count",
            "exact_keyword_count",
            "num_negative_keywords",
        ],
    )
    write_csv(
        keyword_set_rows,
        keyword_sets_output,
        [
            "keyword_set_id",
            "broad_keywords",
            "phrase_keywords",
            "exact_keywords",
            "unique_keywords",
            "negative_broad_keywords",
            "negative_phrase_keywords",
            "negative_exact_keywords",
            "negative_keywords",
            "used_by_campaign_versions",
        ],
    )

    print(
        f"Parsed {len(actions)} change-history action(s).",
        file=sys.stderr,
    )
    print(f"Wrote {len(summary_rows)} campaign-summary row(s).", file=sys.stderr)
    print(f"Wrote {len(keyword_set_rows)} campaign keyword-set row(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
