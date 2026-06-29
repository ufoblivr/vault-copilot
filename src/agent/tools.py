"""
Financial intelligence tool for spending analysis.

Provides query-aware anomaly detection, subscription/periodicity recognition,
time-series trend analysis, and category breakdowns over receipt data stored
in SQLite.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import DB_PATH

# ---------------------------------------------------------------------------
# Keyword groups used for query-aware focus
# ---------------------------------------------------------------------------
_ANOMALY_KEYWORDS: tuple[str, ...] = ("anomal", "unusual", "outlier")
_SUBSCRIPTION_KEYWORDS: tuple[str, ...] = ("subscri", "recurring", "repeat")
_TREND_KEYWORDS: tuple[str, ...] = ("trend", "over time", "month", "week")
_CATEGORY_KEYWORDS: tuple[str, ...] = ("categor", "breakdown", "where")

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_ZSCORE_THRESHOLD: float = 2.0
_WEEKLY_INTERVAL_CENTRE: int = 7
_WEEKLY_INTERVAL_TOLERANCE: int = 3
_MONTHLY_INTERVAL_CENTRE: int = 30
_MONTHLY_INTERVAL_TOLERANCE: int = 7
_MIN_VISITS_FOR_SUBSCRIPTION: int = 2


def _query_matches(query: str, keywords: tuple[str, ...]) -> bool:
    """Return ``True`` if *query* contains any of the *keywords* (case-insensitive)."""
    lowered = query.lower()
    return any(kw in lowered for kw in keywords)


class FinancialIntelligence:
    """Analyse structured receipt data for anomalies, subscriptions, trends, and categories."""

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def analyze_spending(self, query: str = "") -> dict[str, Any]:
        """Run spending analysis, optionally focused by *query* keywords.

        Parameters
        ----------
        query:
            The end-user's question.  Keywords in the query determine which
            analysis sections receive the most detail.  When empty, a
            comprehensive report covering every section is returned.

        Returns
        -------
        dict[str, Any]
            Keys: ``summary``, ``anomalies``, ``subscriptions``, ``trends``,
            ``categories``, and ``report`` (concatenated string of all
            sections for backward compatibility).
        """
        logger.info("Starting spending analysis (query={!r})", query)

        df = self._load_receipts()
        if df is None or df.empty:
            empty_msg = "No structured financial data available yet. Please upload receipts."
            logger.warning(empty_msg)
            return self._empty_result(empty_msg)

        # Parse dates once for all downstream consumers
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Determine which sections the user is interested in
        focused = self._resolve_focus(query)

        # Build each section
        summary = self._build_summary(df)
        anomalies = self._build_anomalies(df) if "anomalies" in focused else ""
        subscriptions = self._build_subscriptions(df) if "subscriptions" in focused else ""
        trends = self._build_trends(df) if "trends" in focused else ""
        categories = self._build_categories(df) if "categories" in focused else ""

        report = self._assemble_report(summary, anomalies, subscriptions, trends, categories)
        logger.info("Spending analysis complete ({} chars)", len(report))

        return {
            "summary": summary,
            "anomalies": anomalies,
            "subscriptions": subscriptions,
            "trends": trends,
            "categories": categories,
            "report": report,
        }

    # ------------------------------------------------------------------ #
    #  Data loading                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_receipts() -> pd.DataFrame | None:
        """Load receipts from SQLite using a **read-only** connection.

        Returns ``None`` when the table does not exist or a database error
        occurs.
        """
        uri = f"file:{DB_PATH}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True)
            try:
                df = pd.read_sql_query(
                    "SELECT store, date, total, category FROM receipts", conn
                )
            finally:
                conn.close()
            logger.debug("Loaded {} receipt rows from DB", len(df))
            return df
        except sqlite3.OperationalError as exc:
            logger.error("Failed to read receipts table: {}", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected DB error: {}", exc)
            return None

    # ------------------------------------------------------------------ #
    #  Focus resolution                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_focus(query: str) -> set[str]:
        """Map user *query* keywords to the set of sections to include.

        When no keyword matches (or *query* is empty), all sections are
        returned for a comprehensive report.
        """
        sections: set[str] = set()
        if _query_matches(query, _ANOMALY_KEYWORDS):
            sections.add("anomalies")
        if _query_matches(query, _SUBSCRIPTION_KEYWORDS):
            sections.add("subscriptions")
        if _query_matches(query, _TREND_KEYWORDS):
            sections.add("trends")
        if _query_matches(query, _CATEGORY_KEYWORDS):
            sections.add("categories")

        if not sections:
            # Default: comprehensive
            sections = {"anomalies", "subscriptions", "trends", "categories"}

        logger.debug("Analysis focus sections: {}", sections)
        return sections

    # ------------------------------------------------------------------ #
    #  Section builders                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_summary(df: pd.DataFrame) -> str:
        """Return a high-level summary line."""
        total = df["total"].sum()
        avg = df["total"].mean()
        count = len(df)
        return (
            f"Overall Total Spend: ${total:,.2f}. "
            f"Transactions: {count}. "
            f"Average Transaction: ${avg:,.2f}."
        )

    # -- anomalies ---------------------------------------------------- #

    @staticmethod
    def _build_anomalies(df: pd.DataFrame) -> str:
        """Per-category z-score anomaly detection.

        Flags transactions whose z-score within their category exceeds
        ``_ZSCORE_THRESHOLD``.
        """
        if "category" not in df.columns or df["total"].isna().all():
            return ""

        results: list[str] = []

        for category, group in df.groupby("category", dropna=True):
            if len(group) < 2:
                continue  # need at least 2 observations for std dev
            mean = group["total"].mean()
            std = group["total"].std(ddof=1)
            if std == 0 or pd.isna(std):
                continue
            group = group.copy()
            group["zscore"] = (group["total"] - mean) / std
            outliers = group[group["zscore"] > _ZSCORE_THRESHOLD]
            for _, row in outliers.iterrows():
                results.append(
                    f"  • {row['store']}: ${row['total']:,.2f} in '{category}' "
                    f"({row['zscore']:.1f} std devs above category avg of ${mean:,.2f})"
                )

        if not results:
            return "Anomalies: No unusual transactions detected."

        header = f"Anomalies: {len(results)} unusually large transaction(s) detected."
        return header + "\n" + "\n".join(results)

    # -- subscriptions ------------------------------------------------ #

    @staticmethod
    def _build_subscriptions(df: pd.DataFrame) -> str:
        """Detect recurring visits and classify periodicity.

        Stores visited more than once are analysed for interval regularity:
        * **Weekly** – intervals cluster around 7 ± 3 days.
        * **Monthly** – intervals cluster around 30 ± 7 days.
        * **Frequent store** – visited often but with irregular intervals.
        """
        if df["date"].isna().all():
            return "Subscriptions: Not enough date information to detect periodicity."

        valid = df.dropna(subset=["date"]).sort_values("date")
        store_groups = valid.groupby("store")

        entries: list[str] = []
        for store, group in store_groups:
            if len(group) < _MIN_VISITS_FOR_SUBSCRIPTION:
                continue
            dates = group["date"].sort_values()
            intervals = dates.diff().dropna().dt.days.values

            if len(intervals) == 0:
                continue

            median_interval = float(np.median(intervals))
            label = _classify_interval(median_interval, intervals)
            entries.append(f"  • {store}: {label} ({len(group)} visits, ~{median_interval:.0f}-day interval)")

        if not entries:
            return "Subscriptions: No recurring store visits detected."

        return "Subscriptions:\n" + "\n".join(entries)

    # -- trends ------------------------------------------------------- #

    @staticmethod
    def _build_trends(df: pd.DataFrame) -> str:
        """Time-series spending grouped by week and month with trend direction."""
        if df["date"].isna().all():
            return "Trends: Not enough date information for time-series analysis."

        valid = df.dropna(subset=["date"]).copy()

        # --- Monthly ---
        valid["month"] = valid["date"].dt.to_period("M")
        monthly = valid.groupby("month")["total"].sum().sort_index()

        # --- Weekly ---
        valid["week"] = valid["date"].dt.to_period("W")
        weekly = valid.groupby("week")["total"].sum().sort_index()

        parts: list[str] = []

        if len(monthly) >= 2:
            last_month_val = float(monthly.iloc[-1])
            prior_avg = float(monthly.iloc[:-1].mean())
            direction = _trend_direction(last_month_val, prior_avg)
            parts.append(
                f"Monthly trend: Spending is {direction}. "
                f"Last month: ${last_month_val:,.2f} vs. prior-months average: ${prior_avg:,.2f}."
            )
        elif len(monthly) == 1:
            parts.append(f"Monthly trend: Only one month of data (${float(monthly.iloc[0]):,.2f}).")

        if len(weekly) >= 2:
            last_week_val = float(weekly.iloc[-1])
            prior_week_avg = float(weekly.iloc[:-1].mean())
            direction = _trend_direction(last_week_val, prior_week_avg)
            parts.append(
                f"Weekly trend: Spending is {direction}. "
                f"Last week: ${last_week_val:,.2f} vs. prior-weeks average: ${prior_week_avg:,.2f}."
            )
        elif len(weekly) == 1:
            parts.append(f"Weekly trend: Only one week of data (${float(weekly.iloc[0]):,.2f}).")

        if not parts:
            return "Trends: Insufficient data for trend analysis."

        return "Trends:\n" + "\n".join(f"  • {p}" for p in parts)

    # -- categories --------------------------------------------------- #

    @staticmethod
    def _build_categories(df: pd.DataFrame) -> str:
        """Category breakdown with totals and percentages."""
        if "category" not in df.columns or df["category"].isna().all():
            return "Categories: No category data available."

        total_spend = df["total"].sum()
        if total_spend == 0:
            return "Categories: Total spend is $0.00."

        cat_totals = (
            df.groupby("category", dropna=True)["total"]
            .sum()
            .sort_values(ascending=False)
        )

        lines: list[str] = []
        for idx, (cat, amount) in enumerate(cat_totals.items()):
            pct = (amount / total_spend) * 100
            prefix = "★ " if idx == 0 else "  "
            lines.append(f"  {prefix}{cat}: ${amount:,.2f} ({pct:.1f}%)")

        top_cat = cat_totals.index[0]
        header = f"Categories (top: {top_cat}):"
        return header + "\n" + "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _assemble_report(*sections: str) -> str:
        """Concatenate non-empty sections into a single report string."""
        return "\n\n".join(s for s in sections if s)

    @staticmethod
    def _empty_result(message: str) -> dict[str, Any]:
        """Return a result dict where every field carries *message*."""
        return {
            "summary": message,
            "anomalies": "",
            "subscriptions": "",
            "trends": "",
            "categories": "",
            "report": message,
        }


# ---------------------------------------------------------------------- #
#  Module-level helpers                                                    #
# ---------------------------------------------------------------------- #


def _classify_interval(median_interval: float, intervals: np.ndarray) -> str:
    """Classify a store's visit pattern as weekly, monthly, or frequent.

    Parameters
    ----------
    median_interval:
        Median number of days between consecutive visits.
    intervals:
        Array of all inter-visit intervals in days.
    """
    weekly_low = _WEEKLY_INTERVAL_CENTRE - _WEEKLY_INTERVAL_TOLERANCE
    weekly_high = _WEEKLY_INTERVAL_CENTRE + _WEEKLY_INTERVAL_TOLERANCE
    monthly_low = _MONTHLY_INTERVAL_CENTRE - _MONTHLY_INTERVAL_TOLERANCE
    monthly_high = _MONTHLY_INTERVAL_CENTRE + _MONTHLY_INTERVAL_TOLERANCE

    weekly_ratio = np.mean((intervals >= weekly_low) & (intervals <= weekly_high))
    monthly_ratio = np.mean((intervals >= monthly_low) & (intervals <= monthly_high))

    # A pattern is "regular" if ≥ 60 % of intervals fall into the window
    if weekly_ratio >= 0.6 and weekly_low <= median_interval <= weekly_high:
        return "Weekly subscription"
    if monthly_ratio >= 0.6 and monthly_low <= median_interval <= monthly_high:
        return "Monthly subscription"
    return "Frequent store (irregular)"


def _trend_direction(last_value: float, prior_avg: float) -> str:
    """Return a human-readable trend label comparing *last_value* to *prior_avg*.

    A ±10 % band around the prior average is considered "stable".
    """
    if prior_avg == 0:
        return "increasing" if last_value > 0 else "stable"
    ratio = last_value / prior_avg
    if ratio > 1.10:
        return "increasing"
    if ratio < 0.90:
        return "decreasing"
    return "stable"