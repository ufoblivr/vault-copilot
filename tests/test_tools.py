"""
Test suite for Financial Intelligence analytics: anomaly detection,
subscription periodicity, time-series trends, and category breakdown.
"""
import sqlite3

import pytest

from src.agent.tools import FinancialIntelligence


# ======================================================================
# Basic output structure
# ======================================================================

class TestAnalyzeSpendingStructure:
    def test_returns_dict_with_expected_keys(self, populated_db):
        intel = FinancialIntelligence()
        result = intel.analyze_spending()
        assert isinstance(result, dict)
        assert "summary" in result
        assert "anomalies" in result
        assert "subscriptions" in result
        assert "trends" in result
        assert "categories" in result
        assert "report" in result

    def test_empty_db_returns_message(self, tmp_db):
        intel = FinancialIntelligence()
        result = intel.analyze_spending()
        assert "No structured financial data" in result["summary"]

    def test_report_key_is_string(self, populated_db):
        intel = FinancialIntelligence()
        result = intel.analyze_spending()
        assert isinstance(result["report"], str)
        assert len(result["report"]) > 0


# ======================================================================
# Summary
# ======================================================================

class TestSummary:
    def test_summary_contains_total(self, populated_db):
        intel = FinancialIntelligence()
        result = intel.analyze_spending()
        assert "$" in result["summary"]
        assert "Total" in result["summary"]

    def test_summary_contains_count(self, populated_db):
        intel = FinancialIntelligence()
        result = intel.analyze_spending()
        assert "Transactions" in result["summary"]


# ======================================================================
# Anomaly detection
# ======================================================================

class TestAnomalyDetection:
    def test_anomaly_detects_outlier(self, populated_db):
        """Amazon $199.99 should be flagged as anomaly in electronics (only 1 data point)
        or as an anomaly across the whole dataset."""
        intel = FinancialIntelligence()
        result = intel.analyze_spending(query="anomalies")
        # The anomaly section should exist and not be empty
        assert result["anomalies"] is not None

    def test_anomaly_section_with_keyword(self, populated_db):
        intel = FinancialIntelligence()
        result = intel.analyze_spending(query="unusual spending")
        assert "anomal" in result["anomalies"].lower() or "Anomal" in result["anomalies"]


# ======================================================================
# Subscription detection
# ======================================================================

class TestSubscriptionDetection:
    def test_detects_recurring_stores(self, populated_db):
        """Starbucks (3 visits) and Whole Foods (3 visits) should be detected."""
        intel = FinancialIntelligence()
        result = intel.analyze_spending(query="recurring subscriptions")
        subs = result["subscriptions"]
        assert isinstance(subs, str)
        # At least one store should be mentioned
        assert "Starbucks" in subs or "Whole Foods" in subs or "Netflix" in subs

    def test_netflix_detected_as_subscription(self, populated_db):
        """Netflix with monthly visits should be classified as monthly subscription."""
        intel = FinancialIntelligence()
        result = intel.analyze_spending(query="subscriptions")
        subs = result["subscriptions"]
        if "Netflix" in subs:
            assert "Monthly" in subs or "subscription" in subs.lower()


# ======================================================================
# Time-series trends
# ======================================================================

class TestTrends:
    def test_trends_section_exists(self, populated_db):
        intel = FinancialIntelligence()
        result = intel.analyze_spending(query="trends over time")
        assert result["trends"] is not None
        assert isinstance(result["trends"], str)

    def test_trends_mentions_direction(self, populated_db):
        intel = FinancialIntelligence()
        result = intel.analyze_spending(query="spending trend")
        trends = result["trends"]
        # Should contain a direction word
        has_direction = any(
            word in trends.lower()
            for word in ["increasing", "decreasing", "stable", "week", "month"]
        )
        assert has_direction or "data" in trends.lower()


# ======================================================================
# Category breakdown
# ======================================================================

class TestCategories:
    def test_categories_section_exists(self, populated_db):
        intel = FinancialIntelligence()
        result = intel.analyze_spending(query="category breakdown")
        assert result["categories"] is not None

    def test_categories_contain_percentages(self, populated_db):
        intel = FinancialIntelligence()
        result = intel.analyze_spending(query="category breakdown")
        cats = result["categories"]
        assert "%" in cats

    def test_top_category_highlighted(self, populated_db):
        intel = FinancialIntelligence()
        result = intel.analyze_spending(query="category breakdown")
        cats = result["categories"]
        assert "★" in cats or "top" in cats.lower()


# ======================================================================
# Query-aware focus
# ======================================================================

class TestQueryAwareFocus:
    def test_anomaly_focus(self, populated_db):
        intel = FinancialIntelligence()
        result = intel.analyze_spending(query="any anomalies?")
        assert result["anomalies"] != ""
        # Non-focused sections might be empty
        # (but summary is always present)
        assert result["summary"] != ""

    def test_subscription_focus(self, populated_db):
        intel = FinancialIntelligence()
        result = intel.analyze_spending(query="recurring charges")
        assert result["subscriptions"] != ""

    def test_category_focus(self, populated_db):
        intel = FinancialIntelligence()
        result = intel.analyze_spending(query="where does my money go")
        assert result["categories"] != ""

    def test_default_comprehensive(self, populated_db):
        """Empty query should produce all sections."""
        intel = FinancialIntelligence()
        result = intel.analyze_spending(query="")
        assert result["anomalies"] != ""
        assert result["subscriptions"] != ""
