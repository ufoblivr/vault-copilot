"""
Test suite for the multi-step ReAct agent: structured tool selection,
parsing, and chat interface.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.agent.graph import ToolDecision


# ======================================================================
# ToolDecision Pydantic model
# ======================================================================

class TestToolDecision:
    def test_valid_sql_decision(self):
        d = ToolDecision(reasoning="Need totals", tool="SQL")
        assert d.tool == "SQL"
        assert d.tool_input == ""

    def test_valid_rag_decision(self):
        d = ToolDecision(reasoning="Find items", tool="RAG", tool_input="milk")
        assert d.tool == "RAG"
        assert d.tool_input == "milk"

    def test_valid_intel_decision(self):
        d = ToolDecision(reasoning="Anomalies", tool="INTEL")
        assert d.tool == "INTEL"

    def test_valid_done_decision(self):
        d = ToolDecision(reasoning="Have enough info", tool="DONE")
        assert d.tool == "DONE"

    def test_invalid_tool_rejected(self):
        with pytest.raises(ValidationError):
            ToolDecision(reasoning="test", tool="INVALID_TOOL")

    def test_from_json_string(self):
        data = json.loads('{"reasoning": "Need SQL", "tool": "SQL", "tool_input": "totals"}')
        d = ToolDecision(**data)
        assert d.tool == "SQL"


# ======================================================================
# _parse_tool_decision (requires FinancialCopilot instance)
# ======================================================================

class TestParseToolDecision:
    """Test the parsing logic by instantiating the method directly."""

    @pytest.fixture
    def copilot_parser(self):
        """Return just the parser method without loading real models."""
        from src.agent.graph import FinancialCopilot
        # Get the unbound method — we don't need a full instance
        return FinancialCopilot._parse_tool_decision

    def test_parse_valid_json(self, copilot_parser):
        response = '{"reasoning": "User wants totals", "tool": "SQL", "tool_input": ""}'
        # Call as static-like (pass None as self since it doesn't use self state)
        result = copilot_parser(None, response)
        assert result.tool == "SQL"

    def test_parse_json_with_surrounding_text(self, copilot_parser):
        response = 'I think we need {"reasoning": "Check data", "tool": "RAG"} for this'
        result = copilot_parser(None, response)
        assert result.tool == "RAG"

    def test_parse_keyword_fallback_sql(self, copilot_parser):
        response = "I think we should use the SQL tool to get the data"
        result = copilot_parser(None, response)
        assert result.tool == "SQL"

    def test_parse_keyword_fallback_intel(self, copilot_parser):
        response = "This needs INTEL analysis for anomalies"
        result = copilot_parser(None, response)
        assert result.tool == "INTEL"

    def test_parse_keyword_fallback_rag(self, copilot_parser):
        response = "Let me search using RAG"
        result = copilot_parser(None, response)
        assert result.tool == "RAG"

    def test_parse_keyword_fallback_done(self, copilot_parser):
        response = "DONE, I have enough information"
        result = copilot_parser(None, response)
        assert result.tool == "DONE"

    def test_parse_unknown_defaults_to_rag(self, copilot_parser):
        response = "I'm not sure what to do here"
        result = copilot_parser(None, response)
        assert result.tool == "RAG"


# ======================================================================
# Chat output structure (with fully mocked copilot)
# ======================================================================

class TestChatOutputStructure:
    """Verify the chat() method returns the expected dict structure."""

    def test_chat_returns_expected_keys(self):
        """Mock the entire copilot to test only the return structure."""
        with patch("src.agent.graph.FinancialCopilot.__init__", return_value=None):
            from src.agent.graph import FinancialCopilot
            copilot = FinancialCopilot.__new__(FinancialCopilot)

            # Mock the graph
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "messages": ["test query", "Here is your answer"],
                "tool_results": ["some data"],
                "execution_trace": [
                    {"step_number": 1, "tool_selected": "SQL",
                     "tool_input": "", "tool_output": "",
                     "reasoning": "test", "latency_ms": 50}
                ],
                "steps_taken": 1,
                "session_id": "test",
            }
            copilot.graph = mock_graph
            copilot.checkpointer = MagicMock()

            result = copilot.chat("test query", session_id="test-session")

            assert "response" in result
            assert "execution_trace" in result
            assert "total_latency_ms" in result
            assert "steps_taken" in result
            assert isinstance(result["response"], str)
            assert isinstance(result["execution_trace"], list)
            assert isinstance(result["total_latency_ms"], float)
