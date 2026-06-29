"""
Multi-step ReAct Agent with genuine reasoning, structured tool selection,
conversation memory, and execution tracing.
"""
import json
import time
import operator
from typing import Annotated, Literal, TypedDict, Optional
from dataclasses import dataclass, field, asdict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from transformers import pipeline as hf_pipeline
from pydantic import BaseModel, ValidationError
from loguru import logger
import torch

from src.config import (
    LLM_MODEL, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE,
    MAX_AGENT_STEPS,
)
from src.memory.vector_db import HybridRAGEngine
from src.memory.sqlite_db import execute_sql
from src.agent.tools import FinancialIntelligence


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------
class ToolDecision(BaseModel):
    """Structured schema for the supervisor's tool selection."""
    reasoning: str
    tool: Literal["SQL", "RAG", "INTEL", "DONE"]
    tool_input: str = ""


@dataclass
class ExecutionStep:
    """Records one step of agent execution for tracing."""
    step_number: int
    tool_selected: str
    tool_input: str
    tool_output: str
    reasoning: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]       # Full conversation history
    tool_results: Annotated[list, operator.add]   # All tool observations (list)
    steps_taken: int                              # Loop counter for safety
    execution_trace: Annotated[list, operator.add] # Trace of each step
    session_id: str                               # For conversation persistence


# ---------------------------------------------------------------------------
# The Copilot
# ---------------------------------------------------------------------------
class FinancialCopilot:
    def __init__(self):
        logger.info("Booting Financial Copilot Engine")
        self.rag = HybridRAGEngine()
        self.intel = FinancialIntelligence()

        # Load the LLM
        logger.info("Loading LLM: {}", LLM_MODEL)
        self.pipe = hf_pipeline(
            "text-generation",
            model=LLM_MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
        )

        # Build the graph with checkpointing for conversation persistence
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        logger.info("Financial Copilot ready")

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build_graph(self):
        """Constructs the multi-step ReAct state machine."""
        workflow = StateGraph(AgentState)

        workflow.add_node("supervisor", self.supervisor_node)
        workflow.add_node("sql_tool", self.sql_tool_node)
        workflow.add_node("rag_tool", self.rag_tool_node)
        workflow.add_node("intel_tool", self.intel_tool_node)

        def route(state: AgentState) -> str:
            last_message = state["messages"][-1]

            # Safety: hard cap on steps to prevent infinite loops
            if state.get("steps_taken", 0) >= MAX_AGENT_STEPS:
                logger.warning(
                    "Max agent steps ({}) reached, forcing synthesis",
                    MAX_AGENT_STEPS,
                )
                return END

            if isinstance(last_message, str):
                if "<TOOL:SQL>" in last_message:
                    return "sql_tool"
                if "<TOOL:RAG>" in last_message:
                    return "rag_tool"
                if "<TOOL:INTEL>" in last_message:
                    return "intel_tool"
            return END

        workflow.add_conditional_edges("supervisor", route)
        workflow.add_edge("sql_tool", "supervisor")
        workflow.add_edge("rag_tool", "supervisor")
        workflow.add_edge("intel_tool", "supervisor")

        workflow.set_entry_point("supervisor")
        return workflow.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # LLM prompting helpers
    # ------------------------------------------------------------------
    def _build_chat_prompt(self, system: str, user: str) -> str:
        """Build a Qwen-format chat prompt. Centralizes template logic."""
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Run the LLM and return the generated text."""
        result = self.pipe(
            prompt,
            max_new_tokens=max_tokens,
            return_full_text=False,
        )
        return result[0]["generated_text"].strip()

    def _parse_tool_decision(self, response: str) -> ToolDecision:
        """
        Parse the LLM's tool selection output into a structured ToolDecision.
        Uses JSON parsing first, falls back to keyword matching.
        """
        # Strategy 1: Try JSON parsing
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return ToolDecision(**data)
        except (json.JSONDecodeError, ValidationError):
            pass

        # Strategy 2: Keyword fallback (robust)
        upper = response.upper()
        if "SQL" in upper:
            tool = "SQL"
        elif "INTEL" in upper:
            tool = "INTEL"
        elif "RAG" in upper:
            tool = "RAG"
        elif "DONE" in upper:
            tool = "DONE"
        else:
            tool = "RAG"  # Default to RAG for unknown queries

        return ToolDecision(
            reasoning=response[:200],
            tool=tool,
            tool_input="",
        )

    # ------------------------------------------------------------------
    # Supervisor node — multi-step ReAct
    # ------------------------------------------------------------------
    def supervisor_node(self, state: AgentState) -> dict:
        """
        The core reasoning engine. Supports multi-step reasoning:
        1. If no tool results yet → plan and select first tool
        2. If tool results exist but more info needed → select next tool
        3. If enough info or max steps → synthesize final answer
        """
        query = state["messages"][0]
        tool_results = state.get("tool_results", [])
        steps = state.get("steps_taken", 0)

        start = time.perf_counter()

        if not tool_results:
            # ── Phase 1: Initial planning & tool selection ──
            system_prompt = (
                "You are a financial AI planning agent. Analyze the user's query and select the best tool.\n"
                "Respond with JSON: {\"reasoning\": \"...\", \"tool\": \"SQL|RAG|INTEL\", \"tool_input\": \"...\"}\n\n"
                "Tools:\n"
                "- SQL: For numerical queries (totals, sums, counts, averages, specific amounts)\n"
                "- RAG: For searching specific items, products, stores, or receipt details\n"
                "- INTEL: For spending analysis, anomalies, trends, subscriptions, category breakdowns\n"
            )
            response = self._generate(
                self._build_chat_prompt(system_prompt, query),
                max_tokens=100,
            )
            decision = self._parse_tool_decision(response)
            logger.info(
                "Step {}: tool={}, reasoning={}",
                steps + 1, decision.tool, decision.reasoning[:100],
            )

            if decision.tool == "DONE":
                return {"messages": ["I need more context to help you. Could you rephrase your question?"]}

            trace_step = ExecutionStep(
                step_number=steps + 1,
                tool_selected=decision.tool,
                tool_input=decision.tool_input or query,
                tool_output="",  # Filled by tool node
                reasoning=decision.reasoning,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
            return {
                "messages": [f"<TOOL:{decision.tool}>"],
                "steps_taken": steps + 1,
                "execution_trace": [asdict(trace_step)],
            }
        else:
            # ── Phase 2+: Decide — chain another tool or synthesize ──
            observations = "\n---\n".join(
                f"[Tool Result {i+1}]: {r}" for i, r in enumerate(tool_results)
            )

            if steps < MAX_AGENT_STEPS:
                # Ask the LLM if it needs another tool
                system_prompt = (
                    "You are a financial AI agent. You have gathered some information.\n"
                    "Decide: do you need another tool, or can you answer the user?\n\n"
                    f"Information gathered so far:\n{observations}\n\n"
                    "If you need more info, respond with JSON: "
                    "{\"reasoning\": \"...\", \"tool\": \"SQL|RAG|INTEL\", \"tool_input\": \"...\"}\n"
                    "If you can answer, respond with JSON: {\"reasoning\": \"...\", \"tool\": \"DONE\"}\n"
                )
                response = self._generate(
                    self._build_chat_prompt(system_prompt, query),
                    max_tokens=100,
                )
                decision = self._parse_tool_decision(response)

                if decision.tool != "DONE":
                    logger.info(
                        "Step {}: chaining tool={}, reasoning={}",
                        steps + 1, decision.tool, decision.reasoning[:100],
                    )
                    trace_step = ExecutionStep(
                        step_number=steps + 1,
                        tool_selected=decision.tool,
                        tool_input=decision.tool_input or query,
                        tool_output="",
                        reasoning=decision.reasoning,
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )
                    return {
                        "messages": [f"<TOOL:{decision.tool}>"],
                        "steps_taken": steps + 1,
                        "execution_trace": [asdict(trace_step)],
                    }

            # ── Final synthesis ──
            logger.info("Synthesizing final response from {} tool results", len(tool_results))
            system_prompt = (
                "You are a professional financial AI copilot. "
                "Synthesize a clear, helpful, and well-structured answer "
                "using ONLY the following tool observations. "
                "Be specific with numbers and details.\n\n"
                f"Tool Observations:\n{observations}"
            )
            response = self._generate(
                self._build_chat_prompt(system_prompt, query),
                max_tokens=LLM_MAX_NEW_TOKENS,
            )
            latency = (time.perf_counter() - start) * 1000
            logger.info("Final synthesis complete ({:.0f}ms)", latency)
            return {"messages": [response]}

    # ------------------------------------------------------------------
    # Tool nodes
    # ------------------------------------------------------------------
    def sql_tool_node(self, state: AgentState) -> dict:
        """Generates and executes SQL through AST-validated safety guardrail."""
        query = state["messages"][0]
        start = time.perf_counter()

        sql_prompt = self._build_chat_prompt(
            "You are a SQLite query generator. "
            "Table 'receipts' has columns: id, store, date, total, category. "
            "Table 'items' has columns: id, receipt_id, name. "
            "Write a single SELECT query. Use LIKE '%word%' for string matching. "
            "Output ONLY the SQL query, nothing else.",
            query,
        )
        # Seed with "SELECT" to guide generation
        raw = self._generate(sql_prompt + "SELECT ", max_tokens=100)
        sql_query = "SELECT " + raw.split(";")[0].replace("```sql", "").replace("```", "").strip()

        try:
            result = execute_sql(sql_query)
            obs = f"SQL Query: {sql_query}\nDatabase returned: {result}"
        except ValueError as e:
            obs = f"SQL Guardrail blocked query: {e}"
            logger.warning("SQL guardrail triggered: {}", e)
        except Exception as e:
            obs = f"SQL execution failed: {e}"
            logger.error("SQL execution error: {}", e)

        latency = (time.perf_counter() - start) * 1000
        logger.info("SQL tool completed ({:.0f}ms): {}", latency, sql_query[:100])
        return {"tool_results": [obs]}

    def rag_tool_node(self, state: AgentState) -> dict:
        """Performs hybrid search with re-ranking."""
        query = state["messages"][0]
        start = time.perf_counter()

        results = self.rag.hybrid_search(query)
        if isinstance(results, list) and results:
            if hasattr(results[0], "document"):
                # SearchResult objects
                docs_text = "\n".join(
                    f"- [{r.source}] (score: {r.score:.3f}) {r.document}"
                    for r in results
                )
            else:
                docs_text = "\n".join(f"- {r}" for r in results)
        else:
            docs_text = "No relevant receipts found in memory."

        obs = f"Retrieved Memories:\n{docs_text}"
        latency = (time.perf_counter() - start) * 1000
        logger.info("RAG tool completed ({:.0f}ms): {} results", latency, len(results))
        return {"tool_results": [obs]}

    def intel_tool_node(self, state: AgentState) -> dict:
        """Runs financial analytics with query-aware focus."""
        query = state["messages"][0]
        start = time.perf_counter()

        analysis = self.intel.analyze_spending(query)
        if isinstance(analysis, dict):
            obs = analysis.get("report", str(analysis))
        else:
            obs = str(analysis)

        latency = (time.perf_counter() - start) * 1000
        logger.info("INTEL tool completed ({:.0f}ms)", latency)
        return {"tool_results": [obs]}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(self, query: str, session_id: str = "default") -> dict:
        """
        Public interface for invoking the agent.
        Returns the response text and execution trace.
        """
        config = {"configurable": {"thread_id": session_id}}
        initial_state = {
            "messages": [query],
            "tool_results": [],
            "steps_taken": 0,
            "execution_trace": [],
            "session_id": session_id,
        }

        start = time.perf_counter()
        final_state = self.graph.invoke(initial_state, config=config)
        total_latency = (time.perf_counter() - start) * 1000

        response = final_state["messages"][-1] if final_state["messages"] else "Unable to process query."
        trace = final_state.get("execution_trace", [])

        logger.info(
            "Chat completed: session={}, steps={}, latency={:.0f}ms",
            session_id, len(trace), total_latency,
        )

        return {
            "response": response,
            "execution_trace": trace,
            "total_latency_ms": total_latency,
            "steps_taken": len(trace),
        }