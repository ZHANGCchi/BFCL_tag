from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .turns import DIALOGUE_MULTI, DIALOGUE_SINGLE

LLM_BASE = "<Base>"
LLM_MISSING_PARAM_MULTI = "<缺少相关参数>"
LLM_MISSING_TOOL_MULTI = "<缺少所需工具>"
LLM_MISSING_PARAM_SINGLE = "<幻觉：缺少相关参数>"
LLM_MISSING_TOOL_SINGLE = "<幻觉：缺少所需工具>"


@dataclass(slots=True)
class SemanticLLMResult:
    missing_parameters: bool
    missing_tools: bool

    def resolve_label(self, dialogue_type: str) -> str | None:
        if dialogue_type == DIALOGUE_MULTI:
            if self.missing_parameters:
                return LLM_MISSING_PARAM_MULTI
            if self.missing_tools:
                return LLM_MISSING_TOOL_MULTI
            return LLM_BASE
        if dialogue_type == DIALOGUE_SINGLE:
            if self.missing_parameters:
                return LLM_MISSING_PARAM_SINGLE
            if self.missing_tools:
                return LLM_MISSING_TOOL_SINGLE
        return None


class SemanticJudge:
    """Routes turn-level semantic labeling to an LLM or heuristic judge."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @staticmethod
    def _find_last_assistant(turn_messages: Iterable[Dict[str, Any]]) -> Dict[str, Any] | None:
        last_msg: Dict[str, Any] | None = None
        for message in turn_messages:
            if message.get("role") == "assistant":
                last_msg = message
        return last_msg

    def _call_llm(
        self,
        turn_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None,
        content: str,
        context: Dict[str, Any] | None = None,
    ) -> SemanticLLMResult:
        prompt = {
            "assistant_content": content,
        }
        response = self._client(prompt, context=context)
        if not isinstance(response, dict):
            raise ValueError("semantic LLM client must return a dict")
        missing_parameters = bool(response.get("missing_parameters"))
        missing_tools = bool(response.get("missing_tools"))
        return SemanticLLMResult(missing_parameters, missing_tools)

    def judge(
        self,
        turn_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None,
        dialogue_type: str,
        context: Dict[str, Any] | None = None,
    ) -> str | None:
        last_assistant = self._find_last_assistant(turn_messages)
        if not last_assistant:
            return None
        if last_assistant.get("tool_calls"):
            return None

        content = last_assistant.get("content")
        if not isinstance(content, str) or not content.strip():
            return None

        llm_result = self._call_llm(turn_messages, tools, content, context=context)
        return llm_result.resolve_label(dialogue_type)
