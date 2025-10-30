from __future__ import annotations

from typing import Any, Dict, List, MutableMapping

STRUCTURAL_NO_CALL = "<无工具调用>"
STRUCTURAL_SINGLE_TOOL_SINGLE_CALL = "<单工具单调用>"
STRUCTURAL_MULTI_TOOL_SINGLE_CALL = "<多工具单调用>"
STRUCTURAL_SINGLE_TOOL_MULTI_CALL = "<单工具多调用>"
STRUCTURAL_MULTI_TOOL_MULTI_CALL = "<多工具多调用>"


def _extract_tool_name(call: MutableMapping[str, Any]) -> str | None:
    """Try to extract the function name from a tool call payload."""
    if not isinstance(call, MutableMapping):
        return None
    function_payload = call.get("function")
    if isinstance(function_payload, MutableMapping):
        name = function_payload.get("name")
        if isinstance(name, str) and name:
            return name
    name = call.get("name")
    if isinstance(name, str) and name:
        return name
    return None


def analyse_structural_label(
    turn_messages: List[Dict[str, Any]],
    available_tools: List[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    """Return structural labeling metadata for a dialogue turn."""
    all_calls: List[MutableMapping[str, Any]] = []
    for message in turn_messages:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, MutableMapping):
                    all_calls.append(call)

    total_calls = len(all_calls)
    tool_names = [name for name in (_extract_tool_name(call) for call in all_calls) if name]
    unique_names = len(set(tool_names))
    available_tool_count = len(available_tools or [])

    if total_calls == 0:
        label = STRUCTURAL_NO_CALL
    elif total_calls == 1:
        label = (
            STRUCTURAL_MULTI_TOOL_SINGLE_CALL
            if available_tool_count > 1
            else STRUCTURAL_SINGLE_TOOL_SINGLE_CALL
        )
    else:
        label = (
            STRUCTURAL_SINGLE_TOOL_MULTI_CALL
            if unique_names == 1
            else STRUCTURAL_MULTI_TOOL_MULTI_CALL
        )

    return {
        "label": label,
        "total_calls": total_calls,
        "unique_tool_count": unique_names,
        "tool_names": tool_names,
        "available_tool_count": available_tool_count,
    }
