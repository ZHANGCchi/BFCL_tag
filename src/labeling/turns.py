from __future__ import annotations

from typing import Any, Dict, List


DIALOGUE_SINGLE = "Single-Turn"
DIALOGUE_MULTI = "Multi-Turn"


def detect_dialogue_type(messages: List[Dict[str, Any]]) -> str:
    """Classify a dialogue as single or multi turn based on user message count."""
    user_count = sum(1 for message in messages if message.get("role") == "user")
    return DIALOGUE_SINGLE if user_count <= 1 else DIALOGUE_MULTI


def split_turns(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split a message list into turn-sized slices using user messages as anchors."""
    turns: List[List[Dict[str, Any]]] = []
    current_turn: List[Dict[str, Any]] = []
    waiting_prefix: List[Dict[str, Any]] = []

    for message in messages:
        role = message.get("role")

        if role == "user":
            if current_turn:
                turns.append(current_turn)
                current_turn = []
            if waiting_prefix:
                current_turn.extend(waiting_prefix)
                waiting_prefix = []
            current_turn.append(message)
            continue

        if not current_turn:
            # Collect non-user content that appears before the next user message.
            waiting_prefix.append(message)
            continue

        current_turn.append(message)

    if current_turn:
        turns.append(current_turn)
    elif waiting_prefix and not turns:
        # Degenerate dialogues without explicit user input.
        turns.append(waiting_prefix)

    return turns
