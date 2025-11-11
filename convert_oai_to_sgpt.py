import json
import os
import copy
import uuid
from tqdm import tqdm
from typing import Any, Iterable, Iterator, List

def load_jsonl(path: str) -> Any:
    """Load JSON Lines data from a file.

    Args:
        path: Path to the JSONL file

    Returns:
        List of parsed JSON objects
    """
    with open(path, "r", encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(data: Iterable[Any], path: str) -> None:
    """Write data to a JSON Lines file.

    Args:
        data: Iterable of JSON-serializable objects
        path: Output file path
    """
    with open(path, "w", encoding="utf8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_multiturn_to_multi_sample(data: dict):
    messages = data["messages"]
    tools = data.get("tools")
    data_id = data.get("id", str(uuid.uuid4()))
    samples = []
    turn = 0
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and msg.get("loss", True):
            # remove prev reasoning_content
            input_messages = copy.deepcopy(messages[:i])
            for _msg in input_messages:
                if _msg.get("content"):
                    _msg["content"] = _msg["content"].strip()
                if "reasoning_content" in _msg:
                    del _msg["reasoning_content"]
            response = msg
            # import pdb
            # pdb.set_trace()
            formatted_data = convert_oai_to_sgpt(
                {
                    "id": data_id + f"_turn_{turn}",
                    "messages": input_messages + [response],
                    "tools": tools,
                },
            )
            turn += 1
            if formatted_data:
                samples.append(formatted_data)
    return samples


def convert_oai_to_sgpt(data: dict, use_think: bool = True) -> dict:
    messages = data["messages"]
    tools = data.get("tools", [])
    system_prompt = "\n".join(
        [msg["content"] for msg in messages if msg["role"] == "system"]
    ).strip()
    if tools:
        system_prompt += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
        system_prompt += "\n".join(
            [json.dumps(tool["function"], ensure_ascii=False) for tool in tools]
        )
        system_prompt += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    system_prompt = system_prompt.strip()

    conversations = [{"from": "system", "value": system_prompt}]
    input_messages = [msg for msg in messages[:-1] if msg["role"] != "system"]
    output_message = messages[-1]
    input_prompt = ""

    for i, msg in enumerate(input_messages):
        if msg["role"] == "user":
            if i != 0:
                input_prompt += "<|im_start|>user\n"
            input_prompt += msg["content"].strip()
            if i != len(input_messages) - 1:
                input_prompt += "<|im_end|>\n"
        elif msg["role"] == "tool":
            if i == 0 or input_messages[i - 1]["role"] != "tool":
                input_prompt += f"<|im_start|>user\n<tool_response>\n{msg['content'].strip()}\n</tool_response>"
            else:
                assert input_prompt.endswith("</tool_response>")
                input_prompt += (
                    f"\n<tool_response>\n{msg['content'].strip()}\n</tool_response>"
                )
        elif msg["role"] == "assistant":
            if not input_prompt.endswith("<|im_end|>\n"):
                input_prompt += "<|im_end|>\n"
            if msg.get("tool_calls"):
                input_prompt += "<|im_start|>assistant\n"
                for tool_call in msg["tool_calls"]:
                    input_prompt += f"<tool_call>\n{json.dumps(tool_call['function'], ensure_ascii=False)}\n</tool_call>\n"
                input_prompt = input_prompt.strip()
                input_prompt += "<|im_end|>\n"
            else:
                input_prompt += (
                    "<|im_start|>assistant\n" + msg["content"].strip() + "<|im_end|>\n"
                )

    if input_prompt.endswith("<|im_end|>\n"):
        print("input_prompt should not end with <|im_end|>\n")
        return None
    conversations.append(
        {
            "from": "human",
            "value": input_prompt,
        }
    )
    if not output_message.get("reasoning_content"):
        return None

    output_prompt = (
        "<think>\n" + output_message["reasoning_content"].strip() + "\n</think>\n\n"
    )
    if output_message.get("tool_calls"):
        for tool_call in output_message["tool_calls"]:
            output_prompt += f"<tool_call>\n{json.dumps(tool_call['function'], ensure_ascii=False)}\n</tool_call>\n"
        output_prompt = output_prompt.strip()
    else:
        output_prompt += output_message["content"].strip()
    conversations.append(
        {
            "from": "gpt",
            "value": output_prompt,
        }
    )
    return {"id": data.get("id", ""), "conversations": conversations}


def main(
    data_dir: str,
    output_dir: str,
    use_think: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(data_dir):
        for file in tqdm(files):
            if file.endswith(".jsonl"):
                oai_jsonl = os.path.join(root, file)
                sgpt_jsonl = os.path.join(output_dir, file)
                dataset = load_jsonl(oai_jsonl)
                output_dataset = sum(
                    [convert_multiturn_to_multi_sample(item) for item in dataset], []
                )
                write_jsonl(output_dataset, sgpt_jsonl)


if __name__ == "__main__":
    main(
        "data/llm_train_data/agent/agent_data_sft_251022_v1.2",
        "data/llm_train_data/agent/agent_data_sft_251022_v1.2_sgpt",
        use_think=True,
    )
