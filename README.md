# 数据标注流程说明

本文档说明 `data/` 目录下对话数据的打标算法和项目结构，目标是按照用户消息（Turn）粒度生成结构化和语义化标签，同时给出标签分布可视化。

## 算法概览

1. **全局对话类型判定**
   - 扫描 `messages` 列表统计 `role == "user"` 的数量。
   - 若数量 ≤ 1，标记 `dialogue_type = "Single-Turn"`；否则标记为 `"Multi-Turn"`。

2. **Turn 拆分**
   - 以 `role == "user"` 的消息为锚点，将完整 `messages` 列表拆分成多个 Turn。
   - 每个 Turn 至少包含一条用户消息，并保留其后的 assistant/tool 回复。
   - 处理特殊情况：
     - 如果在第一个用户消息之前有非用户消息（如system消息），这些消息会被添加到第一个Turn中
     - 如果对话中完全没有用户消息，所有消息会被聚合成单一Turn
     - 每遇到一个新的用户消息，就会开始一个新的Turn

3. **结构化打标**
   - 针对 Turn 内所有 `role == "assistant"` 的消息，收集其中的 `tool_calls`。
   - 支持两种tool_calls格式：
     - 标准格式：`{"function": {"name": "function_name", ...}}`
     - 简化格式：`{"name": "function_name", ...}`
   - 统计总调用数 `total_calls` 与不同函数名数量 `unique_tool_count`，以及可用工具总数 `available_tool_count`。
   - 根据以下规则输出标签：
     - `total_calls == 0`：`<无工具调用>`
     - `total_calls == 1` 且 `available_tool_count > 1`：`<多工具单调用>`
     - `total_calls == 1` 且 `available_tool_count <= 1`：`<单工具单调用>`
     - `total_calls > 1` 且 `unique_tool_count == 1`：`<单工具多调用>`
     - `total_calls > 1` 且 `unique_tool_count > 1`：`<多工具多调用>`
   - 同时写入辅助统计信息（调用数、去重后的工具名、可用工具总数等），便于后续分析。

4. **语义化打标**
   - 寻找 Turn 内最后一条 `role == "assistant"` 的消息。
   - 跳过语义标签的条件：
     - 该消息包含 `tool_calls`（即仍然是工具调用）
     - 该消息的 `content` 字段不存在、不是字符串类型、或为空字符串
     - Turn 中没有找到任何 assistant 消息
   - 否则，将目标回复文本发送至 LLM（通过 `qwen3.py`）。
   - LLM 仅需返回布尔值：
     - `missing_parameters`: 是否缺少必要参数（助手机器人明确表示缺少所需的输入、字段、参数等）
     - `missing_tools`: 是否缺少可用工具（助手机器人明确表示可用工具集中缺少所需功能）
   - 根据全局 `dialogue_type` 转换为最终标签：
     - `Multi-Turn`：
       - `missing_parameters=True`：`<缺少相关参数>`
       - `missing_tools=True`：`<缺少所需工具>`
       - 否则：`<Base>`
     - `Single-Turn`：
       - `missing_parameters=True`：`<幻觉：缺少相关参数>`
       - `missing_tools=True`：`<幻觉：缺少所需工具>`
       - 否则：不打标（返回None）

5. **结果写回**
   - 在每条数据中追加：
     - `dialogue_type`
     - `turn_labels`（Turn 索引、结构标签、语义标签及统计信息）。
   - 输出文件与输入 `.jsonl` 同名，存放在指定的输出目录。

6. **分布可视化**
   - `scripts/plot_distribution.py` 读取标注结果，统计结构与语义标签的分布，并按 `dialogue_type` 分组。
   - 生成柱状图（PNG）与对应的 CSV 汇总，便于报告和质量分析。

## 项目结构

```
├─ qwen3.py                  # 调用 Qwen3 OpenAI 接口的轻量客户端
├─ requirements.txt          # 运行及可视化依赖（pandas, matplotlib）
├─ src/
│  └─ labeling/
│     ├─ __init__.py         # 导出 LabelingPipeline
│     ├─ turns.py            # 对话类型判定和 Turn 拆分
│     ├─ structural.py       # 结构化标注逻辑
│     ├─ semantic.py         # 语义化标注（LLM 适配）
│     ├─ io_utils.py         # JSONL 读写与路径工具
│     └─ pipeline.py         # 管道封装，整合流程
├─ scripts/
│  ├─ label_data.py          # 命令行入口，运行全量标注
│  └─ plot_distribution.py   # 标签统计与分布绘图
└─ docs/
   └─ ALGORITHM.md           # 本说明文档
```

## 使用说明

1. **安装依赖**
   ```powershell
   python -m pip install -r requirements.txt
   ```

2. **执行数据标注**
   ```powershell
   python scripts/label_data.py --data-dir data --output-dir output/labeled --max-workers 4 --enable-llm-log
   ```
   - 运行时会调用 `qwen3.py`，请提前配置环境变量 `QWEN_BASE_URL`、`QWEN_MODEL`、`OPENAI_API_KEY`（如需鉴权）。
   - 默认开启全局及单文件进度条；如果需要禁用单文件进度条，可追加 `--no-turn-progress`。
   - 可通过 `--max-workers` 控制并发线程数量（缺省值由系统决定）。
   - `--enable-llm-log` 打开语义判定的日志记录，文件位置可通过 `--llm-log-file` 指定（默认 `logs/semantic_llm.log`），日志中包含每条请求与响应的上下文定位信息，且不会在终端显示。
      - 语义化判定仅使用最后一条 assistant 自然语言回复作为 LLM 输入，避免冗长上下文导致请求超长。
   - 命令执行结束后，脚本会展示单个文件及总体的标签统计概览。

3. **生成分布图与汇总**
   ```powershell
   python scripts/plot_distribution.py --labeled-dir output/labeled --figure-dir output/figures --summary-dir output/summary
   ```
   - 输出包括结构标签、语义标签的堆叠柱状图和 CSV。

## 注意事项

- LLM 输出需是合法 JSON（包含 `missing_parameters` 与 `missing_tools` 布尔值），脚本会做解析校验。
- 若数据极大，标注脚本以流式方式读取及写回，避免占用过多内存。
- 可通过 `--log-level DEBUG` 查看更详细的处理日志。
- 若需自定义语义判定策略，可实现符合 `SemanticJudge` 接口的客户端并替换。
