# 数据标注流程说明

本文档说明 `data/` 目录下对话数据的打标算法和项目结构，目标是按照用户消息（Turn）粒度生成结构化和语义化标签，同时给出标签分布可视化。

## 算法概览

1. **全局对话类型判定**  
   - 扫描 `messages` 列表统计 `role == "user"` 的数量。  
   - 若数量 ≤ 1，标记 `dialogue_type = "Single-Turn"`；否则标记为 `"Multi-Turn"`。

2. **Turn 拆分**  
   - 以 `role == "user"` 的消息为锚点，将完整 `messages` 列表拆分成多个 Turn。  
   - 每个 Turn 至少包含一条用户消息，并保留其后的 assistant/tool 回复。  
   - 对于非典型对话（例如缺少用户输入），算法会将现有消息聚合成单一 Turn，保证后续流程可执行。

3. **结构化打标**  
   - 针对 Turn 内所有 `role == "assistant"` 的消息，收集其中的 `tool_calls`。  
   - 统计总调用数 `total_calls` 与不同函数名数量 `unique_tool_count`。  
   - 根据规则输出以下标签之一：
     - `<无工具调用>`
     - `<单工具单调用>`
     - `<多工具单调用>`
     - `<单工具多调用>`
     - `<多工具多调用>`  
   - 同时写入辅助统计信息（调用数、去重后的工具名等），便于后续分析。

4. **语义化打标**  
   - 寻找 Turn 内最后一条 `role == "assistant"` 的消息。
   - 若该消息仍是 `tool_calls`，跳过语义标签。  
   - 否则，将完整 Turn、`tools` 列表以及目标回复文本发送至 LLM（通过 `qwen3.py`）。  
   - LLM 仅需返回布尔值：
     - `missing_parameters`: 是否缺少必要参数。
     - `missing_tools`: 是否缺少可用工具。
   - 根据全局 `dialogue_type` 转换为最终标签：
     - `Multi-Turn`：`<缺少相关参数>`、`<缺少所需工具>`、`<Base>`。
     - `Single-Turn`：`<幻觉：缺少相关参数>`、`<幻觉：缺少所需工具>`、或不打标。

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
   python scripts/label_data.py --data-dir data --output-dir output/labeled
   ```
   - 运行时会调用 `qwen3.py`，请提前配置环境变量 `QWEN_BASE_URL`、`QWEN_MODEL`、`OPENAI_API_KEY`（如需鉴权）。

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
