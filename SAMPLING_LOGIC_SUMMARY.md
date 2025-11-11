# 标注数据转换为 SGPT 格式说明

本文档说明如何将打完标签的对话数据转换为 SGPT 训练格式。

## 一、标注数据结构

标注后的数据位于 `output/labeled/*.jsonl`，每行是一个完整对话：

```json
{
  "id": "conversation_id",
  "messages": [...],           // OpenAI 格式的消息列表
  "tools": [...],              // 工具定义
  "turn_labels": [             // 每个 turn 的标签
    {
      "turn_index": 0,
      "structural_label": "Simple",
      "semantic_label": "Normal"
    }
  ],
  "dialogue_type": "Multi-Turn"
}
```

## 二、Turn 划分规则

采用 `src/labeling/turns.py::split_turns` 函数，**以 user 消息为锚点划分**：

- 每个 turn 从一个 user 消息开始
- 包含该 user 之后、下一个 user 之前的所有 assistant、tool 消息
- Turn 之间不重叠

**示例**：
```
[system, user_0, assistant_0, tool, assistant_0, user_1, assistant_1]
     └─────── Turn 0 ─────────────────────┘        └── Turn 1 ──┘
```

> **注意**：system 消息会被纳入 Turn 0 的上下文

## 三、数据采样（可选）

脚本：`scripts/sample_training_by_tag.py`

### 目的
按标签分布采样，构建平衡的训练集。

### 核心逻辑

1. **索引构建**：扫描所有标注文件，为每个 turn 创建索引
   - 为每个 turn 构建包含**完整历史的 raw_sample**（从对话开始到该 turn 结束）
   - 按标签组合（如 `(structural, semantic)`）分组索引

2. **按标签采样**：根据配置文件（如 `tag_sampling_config_example.json`）的目标分布抽取样本
   - 单位是 **Turn**（不是对话）
   - 支持单维度或双维度标签平衡

3. **输出**：
   - **RAW 格式**：每个 turn 一个样本，包含累积的完整消息历史
   - **SGPT 格式**：将 RAW 样本转换为训练格式（见下文）

### 为什么要包含完整历史？

**问题**：如果每个 turn 只包含自己的消息，模型训练时会缺少上下文。

**解决**：每个 turn 的 raw_sample 包含从对话开始的所有消息，确保上下文完整。

**代价**：转换时需要过滤，避免重复生成之前 turn 的训练样本。


## 四、SGPT 格式转换

脚本：`convert_oai_to_sgpt.py`

### 转换入口

```python
convert_multiturn_to_multi_sample(data)
```

**输入**：一个对话或 raw_sample，包含 messages 和 tools  
**输出**：SGPT 格式的样本列表

### 核心机制

**关键问题**：一个 turn 可能生成多个训练样本。

**原因**：一个 turn 中可能有多个 `loss=True` 的 assistant 消息（例如先调用工具，再返回最终答案）。

**处理方式**：
- 遍历 messages，为每个 `loss=True` 的 assistant 生成一个 SGPT 样本
- 每个样本的 input 包含该 assistant **之前**的所有消息
- 样本 ID 自动添加计数后缀：`_turn_0`, `_turn_1`, `_turn_2`...

### 单个样本的转换（convert_oai_to_sgpt）

将一个 assistant 响应转换为 SGPT 格式：

**转换规则**：
1. **System prompt**：合并 system 消息 + 工具定义（用 XML 标签包裹）
2. **Human input**：拼接之前的消息，用 `<|im_start|>` 和 `<|im_end|>` 标记
3. **GPT output**：
   - `reasoning_content` → `<think>...</think>`
   - `tool_calls` → `<tool_call>...</tool_call>`
   - `content` → 最终文本

**输出格式**：
```json
{
  "id": "conv_id_turn_0",
  "conversations": [
    {"from": "system", "value": "系统提示 + 工具说明"},
    {"from": "human", "value": "历史对话"},
    {"from": "gpt", "value": "<think>推理</think>\n\n响应内容"}
  ]
}
```

### 避免重复的过滤机制

**问题**：在采样场景中，由于 raw_sample 包含完整历史，直接转换会生成之前 turn 的样本。

**解决**（仅在 `sample_training_by_tag.py` 中）：
- 计算当前 turn 的样本计数区间 `[start, end]`
  - `start` = 之前所有 turn 的 `loss=True` 数量
  - `end` = start + 当前 turn 的 `loss=True` 数量 - 1
- 只保留 ID 后缀计数在区间内的样本

**示例**：
```
Turn 0: 2 个 loss=True → 样本计数 0, 1
Turn 1: 1 个 loss=True → 样本计数 2

选中 Turn 1 时，raw_sample 包含完整历史（3 个 loss=True）
转换会生成 3 个样本（计数 0, 1, 2）
过滤后只保留计数 2 的样本（属于 Turn 1）
```


## 五、完整示例

### 输入对话（已标注）
```json
{
  "id": "conv_123",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "天气如何？"},
    {"role": "assistant", "loss": true, "reasoning_content": "需要查询", "tool_calls": [...]},
    {"role": "tool", "content": "晴天"},
    {"role": "assistant", "loss": true, "reasoning_content": "总结结果", "content": "今天晴天"},
    {"role": "user", "content": "谢谢"},
    {"role": "assistant", "loss": true, "reasoning_content": "礼貌回应", "content": "不客气"}
  ],
  "turn_labels": [
    {"turn_index": 0, "structural_label": "Parallel", "semantic_label": "Normal"},
    {"turn_index": 1, "structural_label": "Simple", "semantic_label": "Normal"}
  ]
}
```

### Turn 划分
- **Turn 0**：user_0, assistant_0, tool, assistant_0（2 个 loss=True）
- **Turn 1**：user_1, assistant_1（1 个 loss=True）

### 直接转换（不采样）
调用 `convert_multiturn_to_multi_sample(对话)`，生成 **3 个 SGPT 样本**：
1. `conv_123_turn_0`：第 1 个 assistant（tool_calls）
2. `conv_123_turn_1`：第 2 个 assistant（最终答案）
3. `conv_123_turn_2`：第 3 个 assistant（Turn 1 的回应）

### 采样场景（假设只选中 Turn 1）

**RAW 样本**（1 条）：
```json
{
  "id": "conv_123_turn_1",
  "turn_index": 1,
  "messages": [...所有消息，包含 Turn 0 的历史...]
}
```

**转换 + 过滤**：
- 转换生成 3 个样本（计数 0, 1, 2）
- Turn 1 的计数区间：[2, 2]（之前有 2 个 loss=True）
- **过滤后保留 1 个样本**：`conv_123_turn_1_turn_2`


## 六、两种使用场景

### 场景 1：按标签分类（split_and_convert_by_tag.py）

**用途**：分析各标签的样本分布，不做采样。

**处理方式**：
- 对**整个对话**调用 `convert_multiturn_to_multi_sample`
- 根据对话的所有 turn 标签，将生成的样本写入对应标签文件
- 一个对话可能同时出现在多个标签文件中

**输出**：
```
output/tag_split/
  raw/structural/Simple.jsonl        # 原始对话
  sgpt/structural/Simple.jsonl       # 所有 SGPT 样本
  raw/semantic/Normal.jsonl
  sgpt/semantic/Normal.jsonl
```

### 场景 2：按标签采样（sample_training_by_tag.py）

**用途**：构建标签平衡的训练集。

**处理方式**：
- 按 turn 级别采样（根据配置的目标分布）
- 为每个选中的 turn 生成 raw_sample（包含完整历史）
- 转换时使用**计数过滤**，避免重复

**输出**：
```
output/training/
  raw/selected.jsonl                 # RAW 格式（每个 turn 一条）
  training_dataset.jsonl             # SGPT 格式（可能多条/turn）
  sample_report.json                 # 统计报告
```

## 七、关键要点总结

| 项目 | 说明 |
|------|------|
| **Turn 划分** | 以 user 消息为锚点，使用 `split_turns` 函数 |
| **样本单位** | 采样以 **Turn** 为单位，不是对话 |
| **上下文保持** | 每个 turn 的 raw_sample 包含从开始的完整历史 |
| **1:N 关系** | 1 个 turn 可生成多个 SGPT 样本（取决于 `loss=True` 数量）|
| **避免重复** | 采样场景使用计数区间过滤 |
| **格式转换** | reasoning → `<think>`，tool_calls → `<tool_call>` |
| **必要字段** | 必须有 `reasoning_content`，否则样本被跳过 |

## 八、统计数据解读

运行 `sample_training_by_tag.py` 后的报告示例：

```json
{
  "selection": {
    "total_selected": 100,     // 选中的 turn 数量
    "raw_selected": 100,        // RAW 样本数（应相等）
    "sgpt_total": 150,          // SGPT 样本数（≥ turn 数）
    "sgpt_selected": 150        // 写入的 SGPT 数（应 = sgpt_total）
  }
}
```

**正常关系**：
- `total_selected` = `raw_selected`（每个 turn 一个 raw 样本）
- `sgpt_total` ≥ `total_selected`（一个 turn 可能生成多个样本）
- `sgpt_total` = `sgpt_selected`（过滤逻辑确保无重复）
