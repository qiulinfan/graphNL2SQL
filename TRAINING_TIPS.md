# Training Tips & Troubleshooting

## 过拟合问题 (Overfitting)

### 症状
- Training Loss 很低（< 0.1）但 Validation Loss 很高（> 1.0）且持续上升
- 训练集准确率很高，但验证集准确率低

### 解决方案

#### 1. 降低学习率
```json
{
  "training": {
    "learning_rate": 0.0001,  // 从 0.0002 降低到 0.0001
    ...
  }
}
```

#### 2. 增加正则化
```json
{
  "lora": {
    "dropout": 0.1,  // 从 0.05 增加到 0.1
    ...
  }
}
```

#### 3. 减少 LoRA 参数
```json
{
  "lora": {
    "r": 8,      // 从 16 降低到 8
    "alpha": 16, // 从 32 降低到 16（保持 alpha/r = 2）
    ...
  }
}
```

#### 4. 减少 WikiSQL 训练
WikiSQL 太简单，容易导致过拟合：
```json
{
  "training": {
    "wikisql_epochs": 0.5,  // 从 1 降低到 0.5 或 0
    ...
  }
}
```

#### 5. 增加 Warmup
```json
{
  "training": {
    "warmup_ratio": 0.1,  // 从 0.05 增加到 0.1
    ...
  }
}
```

#### 6. 使用更小的批次大小
```json
{
  "training": {
    "batch_size": 1,  // 从 2 降低到 1
    "gradient_accumulation": 16,  // 从 8 增加到 16（保持有效批次大小）
    ...
  }
}
```

### 推荐配置（抗过拟合）

```json
{
  "lora": {
    "r": 8,
    "alpha": 16,
    "dropout": 0.1,
    "target_modules": [
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ]
  },
  "training": {
    "wikisql_epochs": 0.5,
    "spider_epochs": 3,
    "batch_size": 1,
    "gradient_accumulation": 16,
    "learning_rate": 0.0001,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.1,
    "max_seq_length": 1024,
    "gradient_checkpointing": true,
    "use_bf16": true
  }
}
```

---

## 执行错误 (Execution Errors)

### 症状
- 评估时出现 "Execution Errors: 35 (schema parsing issues)"
- EX 准确率低于预期

### 原因
1. Schema 解析不完整（已改进）
2. SQL 查询使用了 mock database 不支持的特性
3. 表名或列名包含特殊字符

### 解决方案

#### 1. 检查 Schema 格式
确保 schema 包含 `[TABLES]` 部分：
```
[TABLES]
table_name:
    column1 (PK)
    column2
```

#### 2. 使用 EGD 模式
EGD 会尝试多个候选，提高成功率：
```python
results = evaluate_with_execution(
    model, tokenizer, eval_data,
    use_egd=True,
    egd_candidates=5
)
```

#### 3. 查看详细错误
评估时会显示前几个执行错误的详细信息，帮助诊断问题。

---

## 性能优化建议

### 1. 两阶段训练策略
- **Phase 1 (WikiSQL)**: 快速适应 SQL 格式，但不要过度训练
- **Phase 2 (Spider)**: 专注于复杂查询

### 2. 数据增强
考虑使用：
- BIRD 数据集（更复杂的查询）
- Spider-Syn（同义改写）
- Spider-Realistic（更真实的查询）

### 3. 评估策略
- 使用 **Execution Match (EX)** 作为主要指标（比 EM 更宽容）
- 使用 **EGD** 提高推理时的准确率
- 在验证集上早停（early stopping）

---

## 监控指标

### 训练时
- **Training Loss**: 应该平稳下降
- **Validation Loss**: 应该下降，如果上升则过拟合
- **Loss Gap**: Training Loss 和 Validation Loss 的差距应该 < 0.5

### 评估时
- **Exact Match (EM)**: 严格匹配，通常较低（20-30%）
- **Execution Match (EX)**: 执行结果匹配，应该 > EM（50-60%+）
- **EX - EM Gap**: 应该 > 20%，说明模型生成了语义等价但格式不同的 SQL

---

## 常见问题

### Q: 为什么 EX 比 EM 高很多？
A: 这是正常的！说明模型生成了语义等价但格式不同的 SQL。例如：
- `SELECT name FROM student` vs `SELECT Name FROM Student`
- `ORDER BY age DESC` vs `ORDER BY Age DESC`

### Q: 训练损失降得太快正常吗？
A: 如果 Validation Loss 也在下降，正常。如果 Validation Loss 上升，说明过拟合。

### Q: 应该训练多少个 epoch？
A: 
- WikiSQL: 0.5-1 epoch（简单数据集）
- Spider: 2-5 epochs（复杂数据集）
- 使用验证集早停

### Q: LoRA rank 应该选多少？
A: 
- 小模型（7B）: r=8-16
- 大模型（13B+）: r=16-32
- 如果过拟合，降低 r

