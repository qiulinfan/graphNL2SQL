## 任务动机

过去的 nl2sql 模型s, 比如

并不是 LLM-based, 而是 task-specific encoder-decoder 模型, 参数量较小 (千万级), 没有通用语言建模能力. 

我们总结这些 pre-llm 的模型.的劣势在于: 

- 不具预训练语言知识, 从而没有通用语言建模能力. 后果是, 它们无法迁移到新 schema 或复杂 NL 指令, few-shot 能力太差
- 受限于参数量, 对于 schema 的理解能力有限, 因而对于多表 join 等复杂问题上正确率较低.

对于我们列举的这两条劣势, 我们进行了实验佐证. 在我们的 baseline 测评中, 我们发现一个模型如果没有在特定的数据集上进行 fine-tune, 则会在这个数据集上表现非常差, 甚至是简单的数据集 (比如 WikiSQL); 并且, 根据原论文的数据, 它们对于多表 join 的复杂问题上表现并不佳 (既然原作者都用数据承认, 我们就不必再复现了).



而 llm 时代, natural language to code 已经很成熟, 比如 gpt-5-codex. 但是问题也很显然: 作为大体量通用模型, 它们的参数量极大, 花费比较昂贵.

因而我们的 project 旨在探索: 如何在轻参数量的小型 llm 上, 达到优秀的 (例如 gpt4o) 的 nl2sql 特定任务能力; 尤其是针对多表 Join 问题, 能够达到好的效果. 这样, 不需要浪费大量级模型的算力级别, 即可完成 domain-specific 的 nl2sql 任务.

我们探索的方法是, 通过把 schema 转化成一个 graph-based representation 作为训练样本, 以数据增强的方式来 fine tune 一个小参数量的 llm.



小 llm 的 source 如下:

| 模型                    | 参数规模  | 优点                               | 备注                  |
| ----------------------- | --------- | ---------------------------------- | --------------------- |
| **Llama-3-8B-Instruct** | 8B        | 高质量指令调优; 可在单卡 24GB 运行 | 强大的 generalization |
| **Mistral-7B**          | 7B        | 高效推理; 可本地部署               |                       |
| **Phi-3-Mini / Small**  | 3.8B / 7B | 低资源友好; 微调速度快             | 适合实验性项目        |
| **Qwen-1.5-7B / 14B**   | 7B        | 中文/英文双语强; 支持结构化任务    |                       |

我们可以选取一个 3-7B 小模型.



## 任务分块

| 模块                           | 内容                                                         | 产出                       |
| ------------------------------ | ------------------------------------------------------------ | -------------------------- |
| **1. Schema Graph 构建**       | 把数据库模式转为图结构 (节点: 表/列, 边: 外键/语义关系)      | Graph 数据结构或图编码模块 |
| **2. Graph + NL 融合输入设计** | 研究如何将图输入给 LLM (linearization, GNN encoder, Graph embedding 等) | Graph-aware encoding 方法  |
| **3. 模型微调**                | 以 Spider/BIRD 训练 baseline LLM (T5, RAT-SQL 等) 并加入 graph 表示 | Fine-tuned 模型            |
| **4. RL 优化**                 | 定义 reward 函数 (compilability, accuracy, complexity)，实现 RL 微调 | RL 训练框架                |
| **5. 评估与对比**              | 对比 baseline、graph-only、graph+RL 的性能                   | 各种 ablation 和指标报告   |



## 算力耗费预计

结论先说：**用小参数量 LLM 做 LoRA/QLoRA 微调完全可行**, 你大概率只需要**一块消费级或一块 A100/H100**就能把 Spider+BIRD 的实验跑通。预算方面，如果你做的是 3B–8B 模型、上下 2–3 轮 ablation + 一轮 RL，小规模实验**50–150 美金**通常够用；更精打细算甚至 **10–40 美金**也能跑出一版可交差的结果。

- **3B 模型**
	- **QLoRA 4-bit**: 12–16 GB VRAM 可跑；一张 **RTX 3090/4090 (24 GB)** 就稳。
	- **LoRA 8-bit/16-bit**: 建议 24 GB 起。
- **7B–8B 模型**
	- **QLoRA 4-bit**: 24 GB VRAM 合理；**RTX 4090 (24 GB)** 可跑。
	- **LoRA (fp16/bf16)**: 建议 **A100 40/80 GB** 或 **H100 80 GB** 更舒服。
- **上下文长度**: NL + schema + graph linearization 控制在 1k–2k tokens 通常足够；2k 以内对显存压力很友好。
- **SQL 执行回放 (RL)**: GPU 用于生成候选 SQL；**评测与执行主要吃 CPU/IO**，可以把 rollouts 分批做，不强求多卡。



按小时的云卡价格 (2025/11 参考)

- **RunPod**: H100 80 GB ≈ **$1.99/hr**, A100 80 GB ≈ **$1.19–$1.39/hr**（不同形态/商家） ([Runpod](https://www.runpod.io/gpu-pricing?utm_source=chatgpt.com))
- **Vast.ai**: H100 PCIe 常见区间 **$1.20–$1.87/hr**，A100、3090 更低；价格随供给浮动 ([Vast AI](https://vast.ai/pricing/gpu/H100-PCIE?utm_source=chatgpt.com))
- **Paperspace (DigitalOcean)**: 有老卡和促销档，页面示例含 V100 $2.48/hr、H100 promo $5.95/hr（相对贵） ([Google Cloud](https://cloud.google.com/compute/gpus-pricing?utm_source=chatgpt.com))
- **Lambda Cloud**: 有 H100/H200/B200 集群定价入口，实际按账户可见；行业价位与 RunPod/Vast 同量级或稍高 ([Lambda](https://lambda.ai/pricing?utm_source=chatgpt.com))
- **Colab**: 采用 Compute Units，**$9.99/100 CU**；T4 ≈11.7 CU/hr，A100 ≈62 CU/hr，Pro+ $49.99/mo 提供更高配额窗口（适合原型，不适合长训） ([Google Colab](https://colab.research.google.com/signup?utm_source=chatgpt.com))

> 现实建议: **性价比首选 RunPod/Vast**；需要可预期带宽与稳定磁盘再考虑 Paperspace/Lambda。公有云 AWS/GCP 同等算力通常更贵。



训练用时与总成本估算

以 Spider(≈10k) + BIRD(≈12.7k) 合计 ~22k 条样本为例，序列化后单条样本 **700–1200 tokens**（NL + schema 描述 + graph 提示 + SQL）。
 粗略记 **每 epoch ≈ 16–25M tokens**，做 **2–3 个 epoch** 的监督微调，再加一次轻量 RL。



3B 模型, QLoRA, 单卡 4090 或 A100

- **监督微调**: 2–5 小时可跑完 2–3 个 epoch（取决于 dataloader、seq_len、accumulation）。
- **轻量 RL**: 生成候选 SQL + 执行打分，设计在 1–3 小时区间（可分批）。
- **总 GPU 时长**: **3–8 小时**。
- **费用**:
	- 用 **A100 80 GB** 在 RunPod/Vast：**$1.2–$1.4/hr** → **$4–$11**。 ([Runpod](https://www.runpod.io/gpu-pricing?utm_source=chatgpt.com))
	- 用 **H100 80 GB**：**$1.9–$2.7/hr** → **$6–$22**。 ([Runpod](https://www.runpod.io/gpu-pricing?utm_source=chatgpt.com))
	- 用 **4090(24 GB)**：**$0.34–$0.8/hr** 常见 → **$1–$6**（不同商家浮动，参照 RunPod/Vast 低价档）。 ([Runpod](https://www.runpod.io/pricing?utm_source=chatgpt.com))



7B–8B 模型, QLoRA/LoRA, 单卡 A100/H100

- **监督微调**: **4–10 小时**（seq_len 更长或用 LoRA fp16 会更慢）。
- **轻量 RL**: **2–6 小时**（取决于你设定的 rollouts 数量与评测方式）。
- **总 GPU 时长**: **6–16 小时**。
- **费用**:
	- **A100 80 GB**：**$7–$22**。 ([Runpod](https://www.runpod.io/gpu-pricing?utm_source=chatgpt.com))
	- **H100 80 GB**：**$12–$43**。 ([Runpod](https://www.runpod.io/gpu-pricing?utm_source=chatgpt.com))

> 这些是**保守可达**的区间。实际更关键的是**工程细节**：把 graph linearization 控制在 1–2k tokens、启用 grad checkpointing、合理的 token-level batch（如 32k–128k tokens/step 累计），都能显著加速。



### 具体预估: 7B, Graph-enhanced NL2SQL + LoRA fine-tuning

1. **参数量刚好在可训练区间**
	- 相比 3B 模型，7B 的语言与语义推理能力显著更强，能更好理解自然语言与 SQL 的映射；
	- 相比 13B 或 70B 模型，仍然能在单卡 A100 80 GB 上完成 LoRA 训练。
2. **和 LoRA 完美匹配**
	- LoRA 只更新少量低秩矩阵，不需要全参数训练，7B 规模下显存与时间都能接受。
	- 4-bit QLoRA 甚至能在 24 GB 显存（如 RTX 4090）上跑。
3. **结构理解能力足够**
	- 7B 模型（Mistral-7B、Llama-3-8B、Qwen-1.5-7B 等）在 code reasoning 和 schema reasoning 任务上普遍优于 3B 小模型；
	- 在 multi-table join 场景，7B 模型能更稳定地记住 schema 结构与外键关系。



推荐训练配置 (LoRA fine-tune for NL2SQL)

| 项目                | 推荐值                                     | 说明                                               |
| ------------------- | ------------------------------------------ | -------------------------------------------------- |
| **Base Model**      | Llama-3-8B-Instruct 或 Mistral-7B-Instruct | 支持 instruction 格式, 英文语义好                  |
| **微调方法**        | LoRA / QLoRA                               | 建议 QLoRA 4-bit 节省显存                          |
| **Rank (r)**        | 16 或 32                                   | 增强表达力但仍显存可控                             |
| **Alpha**           | 32                                         | 常规比例                                           |
| **Dropout**         | 0.05                                       | 稳定训练                                           |
| **学习率**          | 2e-4 ~ 5e-4                                | 对 LoRA 权重                                       |
| **Batch size**      | token 级 32k ~ 64k                         | 梯度累积保证                                       |
| **Sequence length** | 2048                                       | 足够容纳 (NL + schema + graph linearization + SQL) |
| **Epochs**          | 2 ~ 3                                      | Spider + BIRD 总 ≈ 22k 样本                        |
| **优化器**          | PagedAdamW8bit (bnb)                       | LoRA 常用                                          |
| **Scheduler**       | cosine + warmup (0.05)                     | 平滑启动                                           |



显存与成本估算

| GPU                  | 适配方式    | 显存需求     | 每小时价 (RunPod/Vast 参考) | 预估总时长 | 预估成本  |
| -------------------- | ----------- | ------------ | --------------------------- | ---------- | --------- |
| **RTX 4090 (24 GB)** | QLoRA 4-bit | ≈ 22 GB      | $0.4 – $0.8/hr              | 8–12 hr    | $4 – $10  |
| **A100 80 GB**       | LoRA bf16   | ≈ 40 – 50 GB | $1.2 – $1.4/hr              | 6–10 hr    | $7 – $14  |
| **H100 80 GB**       | LoRA bf16   | ≈ 40 GB      | $1.9 – $2.5/hr              | 5–8 hr     | $10 – $20 |

> 这些时间是包括 2–3 个 epoch supervised + 1 轮轻量 RL 的全部训练。

如果你只想做 proof-of-concept 版本，用 4090 + QLoRA 完全足够；
 想要做 论文级 实验，可上 A100 80 GB ，预算 10–20 美元。



LoRA 在 7B 上的实际效果预期

- **Fine-tune 后性能可接近 13B 模型**：LoRA 的结构调整能快速学习 schema-reasoning 模式；
- **多表 join 性能提升明显**：因为 LoRA 会重点更新 attention 层的低秩部分，正好对应你们任务中最关键的“表间关系建模”；
- **Inference 时 LoRA 模块可加载/卸载**，部署轻量。

