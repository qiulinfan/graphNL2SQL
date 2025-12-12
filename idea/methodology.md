# 大纲

我现在要写一个

\section{Methodology}
\subsection{Graph-Modeling Designs}

 有哪些? 分别详述

\subsection{Graph Encoding Methods}

\subsection{LoRA Fine tuning}

选什么模型来 fine-tune? 如何 LoRA fine-tuning? 

\subsection{Reinforcement Learning Discussion}

为什么我们决定使用 EGD 作为替代?





# 1. graph designs 的不同设计

(Jurgens: Interersting! The overall writing is solid and sound. A quick question is why would you like to directly go with RL? I feel there might be some steps before it. e.g. trying different designs of your graph, etc.)

所以我们需要考虑一下 graph designs.

我们要回答的问题是：

> 给定一个数据库 schema (包含多张表、外键、列名等), 如何把它转化成图结构让 LLM 最有效地理解?

换句话说, 我们要探索：

1. **图的节点怎么定义** (表级? 列级? 混合?)
2. **图的边怎么定义** (外键? 语义相似? 上下文连接?)



### **Design A: Table-level Graph (最简 baseline)**

**节点 (nodes):** 每个表 (table)。
 **边 (edges):** 外键关系。

```
Nodes: [Student, Course, Department]
Edges: Student -- Course (student_id)
       Course -- Department (dept_id)
```

**特点:**

- 图结构简单, 边数少。
- LLM 容易理解整体数据库的 join 关系。
- 不包含列级语义, 无法捕捉复杂条件。

**适用场景:**
 先作为 baseline, 测试 “仅表间连接” 的信息量。

------

### **Design B: Column-level Graph**

**节点:** 每个列 (column)。
 **边:**

- 外键列 → 被引用的主键列。
- 同一表的列间 → intra-table edges。

```
[Student.id] -- [Course.student_id]
[Student.name] -- (intra) -- [Student.age]
```

**优点:**

- 细粒度, 模型可感知列名、类型、语义。
- 对复杂查询 (WHERE, GROUP BY) 更有帮助。

**缺点:**

- 节点数爆炸 (一个 schema 可能几十上百节点)。
- 文本线性化时过长, 可能超出上下文长度。

**改进建议:**
 只保留 “参与 join/condition” 的列节点 (pruned graph)。

------

### **Design C: Hybrid Graph (Table + Column 两层结构)**

**节点:** 表 + 列 (两层)。
 **边:**

- table → column (包含关系)
- 外键连接列 → 列
- table → table (语义或外键)

示意：

```
Table: Student
  ↳ id
  ↳ name
  ↳ age
Table: Course
  ↳ cid
  ↳ title
  ↳ student_id
Edges:
  Student.id -- Course.student_id
```

**优点:**

- 层次结构自然, 可表达表-列关系 + 跨表关系。
- 支持图遍历推理 (“哪些表包含与 Student 相关的字段?”)。

**缺点:**

- 稍复杂, 但仍可通过线性化简化输入。

**推荐：**
 作为主实验版本, 结构最完整且容易泛化。

------

### **Design D: Semantic Graph (基于嵌入相似度的增强边)**

在 Hybrid Graph 基础上, 额外添加：

- 语义相似边：
	 若两个列名/表名的文本嵌入余弦相似度 > 阈值 (如 0.8), 添加一条边。
	 例：

	```
	[Birthday] ↔ [DOB]
	[Department] ↔ [Dept]
	```

**优点:**
 补全 schema 中未显式定义但语义上存在关联的关系。
 **缺点:**
 边数显著增加, 要防止噪声过多。

**推荐:**
 作为 “语义增强版”, 用于验证图语义信息是否真的帮助模型捕捉 join 逻辑。

------

### **Design E: Typed Graph (添加边类型 / 节点类型标签)**

在 Design C/D 的基础上, 每条边或节点都有类型标签：

- Edge types: `foreign_key`, `intra_table`, `semantic_similar`
- Node types: `table`, `column`, `primary_key`, `foreign_key`

线性化示例：

```
[table] Student
  [column_primary] id
  [column] name
  [column] age
[foreign_key_edge] Student.id -> Course.student_id
[semantic_edge] Birthday ~ DOB
```

**优点:**
 LLM 可通过 type token 明确理解结构类型。
 **缺点:**
 prompt 更长, 但信息最丰富。

**推荐:**
 可以作为最终 “高表达版本” 测试是否提升结构理解能力。













# 2. graph 融合方法 (如何把图输入给模型): 文本线性化 / Graph Embedding

把 graph 信息作为 LLM 的 “外部知识提示 (structured prompt)”. 有两种可选择的方法. 我们都将进行尝试

**方式 A：文本线性化 (Text Linearization)**
 例如：

```
Schema Graph:
Table: Student(id, name, age)
Table: Course(cid, title)
Foreign Key: Student.id -> Course.student_id
Semantic Link: (DOB) ≈ (Birthday)

Question: "List the names of students taking math."
```

**方式 B：Graph Embedding (Graph Encoder)**
 使用一个小型 GNN / Transformer Encoder 将 schema graph 编码成 dense vector，然后通过 adapter (LoRA/Prefix-tuning) 注入 LLM。

可以让 graph encoder 参数保持冻结或轻微更新。



Option 2: Structured Markup

```
<Graph>
<Table name="Student">
  <Column name="id" type="primary_key"/>
  <Column name="name"/>
  <Column name="age"/>
</Table>
<Table name="Course">
  <Column name="cid" type="primary_key"/>
  <Column name="title"/>
  <Column name="student_id" type="foreign_key" ref="Student.id"/>
</Table>
</Graph>
```

Option 3: Tokenized Tags (适合 LLM fine-tune)

```
[Table] Student [Columns] id(PK), name, age
[Table] Course [Columns] cid(PK), title, student_id(FK->Student.id)
[Relation] Student.id = Course.student_id
```

> 推荐：Option 3，结构紧凑、可控且不依赖 XML 解析。

| 设计编号 | 节点粒度 | 是否含语义边   | 是否有类型标签 | 预期长度 | 实验优先级 |
| -------- | -------- | -------------- | -------------- | -------- | ---------- |
| A        | Table    | ❌              | ❌              | shortest | ✅ Baseline |
| B        | Column   | ❌              | ❌              | long     | ⚠️ 选做     |
| C        | Hybrid   | ✅(FK)          | ❌              | medium   | ✅ 主实验   |
| D        | Hybrid   | ✅(FK+semantic) | ❌              | long     | ✅ 对比实验 |
| E        | Hybrid   | ✅(FK+semantic) | ✅              | long     | ⚡ 高级实验 |











# 3. 关于 RL 的讨论: 是否必要? 现实考量

(1) RL 在 NL2SQL 里的动机（为什么要它）

在传统监督训练中，模型最小化 token-level cross-entropy：
$$
\mathcal{L}_{SFT} = -\sum_t \log p_\theta(y_t | y_{<t}, x)
$$
但这只能保证“SQL token 形式相似”，并不能保证：

- 生成的 SQL 可执行；
- 执行结果正确；
- 查询高效（join 少、运行快）。

**→ RL 目标是把训练信号从“语法”转向“执行结果”。**

具体地，定义 reward $R$：
$$
R = \lambda_1 R_\text{exec} + \lambda_2 R_\text{compile} + \lambda_3 R_\text{efficiency}
$$
其中：

- $R_\text{exec}=1$ 若执行结果正确，否则 0；
- $R_\text{compile}=1$ 若语法可执行；
- $R_\text{efficiency}$ 是负的执行时间或 join 数量惩罚。

最终目标：
$$
\max_\theta \ \mathbb{E}_{y\sim p_\theta} [R(y)]
$$
这就是 **Execution-Guided Reinforcement Learning** 的基本思路。

(2) 二、实践层面：现实问题

| 问题                         | 解释                                           | 后果                                                    |
| ---------------------------- | ---------------------------------------------- | ------------------------------------------------------- |
| **执行成本高**               | 每次生成 SQL 都要运行在 SQLite 上才能算 reward | Spider dev 800 样本 → 几千次 SQL 执行 → GPU idle 等 CPU |
| **Reward sparse**            | 只有正确执行才得 1 分，几乎全是 0              | 学习极慢，policy gradient variance 大                   |
| **SQL runtime errors**       | invalid column/table → reward 无定义           | 需 try-except 大量捕获错误                              |
| **LLM + RL cost**            | 3B LLaMA 每次采样生成 SQL → 巨大显存和时间开销 | 普通 RTX 无法支撑                                       |
| **Reward credit assignment** | 哪个 token导致错误？未知                       | 无法稳定更新梯度                                        |

在论文层面，很多工作（如 **EG-SQL**, **LearnAT**, **QDGAT**) 也承认 RL 训练**不稳定、昂贵、增益有限 (~1–3%)**。
 因此学术界目前常用 *execution-guided decoding*（EGD）替代，而非 full RL。



(3) 替代方案: EGD

 **Execution-Guided Decoding (EGD)**

> 不训练 RL，只在 inference 阶段用执行反馈筛选候选。

流程：

1. beam search 生成 top-k SQL；
2. 对每条 SQL 执行数据库；
3. 取返回结果正确（或可执行）的最高概率样本。

📈 **好处：**

- 无需 RL 框架；
- 不增加训练负担；
- 已验证提升 Execution Accuracy 3–5%。

> EGD = “RL without learning”，可直接写入论文作为 *execution-aware inference*。



我们将会采用 EGD.




# 4. 合理性分析

(1) **小型 LLM 可以通过 task-specific fine-tuning 弥补规模差距**

- 大模型（如 GPT-4, Gemini 1.5）在 NL2SQL 上的强大表现, 本质上依赖“latent SQL grammar knowledge + schema understanding”。
- 但这些知识是可迁移的：
	 小模型若在 domain-specific 数据上 fine-tune, 能学习 “如何将 NL → schema reasoning → SQL”，在该领域达到 **comparable 的效果**。
- 类似的趋势已经在 open-source 社区被验证：
	- **Phi-3-Mini (3.8B)** 在经过少量 fine-tuning 后, 能在 code generation 与 reasoning 上逼近 Llama-13B。
	- **TinyLlama / Mistral-7B** 在经过 instruction fine-tune 后, 对结构化 reasoning 任务性能显著提升。

所以"小模型 + 领域任务 fine-tune"是一个合理且高性价比的策略。



(2) **Graph-modeled schema 恰好弥补 LLM 的结构性缺陷**

- LLM 的弱点是**结构推理 (structured reasoning)**，尤其在多表 join 时：
	 模型需要显式地理解哪些表相连, join 条件来自何处, 以及哪些字段可对齐。
- 你们通过图建模把 schema 的连接关系**外显化**，让模型能：
	- 直接看到哪些表相关；
	- 理解列名间的语义对应；
	- 降低搜索空间和错误组合率。
- 这等价于把 “schema reasoning” 外包给图结构, 而 LLM 只需处理 “semantic alignment + SQL syntax generation”。
	 这是对 LLM 能力的最佳补充, 而不是冗余增强。



(3) **Fine-tuning 结合 Graph input 可以显著提升 Join 性能**

- Join 生成的难点：
	- **表的选择**：哪些表该加入
	- **连接条件**：用哪个键连接
	- **条件的层次与顺序**：是否需要嵌套、GroupBy 等
- Graph-based schema 通过节点 (表/列) 和边 (外键/语义关系) 自然描述这些逻辑结构。
- Fine-tuning 过程让 LLM 学会 “从 graph 推断 join 路径”。
- 对比大型 LLM 的 few-shot/zero-shot, 你们的方法有两个优势：
	1. **参数效率高**：不需要庞大上下文或提示工程；
	2. **泛化更稳**：learned structured reasoning 比纯 prompt reasoning 更稳定。



(4) 实践层面的可行性

- **模型体量**：3B–8B LLM 可以在单张 24 GB GPU 上 fine-tune。
- **数据**：Spider + BIRD 足够覆盖复杂多表结构；
- **Graph preprocessing**：静态构建一次即可缓存；
- **RL 阶段**：用离线策略 (batch rollout + reward scoring) 即可，无需在线环境。

这让整个 pipeline 在研究环境中是 **切实可执行的**。顺势利用了小型 LLM 的参数高效性、融合了 graph 的结构推理能力，并通过 fine-tuning 聚焦多表 join.