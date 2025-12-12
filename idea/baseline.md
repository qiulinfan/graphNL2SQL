## 经典 NL2SQL baseline 模型（2020–2021）

### 1. **Seq2SQL (Salesforce, 2017)**

* **论文:** *Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning*
* **特点:** 最早的 baseline 之一，基于 seq2seq + policy gradient。
* **适合:** 入门理解 NL2SQL task formulation（schema-aware decoding + SQL structure）。
* **优点:** 代码简单，适合教学。
* **代码:** [https://github.com/salesforce/WikiSQL](https://github.com/salesforce/WikiSQL)

---

### 2. **SQLNet (Xiaojun Xu et al., 2017)**

* **论文:** *SQLNet: Generating Structured Queries from Natural Language Without Reinforcement Learning*
* **特点:** 改进 Seq2SQL，去掉 RL，用 sketch-based decoding。
* **优点:** 稳定、无需 RL，结构分解清晰。
* **代码:** [https://github.com/xiaojunxu/SQLNet](https://github.com/xiaojunxu/SQLNet)

---

### 3. **SyntaxSQLNet (Yu et al., 2018, ACL)**

* **论文:** *SyntaxSQLNet: Syntax Tree Networks for Complex and Cross-Domain Text-to-SQL Task*
* **特点:** 基于 SQL AST 的递归生成，能处理复杂 SQL。
* **优点:** 很好的教学例子，体现 grammar-based decoding。
* **代码:** [https://github.com/taoyds/syntaxsqlnet](https://github.com/taoyds/syntaxsqlnet)

---

### 4. **IRNet (Guo et al., 2019)**

* **论文:** *Towards Complex Text-to-SQL in Cross-Domain Databases with Intermediate Representation*
* **特点:** 把 NL 转成中间表示，再生成 SQL。
* **优点:** 是 Spider 官方 baseline 之一，广泛用于复现。
* **代码:** [https://github.com/microsoft/IRNet](https://github.com/microsoft/IRNet)

---

### 5. **RAT-SQL (Wang et al., 2020, ACL)**

* **论文:** *RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers*
* **特点:** Graph attention over schema (relation-aware transformer)。
* **优点:** 最经典的现代 baseline；很多后续模型都在此基础上改进。
* **推荐:** 若课程项目需要一个较强 baseline，这个是首选。
* **代码:** [https://github.com/Microsoft/rat-sql](https://github.com/Microsoft/rat-sql)

---

### 6. **SmBop (Rubin & Berant, 2021, NAACL)**

* **论文:** *SmBop: Semi-autoregressive Bottom-up Semantic Parsing*
* **特点:** bottom-up 生成 SQL，Transformer encoder + semi-autoregressive decoder。
* **优点:** 比 RAT-SQL 简洁一些、依赖更轻。
* **代码:** [https://github.com/benbogin/smbop](https://github.com/benbogin/smbop)

---

## 推荐组合（根据项目复杂度）

| 目标               | 推荐模型               | 说明                  |
| ------------------ | ---------------------- | --------------------- |
| 🔰入门理解 + 可视化 | SQLNet 或 SyntaxSQLNet | 代码少、容易跑        |
| ⚙️标准 baseline     | IRNet                  | 支持 Spider、结构清晰 |
| 🧠稍高性能 baseline | RAT-SQL                | 教程多、性能强、主流  |
| 🧩对比实验用        | Seq2SQL + RAT-SQL      | 代表旧与新两代模型    |



## 数据集支持

所有这些模型都可直接用于：

* **WikiSQL**（单表任务）
* **Spider**（跨库复杂 SQL）
* （部分支持）**BIRD / CoSQL / SParC** 等后续扩展集



## 选择

- 最低 baseline: SQLNet / SyntaxSQLNet

- 期望 baseline: IRNet

- 追赶 baseline: RAT-SQL 

	我们通过 fine-tune 一个 3B 级别的 llama 模型来进行比较.

| 类别              | 代表模型                      | 年代      | 架构                       | 参数规模 | 特点                                |
| ----------------- | ----------------------------- | --------- | -------------------------- | -------- | ----------------------------------- |
| **传统 Baseline** | SQLNet / SyntaxSQLNet         | 2017–2018 | Seq2Seq (LSTM)             | < 0.01 B | 单表或简单 SQL                      |
| **中级 Baseline** | IRNet                         | 2019      | BiLSTM + GNN               | ≈ 0.03 B | 支持中等复杂 SQL, cross-domain      |
| **强力 Baseline** | RAT-SQL                       | 2020      | Relation-Aware Transformer | ≈ 0.1 B  | schema linking, Transformer encoder |
| **现代 LLM**      | LLaMA-3B / Phi-3 / Mistral-3B | 2024–2025 | Decoder-only Transformer   | ≈ 3 B    | few-shot, code-aware, generalizable |



非常好的问题——而且你的定位非常清晰 👏

你们要 **fine-tune 一个 3B 级别的 LLaMA 模型**（即现代 decoder-only LLM，≈3 × 10⁹ 参数），来做 NL2SQL。
 这个规模和能力已经远超早期基于 LSTM/GNN 的传统 NL2SQL baseline，因此要 carefully 选比较对象。

下面我给出一个**严谨对比框架**，说明 3B LLaMA 在 NL2SQL 中的合理比较对象，以及为什么你列的那三个 baseline 对应的分层很合理。

------

## 🧭 总体定位

| 类别              | 代表模型                      | 年代      | 架构                       | 参数规模 | 特点                                |
| ----------------- | ----------------------------- | --------- | -------------------------- | -------- | ----------------------------------- |
| **传统 Baseline** | SQLNet / SyntaxSQLNet         | 2017–2018 | Seq2Seq (LSTM)             | < 0.01 B | 单表或简单 SQL                      |
| **中级 Baseline** | IRNet                         | 2019      | BiLSTM + GNN               | ≈ 0.03 B | 支持中等复杂 SQL, cross-domain      |
| **强力 Baseline** | RAT-SQL                       | 2020      | Relation-Aware Transformer | ≈ 0.1 B  | schema linking, Transformer encoder |
| **现代 LLM**      | LLaMA-3B / Phi-3 / Mistral-3B | 2024–2025 | Decoder-only Transformer   | ≈ 3 B    | few-shot, code-aware, generalizable |

------

## 🧩 对比逻辑

### 🔹 1. 与 SQLNet / SyntaxSQLNet

- 这是**最低基线**：结构化 seq2seq，没有预训练语言知识。

- LLaMA-3B 即使不微调，zero/few-shot 表达能力也轻松超越它。

- ✅ **比较意义：展示 LLM 语言理解的压倒性优势。**

	> “我们的 LLaMA-3B-finetuned 模型在 Spider 上的准确率是传统 LSTM 模型的约 2–3 ×。”

------

### 🔹 2. 与 IRNet

- 这是**合理的期望 baseline**：IRNet 明确建模中间结构（semantic + SQL tree），是经典 Spider baseline。

- 3B LLaMA 可以通过 instruction-tuning 或结构模板约束（如 NL-SQL pair fine-tuning）达到相似甚至更好表现。

- ✅ **比较意义：**

	> “我们的 LLaMA-3B 模型不显式建模 IR，却能学习出等价的 schema-aware mapping。”

- IRNet 的 dev-set accuracy（Spider）大约 **~61 % – 63 %**。
	 如果你的 LLaMA 模型 fine-tune 后能到 **65 % +**，已经相当于“追平传统 architecture baseline”。

------

### 🔹 3. 与 RAT-SQL

- 这是**追赶 baseline**：Transformer encoder + schema graph，Spider 官方 SOTA（2020 年）。
- dev-set accuracy **~69 % – 72 %** (without BERT) / **~77 % – 80 %** (with BERT)。
- LLaMA-3B 若 fine-tune 质量高（高质量 NL-SQL pairs、schema prompt、few-shot exemplars），理论上能逼近或略超 GloVe 版 RAT-SQL。
- ✅ **比较意义：展示 LLM 在 schema linking 上能否取代专用 encoder。**

------

## 📊 对比层级总结

| 对比目标          | 模型示例              | 对比意义               | 你的 LLaMA-3B 预期表现               |
| ----------------- | --------------------- | ---------------------- | ------------------------------------ |
| **最低 baseline** | SQLNet / SyntaxSQLNet | 对比传统 seq2seq       | ✅ 轻松超越 (~50 % → > 60 %)          |
| **期望 baseline** | IRNet                 | 对比 GNN 语义表示      | ✅ 应该持平或略优 (~63 % → 65 – 70 %) |
| **追赶 baseline** | RAT-SQL (GloVe/BERT)  | 对比结构化 Transformer | ⚔️ 有挑战 (~70 – 77 %)                |

------

## 🧠 拓展建议（可写进报告）

> “传统 NL2SQL 模型依赖显式 schema linking、SQL 语法约束和模块化解码，而大模型通过大规模语言预训练隐式学习了这些模式。我们以 SQLNet、IRNet、RAT-SQL 三层 baseline 作为对照，展示结构化模型与通用语言模型在不同复杂度下的性能差距与互补性。”

------

## ⚙️ 实践提示

| 方案                   | GPU 推荐                  | Epoch | 数据规模           | 是否需 LoRA |
| ---------------------- | ------------------------- | ----- | ------------------ | ----------- |
| LLaMA-3B full finetune | ≥ 24 GB (A100 40GB ideal) | 3–5   | Spider (~7k pairs) | ❌           |
| LLaMA-3B + LoRA        | ≥ 12 GB (A6000, RTX 4090) | 5–10  | Spider / BIRD      | ✅ 推荐      |

> 你可以用 **LoRA rank 8 – 16**，**learning rate = 2e-5**，batch size = 8 – 16。

------

✅ **结论：**
 你们的 LLaMA-3B NL2SQL 模型应：

- **对比 SQLNet (轻松超越)**
- **目标对齐 IRNet (合理期望)**
- **挑战 RAT-SQL (stretch goal)**

------

是否希望我帮你写一份对应的 “evaluation table 模板” 和 “报告分析段落草稿”，方便你直接放进课程报告？



# 以下: GPT 生成的

非常棒，这个组合 **SQLNet → IRNet → RAT-SQL** 是课程项目中最经典的三层 baseline progression：

> **SQLNet** → 简单结构化解码
>  **IRNet** → 引入中间表示
>  **RAT-SQL** → 引入 graph transformer 与 schema linking

我们可以按照以下计划分阶段搭好环境、跑通 baseline。

------

## 🧭 总体目标

我们要为课程项目跑通：

1. **SQLNet / SyntaxSQLNet** – 轻量 baseline
2. **IRNet** – Spider 官方 baseline
3. **RAT-SQL** – 现代 strong baseline

数据集：**Spider**（默认）

> https://yale-lily.github.io/spider

------

## 🪜 计划分步执行

------

### **Step 1: 准备环境**

#### ✅ 创建统一的 Conda 环境

```bash
conda create -n nl2sql python=3.8
conda activate nl2sql
pip install torch==1.7.1 torchvision==0.8.2
pip install tqdm nltk spacy pandas
python -m spacy download en_core_web_sm
```

> 若 RAT-SQL 要求更高版本 (CUDA11+)，可单独开一个 `nl2sql_ratsql` 环境。

------

### **Step 2: 下载数据集**

Spider 官方：

```bash
git clone https://github.com/taoyds/spider.git
```

目录结构：

```
spider/
├── train_spider.json
├── dev.json
├── database/
│   ├── academic/
│   ├── ...
└── tables.json
```

> 放在统一路径，例如 `~/datasets/spider/`

------

## 🧩 各模型安装与运行

------

### ① **SQLNet / SyntaxSQLNet**

#### 📦 安装

```bash
git clone https://github.com/taoyds/syntaxsqlnet.git
cd syntaxsqlnet
pip install -r requirements.txt
```

#### 🏃 训练 (Spider)

```bash
python run.py \
    --train \
    --data_path ~/datasets/spider \
    --save_dir runs/sqlnet_baseline
```

#### 🧪 评估

```bash
python run.py \
    --test \
    --data_path ~/datasets/spider \
    --model_path runs/sqlnet_baseline/model_best.pt
```

> **注意**：SyntaxSQLNet 默认包含 SQLNet 代码，可通过参数切换 simple/complex 模式。

------

### ② **IRNet (Microsoft, 2019)**

#### 📦 安装

```bash
git clone https://github.com/microsoft/IRNet.git
cd IRNet
pip install -r requirements.txt
```

#### 🏃 训练

```bash
python train.py \
    --dataset spider \
    --data_root ~/datasets/spider \
    --save_dir runs/irnet
```

#### 🧪 评估

```bash
python eval.py \
    --dataset spider \
    --data_root ~/datasets/spider \
    --model_path runs/irnet/best_model.pt
```

> IRNet 用到了 “intermediate representation”，第一次运行会自动生成缓存文件。

------

### ③ **RAT-SQL (Microsoft, 2020)**

#### 📦 安装

```bash
git clone https://github.com/microsoft/rat-sql.git
cd rat-sql
pip install -r requirements.txt
```

#### 📋 数据预处理

RAT-SQL 需要 schema linking 和 relation encoding：

```bash
python preprocess/get_preprocessed_data.py \
    --dataset spider \
    --data_root ~/datasets/spider \
    --output_dir data/preprocessed/spider
```

#### 🏃 训练

```bash
python run.py \
    --config configs/spider-bert-run.jsonnet \
    --logdir runs/ratsql
```

#### 🧪 评估

```bash
python run.py \
    --config configs/spider-bert-run.jsonnet \
    --evaluate runs/ratsql/checkpoint_best.pt
```

> **可选**: 你可以改为 `--config configs/spider-glove-run.jsonnet` 若没有 GPU。

------

## 📊 推荐结果记录格式（课程报告可直接用）

| Model   | Year | Key Idea                    | Dev Acc (Spider) | Params | Runtime |
| ------- | ---- | --------------------------- | ---------------- | ------ | ------- |
| SQLNet  | 2017 | Sketch-based seq2seq        | ~55%             | ~10M   | ~30min  |
| IRNet   | 2019 | Intermediate Representation | ~61%             | ~30M   | ~1h     |
| RAT-SQL | 2020 | Relation-aware Transformer  | ~69%             | ~100M  | ~4h     |

------

## 📁 推荐目录结构

```
nl2sql_project/
├── datasets/
│   └── spider/
├── models/
│   ├── sqlnet/
│   ├── irnet/
│   └── ratsql/
├── runs/
│   ├── sqlnet_baseline/
│   ├── irnet/
│   └── ratsql/
└── report/
    ├── results.md
    └── plots/
```

------

## 🚀 下一步（可选）

我可以帮你：

- 生成 **统一运行脚本** (`run_all.sh`)
- 生成 **统一结果日志解析器**（自动提取 acc, loss, epoch）
- 或在 Colab 上自动部署 SQLNet/IRNet baseline（轻量演示版）

------

是否希望我帮你写一个 `run_all.sh`（一次性跑三个 baseline + 保存日志）？







好问题 👍，我们来精确比较一下这三个模型的规模，包括 **参数量 (parameter count)**、**磁盘占用 (code + model)**、以及 **训练显存/内存需求**。这些数据基于原论文 + 官方 repo + 常见 Spider 任务设置。

------

## ⚖️ 模型体量对比表

| 模型                      | 年份      | 参数量      | 模型文件大小       | 代码体积 | 显存需求 (训练) | 显存需求 (推理) | 主要依赖                              |
| ------------------------- | --------- | ----------- | ------------------ | -------- | --------------- | --------------- | ------------------------------------- |
| **SQLNet / SyntaxSQLNet** | 2017–2018 | ≈ 8–12 M    | ≈ 40 MB            | ≈ 50 MB  | **2 GB**        | **<1 GB**       | PyTorch 1.1, LSTM                     |
| **IRNet**                 | 2019      | ≈ 25–35 M   | ≈ 130 MB           | ≈ 120 MB | **4–6 GB**      | **1–2 GB**      | PyTorch 1.4, BiLSTM + GNN             |
| **RAT-SQL** (Glove)       | 2020      | ≈ 60–80 M   | ≈ 300 MB           | ≈ 250 MB | **6–8 GB**      | **2–3 GB**      | PyTorch 1.6, GAT/Transformer          |
| **RAT-SQL + BERT**        | 2020      | ≈ 110–140 M | ≈ 600 MB (含 BERT) | ≈ 300 MB | **10–12 GB**    | **3–4 GB**      | PyTorch 1.7, HuggingFace Transformers |

------

## 📊 更直观地理解：

- 🟩 **SQLNet** → 轻量级 baseline，老旧 seq2seq 架构，几乎可在任何显卡上跑。

	> 完整项目 < 100 MB，总参数量 < 1e7。

- 🟨 **IRNet** → 中等规模，有 GNN schema encoder，但不依赖 Transformer。

	> 一般在 RTX 3060（6 GB）即可稳定训练。

- 🟥 **RAT-SQL (BERT)** → 大规模 baseline，BERT encoder + relation graph，训练最耗显存。

	> 若 GPU < 10 GB，可先用 GloVe 版或 `--batch_size 4`。

------

## 💾 实际磁盘占用参考

假设都放在 `~/nl2sql_project/models` 下：

```
sqlnet/       ~80 MB
irnet/        ~250 MB
ratsql/       ~700 MB   (含 BERT-base)
--------------------------------
Total         ~1.0 GB
```

> 不算 Spider 数据 (~300 MB) 和缓存 (~1 GB)，整个项目在 2 GB 内完全可控。

------

## ✅ 建议配置

| 用途            | 最低 GPU                           | 推荐配置 |
| --------------- | ---------------------------------- | -------- |
| SQLNet / IRNet  | GTX 1050Ti / RTX 2060 (≥4 GB)      | ✅        |
| RAT-SQL (GloVe) | RTX 3060 (6 GB)                    | ✅        |
| RAT-SQL (BERT)  | RTX 3080 (10 GB) / Colab T4 / A100 | ✅        |

------

是否希望我帮你选出 **每个模型的轻量配置方案**（比如 RAT-SQL 的 GloVe 版、IRNet 的 batch size 减小版），以便三者能在同一台中端 GPU 上顺利运行？