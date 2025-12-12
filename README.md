<p align="center">
  <h1 align="center">GraphNL2SQL</h1>
  <p align="center">
    <strong>Graph-Enhanced Schema Modeling for Natural Language to SQL</strong>
  </p>
  <p align="center">
    Fine-tune small LLMs (3-8B) to achieve strong NL2SQL performance using graph-based schema representations
  </p>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#methodology">Methodology</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#training">Training</a> •
  <a href="#references">References</a>
</p>

---

## Overview

**GraphNL2SQL** explores how small-parameter LLMs can be fine-tuned to achieve strong NL2SQL performance comparable to large models, but with a fraction of the computational cost.

Our approach focuses on **graph-based schema modeling** to strengthen the model's structural reasoning capabilities. By incorporating graph representations, we help smaller LLMs internalize multi-table relationships and bridge the gap between structural understanding and natural language semantics.

### Why Graph-Enhanced Schema Modeling?

Traditional NL2SQL approaches face challenges:
- Pre-LLM models lack general language understanding and exhibit poor few-shot generalization
- Large LLMs are computationally expensive and impractical for domain-specific fine-tuning
- Complex queries involving multi-table JOINs remain challenging for smaller models

Our solution: **Hybrid Graph Schema Representation** that explicitly models relational structures within database schemas.

---

## Features

- **Hybrid Graph Schema Representation**
  - Table-level and column-level nodes
  - Foreign key edges, containment edges, and semantic similarity edges
  - Multiple linearization styles for LLM input

- **Parameter-Efficient Fine-tuning**
  - LoRA-based fine-tuning for memory efficiency
  - Support for 4-bit/8-bit quantization
  - Configurations for various GPU sizes (8GB - 24GB+)

- **Multi-Dataset Support**
  - WikiSQL: Single-table queries for warmup training
  - Spider: Complex multi-table queries with JOINs and nested subqueries

- **Semantic Column Linking**
  - Automatic detection of semantically similar columns across tables
  - Embedding-based similarity computation

- **Execution-Guided Decoding (EGD)**
  - Generate multiple SQL candidates during inference
  - Validate syntax and executability on mock database
  - Select the best executable candidate for improved accuracy

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (8GB+ VRAM recommended)

### Basic Setup (Data Preparation & Inference)

```bash
git clone https://github.com/yourusername/graphNL2SQL.git
cd graphNL2SQL
pip install -r requirements.txt
```

### Full Setup (Including Training)

```bash
pip install -r requirements_train.txt
```

---

## Quick Start

### 1. Download Datasets

```bash
python scripts/download_data.py
```

### 2. Prepare Training Data

```bash
# Full preparation with semantic links
python scripts/prepare_training_data.py \
    --style structured \
    --format all \
    --semantic \
    --semantic-threshold 0.8 \
    --wikisql-balanced 5000 \
    --spider

# Spider only (for advanced training)
python scripts/prepare_training_data.py --spider --skip-wikisql
```

### 3. Train the Model

```bash
# Choose configuration based on your GPU
python scripts/train.py --config small_gpu   # 8-12GB VRAM
python scripts/train.py --config default     # 16-20GB VRAM
python scripts/train.py --config large_gpu   # 24GB+ VRAM
```

### 4. Run Inference

```bash
python scripts/inference.py --model ./checkpoints/phase2_spider/final
```

---

## Methodology

### Graph Schema Representation

We model relational database schemas as typed multi-relational graphs:

```
G = (V, E, R)
```

Where:
- **V**: Nodes representing tables and columns
- **E**: Edges with relation types
- **R**: Relation types = {table_column, foreign_key, intra_table, semantic_similar}

### Graph Structures

#### 1. Table & Column-Level Hybrid Graph (Main Design)

```
Hybrid Graph Example:
Table: Student
  Columns: id, name, age
Table: Course
  Columns: cid, title, student_id
Edges: Student.id -- Course.student_id (FK)
```

#### 2. Semantic Edge Extension

Automatically connects columns with semantically similar names:

```txt
Semantic Edges:
[Birthday] <-> [DOB]
[Department] <-> [Dept]
```

#### 3. Typed Graph Extension

Full type annotations for explicit relational roles:

```txt
[table] Student
  [column_primary] id
  [column] name
  [column] age
[foreign_key_edge] Student.id -> Course.student_id
```

### Text Linearization

Graphs are converted to text for LLM input:

```txt
[DATABASE]
university

[TABLES]
Student:
    student_id (PK)
    name
    age
Course:
    course_id (PK)
    title
Enrollment:
    enrollment_id (PK)
    student_id (FK)
    course_id (FK)

[FOREIGN KEYS]
Enrollment.student_id -> Student.student_id
Enrollment.course_id -> Course.course_id
```

### LoRA Fine-tuning

We use Low-Rank Adaptation for parameter-efficient training:

| Parameter | Default Value |
|-----------|---------------|
| Rank (r) | 32 |
| Alpha | 64 |
| Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |

Supported base models:
- Qwen2.5-7B-Instruct (default)
- Phi-3.5-mini-instruct
- DeepSeek-Coder-6.7B
- CodeLlama-7B-Instruct

---

## Datasets

### WikiSQL

- **Source**: Salesforce Research
- **Size**: 15,878 NL-SQL pairs, 4,550 tables
- **Complexity**: Single-table queries
- **Use**: Warmup training

**Example:**
```txt
Question: What institution had 6 wins and a current streak of 2?
SQL: SELECT "Institution" FROM "table" WHERE "Wins" = 6 AND "Current Streak" = '2'
```

### Spider

- **Source**: Yale University
- **Size**: 10,181 questions, 166 databases
- **Complexity**: Multi-table JOINs, nested queries, GROUP BY, ORDER BY
- **Use**: Main training

**Example:**
```python
{
    "db_id": "entrepreneur",
    "question": "Return the dates of birth for entrepreneurs who have 
                 either the investor Simon Woodroffe or Peter Jones.",
    "query": "SELECT T2.Date_of_Birth FROM entrepreneur AS T1 
              JOIN people AS T2 ON T1.People_ID = T2.People_ID 
              WHERE T1.Investor = 'Simon Woodroffe' 
              OR T1.Investor = 'Peter Jones'"
}
```

---

## Training

### Two-Phase Training Strategy

1. **Phase 1: WikiSQL Warmup** (1 epoch)
   - Learn basic SQL patterns
   - Single-table query generation

2. **Phase 2: Spider Training** (3 epochs)
   - Complex multi-table queries
   - JOIN and nested subquery handling

### Configuration Options

```json
// config.json
{
  "model": {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "load_in_4bit": true
  },
  "lora": {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05
  },
  "training": {
    "batch_size": 2,
    "gradient_accumulation": 8,
    "learning_rate": 2e-4,
    "max_seq_length": 1024
  }
}
```

### GPU Memory Requirements

| Config | VRAM | Batch Size | LoRA Rank |
|--------|------|------------|-----------|
| small_gpu | 8-12GB | 1 | 16 |
| default | 16-20GB | 2 | 32 |
| large_gpu | 24GB+ | 4 | 64 |

---

## Project Structure

```shell
graphNL2SQL/
├── scripts/
│   ├── schema_graph.py          # Hybrid graph schema representation
│   ├── text_linearization.py    # Graph-to-text linearization
│   ├── prepare_training_data.py # Data preparation pipeline
│   ├── train.py                 # Training script
│   ├── training_utils.py        # Training utilities and config
│   ├── testing_utils.py         # Testing and evaluation utilities
│   ├── inference.py             # Inference script
│   └── download_data.py         # Dataset download utilities
├── config.json              # Training configuration
├── pipeline.ipynb           # Main notebook for training and testing
├── requirements.txt         # Basic dependencies
└── requirements_train.txt   # Training dependencies
```

---

## Evaluation Metrics

- **Exact Match (EM)**: Predicted SQL exactly matches gold SQL (after normalization)
- **Execution Match (EX)**: Predicted SQL produces same results as gold SQL when executed
- **Execution-Guided Decoding (EGD)**: Generate multiple candidates and select the best executable one

### Evaluation with EGD

EGD improves accuracy by generating multiple SQL candidates and selecting the best one:

```python
from scripts.testing_utils import evaluate_with_execution

results = evaluate_with_execution(
    model, tokenizer, eval_data,
    use_egd=True,        # Enable EGD
    egd_candidates=5,    # Generate 5 candidates per question
)
print(f"EM: {results['exact_match_accuracy']:.2f}%")
print(f"EX: {results['execution_match_accuracy']:.2f}%")
```

---

## References

This project builds upon the following research:

- **Seq2SQL** - Zhong et al., 2017 - [arXiv:1709.00103](https://arxiv.org/abs/1709.00103)
- **SQLNet** - Xu et al., 2017 - [arXiv:1711.04436](https://arxiv.org/abs/1711.04436)
- **SyntaxSQLNet** - Yu et al., 2018 - [arXiv:1810.05237](https://arxiv.org/abs/1810.05237)
- **IRNet** - Guo et al., 2019 - [arXiv:1905.08205](https://arxiv.org/abs/1905.08205)
- **RAT-SQL** - Wang et al., 2021 - [arXiv:1911.04942](https://arxiv.org/abs/1911.04942)
- **LearNAT** - Liao et al., 2025 - [arXiv:2504.02327](https://arxiv.org/abs/2504.02327)
- **Spider 2.0** - Lei et al., 2025 - [arXiv:2411.07763](https://arxiv.org/abs/2411.07763)
- **Execution-Guided Decoding** - Wang et al., 2018 - [arXiv:1807.03100](https://arxiv.org/abs/1807.03100)

---

## License

This project is licensed under the [MIT LICENSE](LICENSE).

---

## Contributing

Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
