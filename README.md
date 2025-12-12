# Introduction

## Background and Motivation

Early research on Natural Language to SQL (NL2SQL) before the LLM era relied on task-specific encoder–decoder architectures such as IRNetcite{IRNet2019}, Seq2SQLcite{Seq2SQL2017}, SQLNetcite{SQLnet2017}, SyntaxSQLNetcite{SyntaxSQLNet2018}, and RAT-SQLcite{RatSQL2021}. These models achieved remarkable progress on benchmark datasets but remain fundamentally limited, since their parameters are typically under the order of tens of millions, without any large-scale pretraining on natural language corpora. As a result:

- They lack general language modeling ability and exhibit poor few-shot generalization, upon domains and databases they have not seen, even simple ones.
- When confronted with complex queries (e.g. multi-table Join), these models often fail to transfer learned knowledge.

In our baseline evaluations, we observed that such models, if not fine-tuned on the specific dataset, perform poorly even on simple benchmarks such as WikiSQL.

Fortunately, with the emergence of large-scale LLMs, complex NL2SQL tasks has become more feasible. General-purpose models such as GPT-5-Codex demonstrate strong natural language to code reasoning. However, this shift also exposes practical limitations. Research this yearcite{LearNAT2025} points out current SOTA methods largely depend on closed-source LLMs combined with prompt engineering, while open-source models still struggle on complex queries involving multiple joins or nested subqueries. In addition, large LLMs entail massive computational cost and memory consumption, making them impractical for domain-specific fine-tuning.

Therefore, our project aims to explore how a small-parameter LLM (3 ~ 8 B) can be fine-tuned to achieve strong NL2SQL performance comparable to large models in handling complex tasks, but with a fraction of the computational cost.

While LearNATcite{LearNAT2025} framework employs task decomposition, abstract syntax tree (AST) encoding and margin-aware reinforcement learning, our work instead focuses on graph-based schema modeling to strengthen the model’s structural reasoning. We hypothesize that incorporating graph representations provides a complementary advantage, helping smaller LLMs internalize multi-table relationships and bridge the gap between structural understanding and natural language semantics.

```
Examples from the WikiSQL Dataset

Example 1
Question: What institution had 6 wins and a current streak of 2?
GOLD SQL: SELECT "Institution" FROM "table" WHERE "Wins" = 6 AND "Current Streak" = '2'

Example 2
Question: Capital of Brze nad Bugiem has what population (1931) in 1,000s?
GOLD SQL: SELECT "Population (1931) in 1,000s" FROM "table" WHERE "Capital" = 'Brze nad Bugiem'
```



## Task Definition

The specific NLP task this project will address is multi-table Natural Language to SQL (NL2SQL) generation. The task of multi-table NL2SQL generation is to take a natural language query as input and automatically produce a syntactically correct and semantically accurate SQL query that retrieves the correct result from a relational database containing multiple interconnected tables.

- **Input:** a pair of (SQL schema, NL query). The schema may contain multiple tables with foreign key relationships. (Note: A foreign key is a column, or set of columns, in one table that refers to the primary key of another table.)
- **Output:** a structured SQL query that can be executed on the target database to return the intended result.

------

# Data

## WikiSQL Dataset

The **WikiSQL** dataset, released by Salesforce along with their paper *Seq2SQL: Generating Structured Queries from Natural Language Using Reinforcement Learning*, is one of the earliest and most widely used benchmarks for NL2SQL models. It is easy to extract and work with, containing 15,878 natural language (NL) to SQL pairs along with 4,550 data tables on which queries can be executed. 

Using this dataset, we evaluated our generated SQL queries in two ways:

1. By directly comparing the predicted SQL with the gold (reference) SQL.  
2. By executing both queries on the actual table using DuckDB and comparing their outputs.

However, the WikiSQL dataset focuses solely on generating SQL queries over a single table. It does not include joins or nested queries, making it relatively simple. Since our project aims to improve models’ ability to generate SQL across multiple tables and databases—which is a key limitation of many current NL2SQL systems—we also incorporated the Spider dataset for a more realistic and challenging evaluation.

> **Examples from the WikiSQL Dataset**  
>
> **Example 1**  
> **Question:** What institution had 6 wins and a current streak of 2?  
> **GOLD SQL:** `SELECT "Institution" FROM "table" WHERE "Wins" = 6 AND "Current Streak" = '2'`  
>
> **Example 2**  
> **Question:** Capital of Brześć nad Bugiem has what population (1931) in 1,000s?  
> **GOLD SQL:** `SELECT "Population (1931) in 1,000s" FROM "table" WHERE "Capital" = 'Brześć nad Bugiem'`

**Table 1. Truncated table for Example 1 from WikiSQL dataset (partial columns shown)**

| Institution                 | Wins | Losses | Home Wins | Home Losses |
| --------------------------- | ---- | ------ | --------- | ----------- |
| Boston College Eagles       | 6    | 1      | 3         | 1           |
| Clemson Tigers              | 9    | 5      | 4         | 3           |
| Duke Blue Devils            | 12   | 2      | 5         | 0           |
| Florida State Seminoles     | 6    | 8      | 4         | 3           |
| Georgia Tech Yellow Jackets | 4    | 9      | 3         | 2           |
| Maryland Terrapins          | 10   | 4      | 5         | 1           |
| ...                         | ...  | ...    | ...       | ...         |

## Spider Dataset

The **Spider** dataset, created by Yale University (*Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*), covers a wide range of domains and includes many complex SQL queries involving multi-table joins and nested structures. Its development set contains 1,023 questions and gold SQL queries across 166 databases, each provided with full schema information.

Compared to WikiSQL, Spider introduces substantially higher difficulty, as queries often involve multiple tables, foreign key reasoning, and advanced SQL operators such as `GROUP BY`, `ORDER BY`, and nested subqueries. This makes Spider a more realistic benchmark for assessing compositional generalization and schema understanding.

To use this dataset, we preprocessed each database schema into textual form by concatenating table and column names along with their relationships, then reorganized the format for compatibility with our model. We evaluated our system against the `Text2SQL-1.5B` model and achieved an exact match accuracy of 41.3\% on SQL generation. While the performance demonstrates reasonable generalization, further improvements may require enhanced schema linking and reasoning across complex relational structures.

*Listing 1. Example from Spider dataset*

```python
{
    "db_id": "entrepreneur",
    "query": "SELECT T2.Date_of_Birth   FROM entrepreneur AS T1 
              JOIN people AS T2 ON T1.People_ID = T2.People_ID 
              WHERE T1.Investor = 'Simon Woodroffe' 
                 OR T1.Investor = 'Peter Jones'",
    "question": "Return the dates of birth for entrepreneurs who have 
    either the investor Simon Woodroffe or Peter Jones."
}
```





For example of Spider schema structure, please see Appendix \ref{app:exp}.

# Related Work

This section summarizes three lines of related work that are closely connected to our project: (1) traditional pre-LLM models for NL2SQL; (2) structurally enhanced models that incorporate schema and syntax information; and (3) recent LLM-based pipelines that achieve state-of-the-art performance through task decomposition and reinforcement learning.

### Pre-LLM Models for NL2SQL

Early research on text-to-SQL adopted sequence-to-sequence architectures without large-scale language pretraining. Seq2SQL \cite{Seq2SQL2017} first introduced a reinforcement-learning objective to directly optimize execution accuracy rather than token-level similarity, but the approach suffered from unstable reward signals. SQLNet \cite{SQLnet2017} improved upon this by using a sketch-based decoder to avoid reinforcement learning altogether, achieving more stable training on the WikiSQL dataset. However, both methods focused on single-table scenarios and lacked the ability to generalize to complex, cross-domain databases.

### Structure-Enhanced Text-to-SQL Models

Subsequent work emphasized the importance of modeling structural dependencies in database schemas and SQL syntax. SyntaxSQLNet \cite{SyntaxSQLNet2018} leveraged a syntax tree decoder to enforce SQL grammar constraints, enabling generation of compositional queries. IRNet \cite{IRNet2019} introduced intermediate representations to capture the semantic alignment between natural language and database elements, improving cross-domain transfer. RAT-SQL \cite{RatSQL2021} further advanced this direction by proposing a relation-aware transformer encoder that explicitly encodes schema linking and foreign-key relations. Despite these advances, all such models remain relatively small in scale (tens of millions of parameters) and lack general-purpose language understanding, resulting in weak few-shot generalization and limited performance on multi-table join queries.

### LLM-based NL2SQL and LearNAT

In the era of large language models, NL2SQL research has shifted toward leveraging general-purpose LLMs with prompt-based reasoning. LearNAT \cite{LearNAT2025}, a framework that substantially improves the NL2SQL performance of open-source LLMs through task decomposition and reinforcement learning. LearNAT decomposes complex SQL generation into structured subtasks using Abstract Syntax Trees (ASTs), combining three key components: (1) an AST-guided decomposition synthesis procedure that generates valid subtasks, (2) margin-aware reinforcement learning that optimizes multi-step reasoning with AST-based preference signals, and (3) adaptive demonstration retrieval during inference. Experiments on Spider and BIRD show that LearNAT enables a 7B-parameter open-source model to approach GPT-4-level accuracy, demonstrating the effectiveness of decomposition and RL-based supervision.

While LearNAT focuses on task decomposition through AST structures, our project extends this idea in a complementary direction by incorporating graph-based schema representations. Instead of decomposing SQL syntax, we explicitly model relational structures within the database schema as a graph to strengthen the model’s understanding of multi-table connections. This approach aims to enhance small-parameter LLMs’ structural reasoning ability in NL2SQL tasks without the computational overhead of large-scale reinforcement learning.





# Methodology

## Graph-Modeling Designs

Given a relational database schema that contains multiple tables, foreign keys, and column names, the central question is how to transform this schema into a graph structure that an LLM can effectively understand.

Basic structures include:

### Table-Level Graph  
Nodes correspond to tables, and edges represent explicit foreign-key relationships.

```text
Example of Table-Level Graph
Nodes: [Student, Course, Department]
Edges: Student -- Course (student_id)
       Course -- Department (dept_id)
```



### Column-Level Graph

Nodes correspond to columns. Edges are constructed between foreign key columns and their referenced primary keys, and columns within the same table (intra-table edges).

```
Example of Column-Level Graph
[Student.id] -- [Course.student_id]
[Student.name] -- (intra) -- [Student.age]
```

### Basic Design: Table&Column-level Hybrid Graph

We create a hybrid of these two structures, letting both tables and columns be represented as nodes. Edges include:

- table–column containment edges
- foreign key connections between columns
- table–table edges for cross-table relationships

```
Hybrid Graph
Table: Student
  Columns: id, name, age
Table: Course
  Columns: cid, title, student_id
Edges: Student.id -- Course.student_id
```

This design naturally expresses hierarchical structure and supports cross-table reasoning such as “which tables contain columns related to Student”. It strikes a balance between expressiveness and length, and is used as our main experimental design. Meanwhile, we plan to add more experimental features:

### Extension 1: Semantic Edge

Built on top of the hybrid structure, this extension adds semantic edges between columns or tables whose names are semantically similar. We compute embedding similarity between names and connect pairs whose cosine similarity exceeds a threshold (e.g., 0.8):

```
Semantic Edge
[Birthday] ↔ [DOB]
[Department] ↔ [Dept]
```

This enriches schema understanding by introducing latent semantic connections not explicitly defined in the database schema, allowing the model to generalize across naming variations. However, the added edges may also increase graph density and introduce potential noise, so we use this variant to evaluate the trade-off between structural richness and model robustness.

### Extension 2: Typed Graph

As a further extension, all nodes and edges are annotated with type labels to explicitly distinguish their relational roles:

- **Edge types:** `foreign_key`, `intra_table`, `semantic_similar`
- **Node types:** `table`, `column`, `primary_key`, `foreign_key`

An example of the linearized input is shown below:

```
Typed Graph
[table] Student
  [column_primary] id
  [column] name
  [column] age
[foreign_key_edge] Student.id -> Course.student_id
[semantic_edge] Birthday ~ DOB
```

The typed graph provides the most expressive structural representation, enabling the LLM to differentiate between relationship types explicitly. However, this design also increases input length and may raise computational cost during fine-tuning. It will be evaluated as an advanced configuration in our ablation experiments.

In future experiments, we will operate on the fundamental graph design, together with the two extensions, combining the results for comparisons.

------

## Graph Encoding Methods

We will try two methods for integrating graph structures into LLM inputs and choose the better one in practice.

### Text Linearization

Graph structures are converted into textual prompts that describe table and column relations explicitly:

```
Text Linearization
Schema Graph:
Table: Student(id, name, age)
Table: Course(cid, title)
Foreign Key: Student.id -> Course.student_id
Semantic Link: (DOB) ≈ (Birthday)

Question: "List the names of students taking math."
```

This approach maintains full compatibility with existing LLM tokenizers and training pipelines.

### Graph Embedding

A lightweight graph encoder such as a GNN encoder encodes the schema graph into a dense embedding vector, which is injected into the LLM using parameter-efficient methods such as LoRA.

For practical fine-tuning, we use tokenized tag-style formatting for schema representation:

```
Graph Embedding
[Table] Student [Columns] id(PK), name, age
[Table] Course [Columns] cid(PK), title, student_id(FK->Student.id)
[Relation] Student.id = Course.student_id
```

Currently, we are using **Text Linearization**.

------

## LoRA Fine-tuning

We will fine-tune a small-parameter open-source LLM using LoRA. LoRA introduces low-rank matrices to the attention and feed-forward layers, allowing the model to learn task-specific adaptations while keeping the original weights frozen.

Candidate models include Llama-8B-Instruct, Mistral-7B, Phi-3-Mini (3.8B), and Qwen-1.5-7B.

Example training configuration:

- **Base model:** Llama-8B
- **Fine-tuning method:** LoRA
- **Rank r:** 32, **alpha:** 32, **dropout:** 0.05
- **Learning rate:** 5e-4
- **Sequence length:** 2048 tokens
- **Epochs:** 5 on Spider
- **Optimizer:** AdamW

We have not run the training since we are still applying for free GPUs. If free GPUs are not powerful enough, we may rent GPUs on Runpod. Moreover, the configuration will be modified later during practical training.

------

## Reinforcement Learning Discussion

Full RL training in NL2SQL faces several practical challenges: high execution cost (each SQL must be run on a database), sparse rewards, unstable gradients, and large computational overhead. As reported by prior workcite{LearNAT2025}, RL often yields marginal improvements (1–3%) with significant complexity.

To address these issues, we would consider not using full RL, but instead adopt **Execution-Guided Decoding (EGD)** as an alternative. EGD applies execution feedback only at inference time:

1. Use beam search to generate top-k SQL candidates.
2. Execute each query on the database.
3. Select the highest-probability query that is executable and returns the correct result.

As mentioned by cite{egd}, this approach provides **3–5% improvement in execution accuracy** without additional training cost, serving as a practical, execution-aware decoding strategy.



## Final Problem Formulation

### Graph Representation of Schema

Let a relational schema be modeled as a typed multi-relational graph  

\[
\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R}),
\]

where each node \(v \in \mathcal{V}\) is either a table or a column, and each edge \(e = (u, v, r) \in \mathcal{E}\) carries a relation type \(r \in \mathcal{R}\). We consider a set of relation types

\[
\mathcal{R} = \{\text{table\_column}, \text{foreign\_key}, \text{intra\_table}, \text{semantic\_similar}\}.
\]

Typed adjacency can be represented by a stack of binary matrices \(A^{(r)} \in \{0,1\}^{|\mathcal{V}|\times|\mathcal{V}|}\), one per relation \(r\).

For the semantic extension, we induce edges by embedding similarity. Let \(e(v)\) denote the textual name of node \(v\) and let  

\[
\mathbf{z}(v) = \mathrm{Enc}(e(v))
\]

be its embedding. A semantic edge is added between \(u\) and \(v\) if

\[
\frac{\mathbf{z}(u)^\top \mathbf{z}(v)}{\|\mathbf{z}(u)\|\,\|\mathbf{z}(v)\|} \ge \tau,
\]

where threshold \(\tau \in (0,1)\) is a hyperparameter to be chosen later.

When using text linearization for integration, the training input is the concatenation

\[
x = \mathrm{Concat}\big(q,\, \mathrm{SchemaText},\, s\big),
\]

where \(q\) is the natural language question and \(\mathrm{SchemaText}\) is a plain-text schema description.

---

### Generation Model

Let \(p_\theta\) be a small-parameter LLM with parameters \(\theta\). Given input \(x\) and optional graph feature \(\mathbf{h}_\mathcal{G}\), the model generates a SQL token sequence \(y = (y_1,\dots,y_T)\) with factorization

\[
p_\theta(y \mid x, \mathbf{h}_\mathcal{G}) = \prod_{t=1}^{T} p_\theta\big(y_t \mid y_{<t}, x, \mathbf{h}_\mathcal{G}\big).
\]

---

### Supervised Fine-tuning Objective

Given a dataset  

\[
\mathcal{D} = \{(q_i, \mathcal{G}_i, y_i^\star)\}_{i=1}^N,
\]

we minimize the token-level negative log-likelihood

\[
\mathcal{L}_{\text{SFT}}(\theta) = - \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log p_\theta\big(y_{i,t}^\star \mid y_{i,<t}^\star,\, x_i,\, \mathbf{h}_{\mathcal{G}_i}\big).
\]

---

### LoRA Parameterization

We adopt LoRA for parameter-efficient tuning. For a weight matrix \(W \in \mathbb{R}^{m \times n}\) in attention or MLP blocks, we learn a low-rank update

\[
W' = W + \Delta W,\quad \Delta W = B A,
\]

where \(A \in \mathbb{R}^{r \times n}\), \(B \in \mathbb{R}^{m \times r}\), and \(r \ll \min(m,n)\). Only \(A\) and \(B\) are trainable while \(W\) remains frozen.

---

### Execution-aware Inference via EGD

Although not yet implemented, we consider execution-guided decoding (EGD) for inference. Define an execution oracle \(\mathcal{E}(y)\) returning a tuple \((c, r)\), where:

- \(c \in \{0,1\}\): compilability  
- \(r \in \{0,1\}\): execution correctness  

At inference time we generate a candidate set

\[
\mathcal{C}_k = \mathrm{BeamSearch}\big(p_\theta(\cdot \mid x, \mathbf{h}_\mathcal{G}), k\big),
\]

evaluate all \(\mathcal{E}(y)\) for \(y \in \mathcal{C}_k\), and select

\[
\hat{y} \in \arg\max_{y \in \mathcal{C}_k}
\Big( r(y),\, c(y),\, \log p_\theta(y \mid x, \mathbf{h}_\mathcal{G}) \Big),
\]

using lexicographic ranking by correctness \(r\), then compilability \(c\), then model score.  
This improves execution accuracy without modifying training.

---

### Evaluation Metrics

We report:

**Exact match accuracy**

\[
\mathrm{EM} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\big[y_i = y_i^\star\big],
\]

**Execution accuracy**

\[
\mathrm{EX} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\big[\mathcal{E}(y_i) = (1,1)\big],
\]

**Join correctness**

\[
\mathrm{JAcc} = 
\frac{1}{N}
\sum_{i=1}^{N}
\frac{
\big|\mathrm{Joins}(y_i) \cap \mathrm{Joins}(y_i^\star)\big|
}{
\big|\mathrm{Joins}(y_i) \cup \mathrm{Joins}(y_i^\star)\big|
}.
\]

These metrics jointly evaluate syntactic correctness, execution validity, and structural reasoning quality.



# Bibliography

@misc{liu2025surveytexttosqlerallms,
      title={A Survey of Text-to-SQL in the Era of LLMs: Where are we, and where are we going?}, 
      author={Xinyu Liu and Shuyu Shen and Boyan Li and Peixian Ma and Runzhi Jiang and Yuxin Zhang and Ju Fan and Guoliang Li and Nan Tang and Yuyu Luo},
      year={2025},
      eprint={2408.05109},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2408.05109}, 
}



@misc{lei2025spider20evaluatinglanguage,
      title={Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows}, 
      author={Fangyu Lei and Jixuan Chen and Yuxiao Ye and Ruisheng Cao and Dongchan Shin and Hongjin Su and Zhaoqing Suo and Hongcheng Gao and Wenjing Hu and Pengcheng Yin and Victor Zhong and Caiming Xiong and Ruoxi Sun and Qian Liu and Sida Wang and Tao Yu},
      year={2025},
      eprint={2411.07763},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.07763}, 
}



@misc{LearNAT2025,
      title={LearNAT: Learning NL2SQL with AST-guided Task Decomposition for Large Language Models}, 
      author={Weibin Liao and Xin Gao and Tianyu Jia and Rihong Qiu and Yifan Zhu and Yang Lin and Xu Chu and Junfeng Zhao and Yasha Wang},
      year={2025},
      eprint={2504.02327},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.02327}, 
}



@misc{Seq2SQL2017,
      title={Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning}, 
      author={Victor Zhong and Caiming Xiong and Richard Socher},
      year={2017},
      eprint={1709.00103},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1709.00103}, 
}



@misc{SQLnet2017,
      title={SQLNet: Generating Structured Queries From Natural Language Without Reinforcement Learning}, 
      author={Xiaojun Xu and Chang Liu and Dawn Song},
      year={2017},
      eprint={1711.04436},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1711.04436}, 
}



@misc{SyntaxSQLNet2018,
      title={SyntaxSQLNet: Syntax Tree Networks for Complex and Cross-DomainText-to-SQL Task}, 
      author={Tao Yu and Michihiro Yasunaga and Kai Yang and Rui Zhang and Dongxu Wang and Zifan Li and Dragomir Radev},
      year={2018},
      eprint={1810.05237},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1810.05237}, 
}



@misc{IRNet2019,
      title={Towards Complex Text-to-SQL in Cross-Domain Database with Intermediate Representation}, 
      author={Jiaqi Guo and Zecheng Zhan and Yan Gao and Yan Xiao and Jian-Guang Lou and Ting Liu and Dongmei Zhang},
      year={2019},
      eprint={1905.08205},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1905.08205}, 
}



@misc{RatSQL2021,
      title={RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers}, 
      author={Bailin Wang and Richard Shin and Xiaodong Liu and Oleksandr Polozov and Matthew Richardson},
      year={2021},
      eprint={1911.04942},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1911.04942}, 
}



@misc{egd,
      title={Robust Text-to-SQL Generation with Execution-Guided Decoding}, 
      author={Chenglong Wang and Kedar Tatwawadi and Marc Brockschmidt and Po-Sen Huang and Yi Mao and Oleksandr Polozov and Rishabh Singh},
      year={2018},
      eprint={1807.03100},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1807.03100}, 
}