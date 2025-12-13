"""
Training Data Preparation Pipeline for NL2SQL.

This script:
1. Loads WikiSQL and Spider datasets
2. Parses schemas into Hybrid Graph format
3. Applies text linearization
4. Exports training data in various formats
"""

import json
import random
from pathlib import Path
from typing import Generator
from tqdm import tqdm

try:
    from .schema_graph import (
        SchemaGraph,
        parse_wikisql_schema,
        parse_spider_schema,
        parse_create_table_schema,
    )
    from .text_linearization import (
        linearize_for_training,
        linearize_wikisql,
        format_training_example,
    )
except ImportError:
    from schema_graph import (
        SchemaGraph,
        parse_wikisql_schema,
        parse_spider_schema,
        parse_create_table_schema,
    )
    from text_linearization import (
        linearize_for_training,
        linearize_wikisql,
        format_training_example,
    )


# Use project root, not scripts/ directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "training_data"


# ============================================================================
# WikiSQL Data Loading
# ============================================================================

def load_wikisql_tables(split: str = "train") -> dict[str, dict]:
    """Load WikiSQL tables as a dictionary keyed by table ID."""
    tables_path = DATA_DIR / "wikisql" / f"{split}.tables.jsonl"

    if not tables_path.exists():
        raise FileNotFoundError(f"WikiSQL tables not found: {tables_path}")

    tables = {}
    with open(tables_path, 'r', encoding='utf-8') as f:
        for line in f:
            table = json.loads(line)
            tables[table["id"]] = table

    return tables


def load_wikisql_examples(split: str = "train") -> Generator[dict, None, None]:
    """
    Load WikiSQL examples.

    WikiSQL example format:
    {
        "phase": 1,
        "table_id": "1-1000181-1",
        "question": "What position does the player who played for...",
        "sql": {
            "sel": 3,
            "conds": [[2, 0, "Butler CC (KS)"]],
            "agg": 0
        }
    }
    """
    examples_path = DATA_DIR / "wikisql" / f"{split}.jsonl"

    if not examples_path.exists():
        raise FileNotFoundError(f"WikiSQL examples not found: {examples_path}")

    with open(examples_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def wikisql_to_sql_string(sql_obj: dict, table: dict) -> str:
    """
    Convert WikiSQL's structured SQL representation to SQL string.

    sql_obj format:
    {
        "sel": 3,  # column index to select
        "conds": [[2, 0, "value"], ...],  # [col_idx, op_idx, value]
        "agg": 0  # aggregation type
    }
    """
    headers = table.get("header", [])

    # Aggregation mapping
    agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]

    # Condition operator mapping
    cond_ops = ["=", ">", "<", ">=", "<=", "!="]

    # SELECT clause
    sel_col = headers[sql_obj["sel"]] if sql_obj["sel"] < len(headers) else "?"
    agg = agg_ops[sql_obj.get("agg", 0)]

    if agg:
        select_clause = f'SELECT {agg}("{sel_col}")'
    else:
        select_clause = f'SELECT "{sel_col}"'

    # FROM clause
    from_clause = 'FROM "table"'

    # WHERE clause
    where_parts = []
    for cond in sql_obj.get("conds", []):
        col_idx, op_idx, value = cond[0], cond[1], cond[2]
        col_name = headers[col_idx] if col_idx < len(headers) else "?"
        op = cond_ops[op_idx] if op_idx < len(cond_ops) else "="

        # Format value
        if isinstance(value, str):
            value_str = f"'{value}'"
        else:
            value_str = str(value)

        where_parts.append(f'"{col_name}" {op} {value_str}')

    if where_parts:
        where_clause = "WHERE " + " AND ".join(where_parts)
    else:
        where_clause = ""

    # Combine
    sql = f"{select_clause} {from_clause}"
    if where_clause:
        sql += f" {where_clause}"

    return sql


def classify_wikisql_pattern(sql_obj: dict) -> str:
    """
    Classify a WikiSQL query into a pattern category.

    Categories:
    - select_only: No aggregation, no conditions
    - where_only: No aggregation, has conditions
    - count: COUNT aggregation
    - sum: SUM aggregation
    - avg: AVG aggregation
    - max: MAX aggregation
    - min: MIN aggregation
    """
    agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    agg_idx = sql_obj.get("agg", 0)
    agg = agg_ops[agg_idx] if agg_idx < len(agg_ops) else ""
    has_conds = len(sql_obj.get("conds", [])) > 0

    if agg == "COUNT":
        return "count"
    elif agg == "SUM":
        return "sum"
    elif agg == "AVG":
        return "avg"
    elif agg == "MAX":
        return "max"
    elif agg == "MIN":
        return "min"
    elif has_conds:
        return "where_only"
    else:
        return "select_only"


def sample_wikisql_balanced(
    split: str = "train",
    total_samples: int = 5000,
    seed: int = 42
) -> list[dict]:
    """
    Sample WikiSQL examples with balanced pattern distribution.

    Evenly samples from each SQL pattern category to ensure diverse training.

    Args:
        split: Data split ("train" or "dev")
        total_samples: Total number of samples to return
        seed: Random seed for reproducibility

    Returns:
        List of raw WikiSQL examples (not yet processed)
    """
    import random
    random.seed(seed)

    print(f"Sampling {total_samples} balanced WikiSQL examples from {split}...")

    # Group examples by pattern
    pattern_examples: dict[str, list] = {
        "select_only": [],
        "where_only": [],
        "count": [],
        "sum": [],
        "avg": [],
        "max": [],
        "min": [],
    }

    for example in load_wikisql_examples(split):
        pattern = classify_wikisql_pattern(example["sql"])
        pattern_examples[pattern].append(example)

    # Print pattern distribution
    print("  Pattern distribution in dataset:")
    for pattern, examples in sorted(pattern_examples.items(), key=lambda x: -len(x[1])):
        print(f"    {pattern}: {len(examples)}")

    # Calculate samples per pattern
    num_patterns = len([p for p in pattern_examples if pattern_examples[p]])
    base_per_pattern = total_samples // num_patterns
    remainder = total_samples % num_patterns

    # Sample from each pattern
    sampled = []
    for i, (pattern, examples) in enumerate(pattern_examples.items()):
        if not examples:
            continue

        # Distribute remainder across first patterns
        n_samples = base_per_pattern + (1 if i < remainder else 0)
        n_samples = min(n_samples, len(examples))

        random.shuffle(examples)
        sampled.extend(examples[:n_samples])

    # Shuffle final result
    random.shuffle(sampled)

    print(f"  Sampled {len(sampled)} examples")
    return sampled


def process_wikisql(
    split: str = "train",
    linearization_style: str = "basic",
    max_examples: int = None,
    include_semantic_links: bool = False,
    semantic_threshold: float = 0.7,
    balanced_sample: int = None
) -> list[dict]:
    """
    Process WikiSQL dataset into training examples.

    Returns list of training examples with schema, question, and SQL.

    Args:
        balanced_sample: If set, sample this many examples with balanced patterns
    """
    print(f"Processing WikiSQL {split} split...")

    tables = load_wikisql_tables(split)
    examples = []

    # Get examples - either balanced sample or all
    if balanced_sample:
        raw_examples = sample_wikisql_balanced(split, balanced_sample)
    else:
        raw_examples = list(load_wikisql_examples(split))
        if max_examples:
            raw_examples = raw_examples[:max_examples]

    for example in tqdm(raw_examples, desc="WikiSQL"):

        table_id = example["table_id"]
        if table_id not in tables:
            continue

        table = tables[table_id]
        question = example["question"]
        sql_obj = example["sql"]

        # Convert to SQL string
        sql_string = wikisql_to_sql_string(sql_obj, table)

        # Create schema graph
        graph = parse_wikisql_schema(table)

        # Linearize schema
        if linearization_style == "wikisql":
            schema_text = linearize_wikisql(graph, table)
        else:
            schema_text = linearize_for_training(
                graph,
                style=linearization_style,
                include_db_id=False,  # WikiSQL table IDs are not meaningful
                include_semantic_links=include_semantic_links,
                semantic_threshold=semantic_threshold
            )

        # Create training example
        training_example = format_training_example(
            question=question,
            schema_text=schema_text,
            sql=sql_string,
            include_instruction=True
        )

        # Add metadata
        training_example["dataset"] = "wikisql"
        training_example["table_id"] = table_id
        training_example["split"] = split

        examples.append(training_example)

    print(f"Processed {len(examples)} WikiSQL examples")
    return examples


# ============================================================================
# SQL-Create-Context Data Loading (HuggingFace Dataset with CREATE TABLE schemas)
# ============================================================================

def process_sql_create_context(
    split: str = "train",
    linearization_style: str = "basic",
    max_examples: int = None,
    include_semantic_links: bool = False,
    semantic_threshold: float = 0.7
) -> list[dict]:
    """
    Process sql-create-context dataset from HuggingFace.

    This dataset has schema as CREATE TABLE statements and includes
    multi-table queries with JOINs (Spider-like complexity).

    Format:
    {
        "answer": "SELECT ...",
        "question": "...",
        "context": "CREATE TABLE ... ; CREATE TABLE ..."
    }
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print(f"Processing sql-create-context dataset...")

    # Load dataset from HuggingFace
    dataset = load_dataset("b-mc2/sql-create-context", split="train")

    # Split into train/dev (90/10)
    total = len(dataset)
    train_size = int(total * 0.9)

    if split == "train":
        indices = range(0, train_size)
    else:  # dev
        indices = range(train_size, total)

    if max_examples:
        indices = list(indices)[:max_examples]

    examples = []

    for idx in tqdm(indices, desc="SQL-Context"):
        example = dataset[int(idx)]

        question = example["question"]
        sql = example["answer"]
        create_statements = example["context"]

        # Parse CREATE TABLE statements into SchemaGraph
        graph = parse_create_table_schema(create_statements, db_id="database")

        # Linearize schema
        schema_text = linearize_for_training(
            graph,
            style=linearization_style,
            include_db_id=False,  # No meaningful db_id
            include_semantic_links=include_semantic_links,
            semantic_threshold=semantic_threshold
        )

        # Create training example
        training_example = format_training_example(
            question=question,
            schema_text=schema_text,
            sql=sql,
            include_instruction=True
        )

        # Add metadata
        training_example["dataset"] = "sql_create_context"
        training_example["split"] = split
        training_example["num_tables"] = len(graph.tables)
        training_example["has_join"] = "JOIN" in sql.upper()

        examples.append(training_example)

    print(f"Processed {len(examples)} sql-create-context examples")

    # Print statistics
    multi_table = sum(1 for ex in examples if ex["num_tables"] > 1)
    has_join = sum(1 for ex in examples if ex["has_join"])
    print(f"  Multi-table examples: {multi_table}")
    print(f"  Examples with JOIN: {has_join}")

    return examples


# ============================================================================
# Gretelai Synthetic Text-to-SQL (Good for multi-table JOINs)
# ============================================================================

def process_gretelai(
    split: str = "train",
    linearization_style: str = "basic",
    max_examples: int = None,
    include_semantic_links: bool = False,
    semantic_threshold: float = 0.7,
    complexity_filter: list[str] = None
) -> list[dict]:
    """
    Process gretelai/synthetic_text_to_sql dataset.

    This dataset has good coverage of:
    - Single and multiple JOINs
    - Subqueries, CTEs, window functions
    - Various SQL complexity levels (100k examples, 21% multi-table)

    Args:
        complexity_filter: Optional list of complexities to include, e.g.,
            ['single join', 'multiple_joins', 'subqueries']
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print(f"Processing gretelai/synthetic_text_to_sql dataset...")

    # Load dataset
    if split == "train":
        dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    else:
        dataset = load_dataset("gretelai/synthetic_text_to_sql", split="test")

    # Apply complexity filter if specified
    if complexity_filter:
        dataset = dataset.filter(
            lambda ex: ex['sql_complexity'] in complexity_filter
        )

    total = len(dataset)
    if split == "train":
        indices = range(0, int(total * 0.9))
    else:
        indices = range(int(total * 0.9), total)

    if max_examples:
        indices = list(indices)[:max_examples]

    examples = []

    for idx in tqdm(indices, desc="Gretelai"):
        ex = dataset[int(idx)]

        question = ex["sql_prompt"]
        sql = ex["sql"]
        create_context = ex["sql_context"]
        complexity = ex["sql_complexity"]

        # Parse CREATE TABLE statements
        graph = parse_create_table_schema(create_context, db_id="database")

        # Linearize schema
        schema_text = linearize_for_training(
            graph,
            style=linearization_style,
            include_db_id=False,
            include_semantic_links=include_semantic_links,
            semantic_threshold=semantic_threshold
        )

        # Create training example
        training_example = format_training_example(
            question=question,
            schema_text=schema_text,
            sql=sql,
            include_instruction=True
        )

        # Add metadata
        training_example["dataset"] = "gretelai"
        training_example["split"] = split
        training_example["num_tables"] = len(graph.tables)
        training_example["has_join"] = "JOIN" in sql.upper()
        training_example["complexity"] = complexity

        examples.append(training_example)

    print(f"Processed {len(examples)} gretelai examples")

    # Print statistics
    multi_table = sum(1 for ex in examples if ex["num_tables"] > 1)
    has_join = sum(1 for ex in examples if ex["has_join"])
    print(f"  Multi-table examples: {multi_table}")
    print(f"  Examples with JOIN: {has_join}")

    return examples


# ============================================================================
# NumbersStation NSText2SQL (Large dataset with many multi-table examples)
# ============================================================================

def process_nstext2sql(
    split: str = "train",
    linearization_style: str = "basic",
    max_examples: int = None,
    include_semantic_links: bool = False,
    semantic_threshold: float = 0.7
) -> list[dict]:
    """
    Process NumbersStation/NSText2SQL dataset.

    Large dataset (289k examples) with 41% multi-table queries.
    Good for training on complex real-world schemas.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print(f"Processing NumbersStation/NSText2SQL dataset...")

    # Load dataset (only has train split)
    dataset = load_dataset("NumbersStation/NSText2SQL", split="train")

    total = len(dataset)
    if split == "train":
        indices = range(0, int(total * 0.9))
    else:
        indices = range(int(total * 0.9), total)

    if max_examples:
        indices = list(indices)[:max_examples]

    examples = []

    for idx in tqdm(indices, desc="NSText2SQL"):
        ex = dataset[int(idx)]

        instruction = ex["instruction"]
        sql = ex["output"]
        source = ex.get("source", "unknown")

        # The instruction contains both schema and question
        # Try to split schema from question
        if "\n\n" in instruction:
            parts = instruction.split("\n\n")
            schema_parts = []
            question_parts = []
            found_question = False
            for part in parts:
                if not found_question and ("CREATE TABLE" in part.upper() or not part.strip()):
                    schema_parts.append(part)
                else:
                    found_question = True
                    question_parts.append(part)

            create_context = "\n\n".join(schema_parts)
            question = "\n\n".join(question_parts) if question_parts else "Generate SQL query"
        else:
            create_context = instruction
            question = "Generate SQL query"

        # Clean up question
        if question.startswith("--"):
            question = question.lstrip("-").strip()

        # Parse CREATE TABLE statements
        graph = parse_create_table_schema(create_context, db_id="database")

        if not graph.tables:
            continue

        # Linearize schema
        schema_text = linearize_for_training(
            graph,
            style=linearization_style,
            include_db_id=False,
            include_semantic_links=include_semantic_links,
            semantic_threshold=semantic_threshold
        )

        # Create training example
        training_example = format_training_example(
            question=question,
            schema_text=schema_text,
            sql=sql,
            include_instruction=True
        )

        # Add metadata
        training_example["dataset"] = "nstext2sql"
        training_example["split"] = split
        training_example["num_tables"] = len(graph.tables)
        training_example["has_join"] = "JOIN" in sql.upper()
        training_example["source"] = source

        examples.append(training_example)

    print(f"Processed {len(examples)} NSText2SQL examples")

    # Print statistics
    multi_table = sum(1 for ex in examples if ex["num_tables"] > 1)
    has_join = sum(1 for ex in examples if ex["has_join"])
    print(f"  Multi-table examples: {multi_table}")
    print(f"  Examples with JOIN: {has_join}")

    return examples


# ============================================================================
# Spider Data Loading (Official benchmark - download from Google Drive)
# https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view
# ============================================================================

def load_spider_schemas() -> dict[str, dict]:
    """Load Spider table schemas as dictionary keyed by db_id."""
    tables_path = DATA_DIR / "spider" / "tables.json"

    if not tables_path.exists():
        raise FileNotFoundError(f"Spider tables not found: {tables_path}")

    with open(tables_path, 'r', encoding='utf-8') as f:
        schemas = json.load(f)

    return {schema["db_id"]: schema for schema in schemas}


def load_spider_examples(split: str = "train") -> list[dict]:
    """
    Load Spider examples.

    Spider example format:
    {
        "db_id": "concert_singer",
        "query": "SELECT name, country FROM singer WHERE age > 20",
        "question": "What are the names and countries of singers older than 20?"
    }
    """
    if split == "train":
        examples_path = DATA_DIR / "spider" / "train_spider.json"
    else:
        examples_path = DATA_DIR / "spider" / "dev.json"

    if not examples_path.exists():
        raise FileNotFoundError(f"Spider examples not found: {examples_path}")

    with open(examples_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_spider(
    split: str = "train",
    linearization_style: str = "basic",
    max_examples: int = None,
    include_semantic_links: bool = False,
    semantic_threshold: float = 0.7,
    include_schema_linking: bool = False,
    prompt_style: str = "detailed",
) -> list[dict]:
    """
    Process Spider dataset into training examples.

    Spider is the gold-standard benchmark for complex, cross-domain NL2SQL.
    7000 train + 1034 dev examples across 166 databases.
    """
    print(f"Processing Spider {split} split...")

    schemas = load_spider_schemas()
    raw_examples = load_spider_examples(split)

    if max_examples:
        raw_examples = raw_examples[:max_examples]

    examples = []
    schema_cache: dict[str, tuple[SchemaGraph, str]] = {}
    skipped = 0

    for example in tqdm(raw_examples, desc="Spider"):
        db_id = example["db_id"]
        question = example["question"]
        sql = example["query"]

        # Get or create schema graph
        if db_id not in schema_cache:
            if db_id not in schemas:
                skipped += 1
                continue

            schema = schemas[db_id]
            # Check if schema has actual content
            if not schema.get("table_names_original"):
                skipped += 1
                continue

            graph = parse_spider_schema(schema)
            schema_text = linearize_for_training(
                graph,
                style=linearization_style,
                include_db_id=True,
                include_semantic_links=include_semantic_links,
                semantic_threshold=semantic_threshold
            )
            schema_cache[db_id] = (graph, schema_text)
        else:
            graph, schema_text = schema_cache[db_id]

        # Create training example
        training_example = format_training_example(
            question=question,
            schema_text=schema_text,
            sql=sql,
            include_instruction=True,
            schema_graph=graph,
            include_schema_linking=include_schema_linking,
            prompt_style=prompt_style,
        )

        # Add metadata
        training_example["dataset"] = "spider"
        training_example["db_id"] = db_id
        training_example["split"] = split
        training_example["num_tables"] = len(graph.tables)
        training_example["has_join"] = "JOIN" in sql.upper()

        examples.append(training_example)

    if skipped > 0:
        print(f"  Skipped {skipped} examples with missing schemas")

    # Print statistics
    multi_table = sum(1 for ex in examples if ex["num_tables"] > 1)
    has_join = sum(1 for ex in examples if ex["has_join"])
    print(f"Processed {len(examples)} Spider examples")
    print(f"  Multi-table examples: {multi_table}")
    print(f"  Examples with JOIN: {has_join}")

    return examples


# ============================================================================
# Export Functions
# ============================================================================

def save_jsonl(examples: list[dict], output_path: Path) -> None:
    """Save examples in JSONL format (one JSON object per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved {len(examples)} examples to {output_path}")


def save_json(examples: list[dict], output_path: Path) -> None:
    """Save examples in JSON format (single JSON array)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(examples)} examples to {output_path}")


def save_alpaca_format(examples: list[dict], output_path: Path) -> None:
    """
    Save examples in Alpaca instruction-tuning format.

    Format:
    {
        "instruction": "Given the database schema...",
        "input": "Schema: ... Question: ...",
        "output": "SELECT ..."
    }
    """
    alpaca_examples = []

    for ex in examples:
        alpaca_ex = {
            "instruction": "Given the following database schema and question, generate the SQL query that answers the question.",
            "input": f"{ex['schema']}\n\nQuestion: {ex['question']}",
            "output": ex.get("sql", ex.get("output", ""))
        }
        alpaca_examples.append(alpaca_ex)

    save_json(alpaca_examples, output_path)


def save_chat_format(examples: list[dict], output_path: Path) -> None:
    """
    Save examples in chat/conversation format for instruction-tuned models.

    Format:
    {
        "messages": [
            {"role": "system", "content": "You are a SQL expert..."},
            {"role": "user", "content": "Schema: ... Question: ..."},
            {"role": "assistant", "content": "SELECT ..."}
        ]
    }
    """
    system_prompt = (
        "You are a SQL expert. Given a database schema and a natural language question, "
        "generate the correct SQL query to answer the question. "
        "Output only the SQL query without any explanation."
    )

    chat_examples = []

    for ex in examples:
        chat_ex = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{ex['schema']}\n\nQuestion: {ex['question']}"},
                {"role": "assistant", "content": ex.get("sql", ex.get("output", ""))}
            ]
        }
        chat_examples.append(chat_ex)

    save_json(chat_examples, output_path)


# ============================================================================
# Main Pipeline
# ============================================================================

def prepare_all_data(
    linearization_style: str = "basic",
    output_format: str = "all",
    max_wikisql: int = None,
    max_examples: int = None,
    seed: int = 42,
    use_spider: bool = False,
    use_sql_context: bool = False,
    use_gretelai: bool = False,
    use_nstext2sql: bool = False,
    include_semantic_links: bool = False,
    semantic_threshold: float = 0.7,
    wikisql_balanced: int = None,
    skip_wikisql: bool = False,
    include_schema_linking: bool = False,
    prompt_style: str = "detailed",
) -> None:
    """
    Main function to prepare all training data.

    Args:
        linearization_style: Schema linearization style
        output_format: Output format - "jsonl", "alpaca", "chat", or "all"
        max_wikisql: Maximum WikiSQL examples (None for all)
        max_examples: Maximum examples for other datasets (None for all)
        seed: Random seed for shuffling
        use_spider: Use Spider dataset (7k train, requires download)
        use_sql_context: Use sql-create-context dataset (78k, ~2% multi-table)
        use_gretelai: Use gretelai dataset (100k, 21% multi-table)
        use_nstext2sql: Use NSText2SQL dataset (289k, 41% multi-table)
        include_semantic_links: Compute semantic links between columns
        semantic_threshold: Cosine similarity threshold for semantic links
        wikisql_balanced: If set, sample this many WikiSQL examples with balanced patterns
        skip_wikisql: Skip WikiSQL entirely
    """
    random.seed(seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_train_examples = []
    all_dev_examples = []

    # Process WikiSQL
    if not skip_wikisql:
        try:
            wikisql_train = process_wikisql(
                "train", linearization_style, max_wikisql,
                include_semantic_links, semantic_threshold,
                balanced_sample=wikisql_balanced
            )
            # For dev, use smaller balanced sample or limit
            dev_balanced = min(wikisql_balanced // 5, 1000) if wikisql_balanced else None
            wikisql_dev = process_wikisql(
                "dev", linearization_style, max_wikisql,
                include_semantic_links, semantic_threshold,
                balanced_sample=dev_balanced
            )

            all_train_examples.extend(wikisql_train)
            all_dev_examples.extend(wikisql_dev)

            save_jsonl(wikisql_train, OUTPUT_DIR / "wikisql_train.jsonl")
            save_jsonl(wikisql_dev, OUTPUT_DIR / "wikisql_dev.jsonl")

        except FileNotFoundError as e:
            print(f"WikiSQL not found: {e}")
            print("Run download_data.py first to download WikiSQL.")

    # Process Spider dataset
    if use_spider:
        try:
            spider_train = process_spider("train", linearization_style, max_examples,
                                          include_semantic_links, semantic_threshold,
                                          include_schema_linking=include_schema_linking,
                                          prompt_style=prompt_style)
            spider_dev = process_spider("dev", linearization_style, max_examples,
                                        include_semantic_links, semantic_threshold,
                                        include_schema_linking=include_schema_linking,
                                        prompt_style=prompt_style)

            all_train_examples.extend(spider_train)
            all_dev_examples.extend(spider_dev)

            save_jsonl(spider_train, OUTPUT_DIR / "spider_train.jsonl")
            save_jsonl(spider_dev, OUTPUT_DIR / "spider_dev.jsonl")

        except FileNotFoundError as e:
            print(f"Spider not found: {e}")
            print("Download from: https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view")

    # Process sql-create-context dataset
    if use_sql_context:
        try:
            sql_context_train = process_sql_create_context(
                "train", linearization_style, max_examples,
                include_semantic_links, semantic_threshold
            )
            sql_context_dev = process_sql_create_context(
                "dev", linearization_style, max_examples,
                include_semantic_links, semantic_threshold
            )

            all_train_examples.extend(sql_context_train)
            all_dev_examples.extend(sql_context_dev)

            save_jsonl(sql_context_train, OUTPUT_DIR / "sql_context_train.jsonl")
            save_jsonl(sql_context_dev, OUTPUT_DIR / "sql_context_dev.jsonl")

        except Exception as e:
            print(f"sql-create-context processing failed: {e}")

    # Process gretelai dataset
    if use_gretelai:
        try:
            gretelai_train = process_gretelai(
                "train", linearization_style, max_examples,
                include_semantic_links, semantic_threshold
            )
            gretelai_dev = process_gretelai(
                "dev", linearization_style, max_examples,
                include_semantic_links, semantic_threshold
            )

            all_train_examples.extend(gretelai_train)
            all_dev_examples.extend(gretelai_dev)

            save_jsonl(gretelai_train, OUTPUT_DIR / "gretelai_train.jsonl")
            save_jsonl(gretelai_dev, OUTPUT_DIR / "gretelai_dev.jsonl")

        except Exception as e:
            print(f"Gretelai processing failed: {e}")

    # Process NSText2SQL dataset
    if use_nstext2sql:
        try:
            nstext2sql_train = process_nstext2sql(
                "train", linearization_style, max_examples,
                include_semantic_links, semantic_threshold
            )
            nstext2sql_dev = process_nstext2sql(
                "dev", linearization_style, max_examples,
                include_semantic_links, semantic_threshold
            )

            all_train_examples.extend(nstext2sql_train)
            all_dev_examples.extend(nstext2sql_dev)

            save_jsonl(nstext2sql_train, OUTPUT_DIR / "nstext2sql_train.jsonl")
            save_jsonl(nstext2sql_dev, OUTPUT_DIR / "nstext2sql_dev.jsonl")

        except Exception as e:
            print(f"NSText2SQL processing failed: {e}")

    if not all_train_examples:
        print("No training examples generated. Please check dataset availability.")
        return

    # Shuffle combined data
    random.shuffle(all_train_examples)

    # Save combined data
    print("\nSaving combined training data...")

    if output_format in ["all", "jsonl"]:
        save_jsonl(all_train_examples, OUTPUT_DIR / "train.jsonl")
        save_jsonl(all_dev_examples, OUTPUT_DIR / "dev.jsonl")

    if output_format in ["all", "alpaca"]:
        save_alpaca_format(all_train_examples, OUTPUT_DIR / "train_alpaca.json")
        save_alpaca_format(all_dev_examples, OUTPUT_DIR / "dev_alpaca.json")

    if output_format in ["all", "chat"]:
        save_chat_format(all_train_examples, OUTPUT_DIR / "train_chat.json")
        save_chat_format(all_dev_examples, OUTPUT_DIR / "dev_chat.json")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING DATA SUMMARY")
    print("=" * 60)
    print(f"Total training examples: {len(all_train_examples)}")
    print(f"Total dev examples: {len(all_dev_examples)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Linearization style: {linearization_style}")

    # Dataset breakdown
    train_wikisql = sum(1 for ex in all_train_examples if ex["dataset"] == "wikisql")
    train_spider = sum(1 for ex in all_train_examples if ex["dataset"] == "spider")
    train_sql_context = sum(1 for ex in all_train_examples if ex["dataset"] == "sql_create_context")
    train_gretelai = sum(1 for ex in all_train_examples if ex["dataset"] == "gretelai")
    train_nstext2sql = sum(1 for ex in all_train_examples if ex["dataset"] == "nstext2sql")
    train_multi_table = sum(1 for ex in all_train_examples if ex.get("num_tables", 1) > 1)
    train_has_join = sum(1 for ex in all_train_examples if ex.get("has_join", False))

    print(f"\nTraining set breakdown:")
    if train_wikisql > 0:
        print(f"  WikiSQL: {train_wikisql}")
    if train_spider > 0:
        print(f"  Spider: {train_spider}")
    if train_sql_context > 0:
        print(f"  SQL-Context: {train_sql_context}")
    if train_gretelai > 0:
        print(f"  Gretelai: {train_gretelai}")
    if train_nstext2sql > 0:
        print(f"  NSText2SQL: {train_nstext2sql}")

    print(f"\nComplexity stats:")
    print(f"  Multi-table examples: {train_multi_table} ({train_multi_table/len(all_train_examples)*100:.1f}%)")
    print(f"  Examples with JOIN: {train_has_join} ({train_has_join/len(all_train_examples)*100:.1f}%)")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare NL2SQL training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset options:
  --spider           Spider benchmark (7k train, 1k dev, requires download)
  --sql-context      sql-create-context (78k examples, ~2%% multi-table)
  --gretelai         gretelai/synthetic_text_to_sql (100k examples, 21%% multi-table)
  --nstext2sql       NumbersStation/NSText2SQL (289k examples, 41%% multi-table)
  --wikisql-balanced Sample N WikiSQL examples with balanced SQL patterns
  --skip-wikisql     Skip WikiSQL entirely (for Spider-only training)

Recommended Training Plan:
  Phase 1 - WikiSQL warmup (1 epoch):
    python prepare_training_data.py --wikisql-balanced 5000 --skip-wikisql false

  Phase 2 - Spider main training (3 epochs):
    python prepare_training_data.py --spider --skip-wikisql

  Combined preparation:
    python prepare_training_data.py --wikisql-balanced 5000 --spider

Examples:
  # Balanced WikiSQL (5000 samples) + full Spider
  python prepare_training_data.py --wikisql-balanced 5000 --spider

  # Spider only (skip WikiSQL)
  python prepare_training_data.py --spider --skip-wikisql

  # Test with limited data
  python prepare_training_data.py --wikisql-balanced 500 --spider --max-examples 500
"""
    )
    parser.add_argument(
        "--style",
        choices=["structured", "basic", "detailed", "typed", "compact"],
        default="structured",
        help="Schema linearization style (default: structured)"
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "alpaca", "chat", "all"],
        default="all",
        help="Output format (default: all)"
    )
    parser.add_argument(
        "--max-wikisql",
        type=int,
        default=None,
        help="Maximum WikiSQL examples (default: all)"
    )
    parser.add_argument(
        "--wikisql-balanced",
        type=int,
        default=None,
        help="Sample N WikiSQL examples with balanced SQL patterns (e.g., 5000)"
    )
    parser.add_argument(
        "--skip-wikisql",
        action="store_true",
        help="Skip WikiSQL entirely"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum examples for other datasets (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    # Dataset selection
    parser.add_argument(
        "--spider",
        action="store_true",
        help="Use Spider dataset (7k train, requires local download)"
    )
    parser.add_argument(
        "--sql-context",
        action="store_true",
        help="Use sql-create-context dataset (78k examples, ~2%% multi-table)"
    )
    parser.add_argument(
        "--gretelai",
        action="store_true",
        help="Use gretelai/synthetic_text_to_sql (100k examples, 21%% multi-table)"
    )
    parser.add_argument(
        "--nstext2sql",
        action="store_true",
        help="Use NumbersStation/NSText2SQL (289k examples, 41%% multi-table)"
    )
    # Semantic links
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Enable semantic links between similar columns (requires sentence-transformers)"
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for semantic links (default: 0.7)"
    )
    # Schema linking
    parser.add_argument(
        "--schema-linking",
        action="store_true",
        help="Enable schema linking (maps question words to schema elements)"
    )
    parser.add_argument(
        "--prompt-style",
        choices=["simple", "detailed", "expert", "few_shot"],
        default="detailed",
        help="Prompt style: simple, detailed (default), expert, or few_shot"
    )

    args = parser.parse_args()

    prepare_all_data(
        linearization_style=args.style,
        output_format=args.format,
        max_wikisql=args.max_wikisql,
        max_examples=args.max_examples,
        seed=args.seed,
        use_spider=args.spider,
        use_sql_context=args.sql_context,
        use_gretelai=args.gretelai,
        use_nstext2sql=args.nstext2sql,
        include_semantic_links=args.semantic,
        semantic_threshold=args.semantic_threshold,
        wikisql_balanced=args.wikisql_balanced,
        skip_wikisql=args.skip_wikisql,
        include_schema_linking=args.schema_linking,
        prompt_style=args.prompt_style,
    )


if __name__ == "__main__":
    main()
