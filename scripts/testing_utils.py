"""
Testing Utilities for NL2SQL Evaluation.
NL2SQL 评估测试工具模块

This module encapsulates all testing and evaluation functions for use in 
notebooks and scripts.
本模块封装了所有测试和评估相关函数，供 notebook 和脚本使用。

Main Functions / 主要功能:
- generate_sql(): Generate SQL from question and schema / 从问题和模式生成 SQL
- load_finetuned_model(): Load model with LoRA adapters / 加载带 LoRA 的模型
- evaluate_model(): Evaluate on dataset with Exact Match / 在数据集上评估精确匹配
- generate_sql_with_egd(): Execution-Guided Decoding / 执行引导解码
- evaluate_with_egd(): Evaluate using EGD method / 使用 EGD 方法评估

Called by / 调用者:
- pipeline.ipynb: Testing section / 测试部分
- inference.py: Command-line inference / 命令行推理
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# =============================================================================
# SQL GENERATION
# =============================================================================

def generate_sql(
    model,
    tokenizer,
    question: str,
    schema: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """
    Generate SQL from question and schema.
    从问题和模式生成 SQL 查询。
    
    Function / 功能:
        Core inference function. Formats input as chat prompt, generates SQL
        using the model, and extracts the SQL from the response.
        核心推理函数。将输入格式化为对话提示，使用模型生成 SQL，并从响应中提取 SQL。
    
    Called by / 调用者:
        - evaluate_model(): For batch evaluation (批量评估时调用)
        - run_quick_test(): For single-sample testing (单样本测试时调用)
        - generate_sql_candidates(): For EGD candidate generation (EGD 候选生成时调用)
    
    Args / 参数:
        model: The language model (语言模型)
        tokenizer: The tokenizer (分词器)
        question: Natural language question (自然语言问题)
        schema: Database schema in text format (文本格式的数据库模式)
        max_new_tokens: Maximum tokens to generate (最大生成 token 数)
        temperature: Sampling temperature (采样温度)
        do_sample: Whether to use sampling (是否使用采样)
        
    Returns / 返回:
        str: Generated SQL query (生成的 SQL 查询)
    """
    system_msg = (
        "You are a SQL expert. Given a database schema and a natural language question, "
        "generate the correct SQL query. Output only the SQL query."
    )

    instruction = (
        "Given the following database schema and question, "
        "generate the SQL query that answers the question."
    )
    user_input = f"{instruction}\n\n{schema}\n\n[QUESTION]\n{question}"

    prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
[SQL]
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract SQL from response
    if "[SQL]" in generated:
        sql_part = generated.split("[SQL]")[-1]
        if "<|im_end|>" in sql_part:
            sql_part = sql_part.split("<|im_end|>")[0]
        return sql_part.strip()

    return generated.strip()


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_finetuned_model(
    adapter_path: str,
    base_model_name: Optional[str] = None,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    device_map: str = "auto",
) -> Tuple[Any, Any]:
    """
    Load a fine-tuned model with LoRA adapters.
    
    Args:
        adapter_path: Path to the LoRA adapter checkpoint
        base_model_name: Base model name (auto-detected if None)
        load_in_4bit: Whether to use 4-bit quantization
        load_in_8bit: Whether to use 8-bit quantization
        device_map: Device mapping strategy
        
    Returns:
        Tuple of (model, tokenizer)
    """
    adapter_path = Path(adapter_path)
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    
    # Auto-detect base model name from adapter config
    if base_model_name is None:
        config_path = adapter_path / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                base_model_name = config.get("base_model_name_or_path")
        
        if base_model_name is None:
            base_model_name = "Qwen/Qwen2.5-7B-Instruct"
            print(f"Warning: Could not detect base model, using default: {base_model_name}")
    
    print(f"Loading base model: {base_model_name}")
    
    # Configure quantization
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization")
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        print("Using 8-bit quantization")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # Load LoRA adapters
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    
    print(" Model loaded successfully!")
    
    return model, tokenizer


# =============================================================================
# EVALUATION
# =============================================================================

def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison.
    规范化 SQL 以便比较。
    
    Function / 功能:
        Standardizes SQL strings for fair comparison by handling common
        variations that don't affect query semantics.
        通过处理不影响查询语义的常见变体来标准化 SQL 字符串以进行公平比较。
    
    Called by / 调用者:
        - evaluate_model(): Compare predicted vs gold SQL (比较预测和标准 SQL)
        - evaluate_with_egd(): Same purpose in EGD evaluation (EGD 评估中同样用途)
    
    Handles / 处理:
        - Case insensitivity: SELECT = select (大小写不敏感)
        - Whitespace normalization: multiple spaces -> single (空格规范化)
        - Quote unification: double quotes -> single quotes (引号统一)
    
    Args / 参数:
        sql: Raw SQL string (原始 SQL 字符串)
    
    Returns / 返回:
        str: Normalized SQL string (规范化后的 SQL 字符串)
    """
    sql = sql.lower()                    # Case insensitive / 忽略大小写
    sql = sql.replace('"', "'")          # Unify quotes / 统一引号为单引号
    sql = " ".join(sql.split())          # Normalize whitespace / 规范化空格
    return sql


def evaluate_model(
    model,
    tokenizer,
    eval_data: List[Dict],
    max_samples: Optional[int] = None,
    max_new_tokens: int = 256,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset using Exact Match metric.
    使用精确匹配指标在数据集上评估模型。
    
    Function / 功能:
        Iterates through evaluation data, generates SQL for each example,
        compares with gold SQL using normalized comparison, and computes accuracy.
        遍历评估数据，为每个样本生成 SQL，使用规范化比较与标准 SQL 对比，计算准确率。
    
    Called by / 调用者:
        - pipeline.ipynb: Main evaluation in testing section (测试部分的主要评估)
    
    Args / 参数:
        model: The language model (语言模型)
        tokenizer: The tokenizer (分词器)
        eval_data: List of dicts with 'question', 'schema', 'sql' keys
                   包含 question, schema, sql 键的字典列表
        max_samples: Maximum samples to evaluate, None=all (最大评估样本数)
        max_new_tokens: Max tokens per generation (每次生成的最大 token 数)
        verbose: Print progress updates (是否打印进度)
        
    Returns / 返回:
        Dict with keys: accuracy, correct, total, results
        包含 accuracy, correct, total, results 的字典
    """
    if max_samples:
        eval_data = eval_data[:max_samples]
    
    results = []
    correct = 0
    
    if verbose:
        print(f"Evaluating on {len(eval_data)} samples...")
    
    for i, example in enumerate(eval_data):
        # Generate SQL
        pred_sql = generate_sql(
            model, tokenizer,
            example["question"],
            example["schema"],
            max_new_tokens=max_new_tokens,
        )
        
        # Normalize and compare
        gold_norm = normalize_sql(example["sql"])
        pred_norm = normalize_sql(pred_sql)
        is_match = gold_norm == pred_norm
        
        if is_match:
            correct += 1
        
        results.append({
            "question": example["question"],
            "gold_sql": example["sql"],
            "pred_sql": pred_sql,
            "match": is_match,
        })
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(eval_data)}] Accuracy: {100*correct/(i+1):.1f}%")
    
    accuracy = 100 * correct / len(eval_data) if eval_data else 0
    
    if verbose:
        print(f"\n Final Accuracy: {accuracy:.2f}% ({correct}/{len(eval_data)})")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(eval_data),
        "results": results,
    }


def show_evaluation_examples(
    eval_results: Dict[str, Any],
    num_correct: int = 3,
    num_incorrect: int = 3,
) -> None:
    """
    Display example predictions from evaluation results.
    
    Args:
        eval_results: Results dictionary from evaluate_model
        num_correct: Number of correct examples to show
        num_incorrect: Number of incorrect examples to show
    """
    results = eval_results["results"]
    
    correct_examples = [r for r in results if r["match"]]
    incorrect_examples = [r for r in results if not r["match"]]
    
    print("=" * 60)
    print("CORRECT PREDICTIONS")
    print("=" * 60)
    for i, ex in enumerate(correct_examples[:num_correct]):
        print(f"\n[{i+1}] Question: {ex['question']}")
        print(f"    SQL: {ex['pred_sql']}")
    
    print("\n" + "=" * 60)
    print("INCORRECT PREDICTIONS")
    print("=" * 60)
    for i, ex in enumerate(incorrect_examples[:num_incorrect]):
        print(f"\n[{i+1}] Question: {ex['question']}")
        print(f"    Gold: {ex['gold_sql']}")
        print(f"    Pred: {ex['pred_sql']}")


# =============================================================================
# QUICK TEST
# =============================================================================

def run_quick_test(
    model,
    tokenizer,
    schema: Optional[str] = None,
    question: Optional[str] = None,
) -> str:
    """
    Run a quick test with default or custom schema/question.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        schema: Optional custom schema (uses default if None)
        question: Optional custom question (uses default if None)
        
    Returns:
        Generated SQL query
    """
    if schema is None:
        schema = """[TABLES]
student:
    id (PK)
    name
    age
    major"""
    
    if question is None:
        question = "How many students are majoring in Computer Science?"
    
    print("=" * 60)
    print("QUICK TEST")
    print("=" * 60)
    print(f"\nQuestion: {question}")
    print(f"\nSchema:\n{schema}")
    
    sql = generate_sql(model, tokenizer, question, schema)
    
    print(f"\nGenerated SQL:\n{sql}")
    
    return sql


# =============================================================================
# INTERACTIVE TESTING
# =============================================================================

def run_interactive_test(
    model,
    tokenizer,
    schema: str,
    questions: List[str],
) -> List[Dict[str, str]]:
    """
    Run interactive testing with multiple questions.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        schema: Database schema
        questions: List of questions to test
        
    Returns:
        List of dictionaries with question and generated SQL
    """
    print("=" * 60)
    print("INTERACTIVE TESTING")
    print("=" * 60)
    print(f"\nSchema:\n{schema}")
    print("\n" + "-" * 60)
    
    results = []
    
    for i, question in enumerate(questions):
        print(f"\n[{i+1}] Q: {question}")
        sql = generate_sql(model, tokenizer, question, schema)
        print(f"    SQL: {sql}")
        results.append({"question": question, "sql": sql})
    
    return results


def run_batch_test(
    model,
    tokenizer,
    test_cases: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """
    Run batch testing with multiple schema/question pairs.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        test_cases: List of dicts with 'schema' and 'question' keys
        
    Returns:
        List of results with question, schema, and generated SQL
    """
    print("=" * 60)
    print("BATCH TESTING")
    print("=" * 60)
    
    results = []
    
    for i, case in enumerate(test_cases):
        schema = case["schema"]
        question = case["question"]
        
        print(f"\n[{i+1}] Q: {question}")
        sql = generate_sql(model, tokenizer, question, schema)
        print(f"    SQL: {sql}")
        
        results.append({
            "question": question,
            "schema": schema,
            "sql": sql,
        })
    
    return results


# =============================================================================
# PREDEFINED TEST SCHEMAS
# =============================================================================

UNIVERSITY_SCHEMA = """[DATABASE]
university

[TABLES]
student:
    student_id (PK)
    name
    age
    department_id (FK)
course:
    course_id (PK)
    title
    credits
enrollment:
    enrollment_id (PK)
    student_id (FK)
    course_id (FK)
    grade

[FOREIGN KEYS]
student.department_id -> department.department_id
enrollment.student_id -> student.student_id
enrollment.course_id -> course.course_id"""

ECOMMERCE_SCHEMA = """[DATABASE]
ecommerce

[TABLES]
customer:
    customer_id (PK)
    name
    email
    city
product:
    product_id (PK)
    name
    price
    category
order:
    order_id (PK)
    customer_id (FK)
    order_date
    total_amount
order_item:
    item_id (PK)
    order_id (FK)
    product_id (FK)
    quantity

[FOREIGN KEYS]
order.customer_id -> customer.customer_id
order_item.order_id -> order.order_id
order_item.product_id -> product.product_id"""

SAMPLE_QUESTIONS = {
    "university": [
        "How many students are there?",
        "What are the names of students older than 20?",
        "List all courses with more than 3 credits.",
        "Which students are enrolled in the Database course?",
        "What is the average age of students?",
    ],
    "ecommerce": [
        "How many customers are from New York?",
        "What is the total revenue from all orders?",
        "List products with price greater than 100.",
        "Which customers placed orders in 2024?",
        "What is the most popular product by quantity sold?",
    ],
}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def test_with_university_schema(model, tokenizer) -> List[Dict[str, str]]:
    """Run tests with the university schema."""
    return run_interactive_test(
        model, tokenizer,
        UNIVERSITY_SCHEMA,
        SAMPLE_QUESTIONS["university"]
    )


def test_with_ecommerce_schema(model, tokenizer) -> List[Dict[str, str]]:
    """Run tests with the ecommerce schema."""
    return run_interactive_test(
        model, tokenizer,
        ECOMMERCE_SCHEMA,
        SAMPLE_QUESTIONS["ecommerce"]
    )


# =============================================================================
# EXECUTION-GUIDED DECODING (EGD)
# 执行引导解码
# =============================================================================

def generate_sql_candidates(
    model,
    tokenizer,
    question: str,
    schema: str,
    num_candidates: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> List[Tuple[str, float]]:
    """
    Generate multiple SQL candidates using sampling.
    使用采样生成多个 SQL 候选。
    
    Function / 功能:
        Uses temperature sampling to generate diverse SQL candidates,
        computes confidence scores for each, and returns them sorted by score.
        使用温度采样生成多样化的 SQL 候选，计算每个候选的置信度分数，按分数排序返回。
    
    Called by / 调用者:
        - generate_sql_with_egd(): First step of EGD pipeline (EGD 流程第一步)
    
    Args / 参数:
        model: The language model (语言模型)
        tokenizer: The tokenizer (分词器)
        question: Natural language question (自然语言问题)
        schema: Database schema (数据库模式)
        num_candidates: Number of candidates to generate (候选数量)
        max_new_tokens: Maximum tokens per candidate (每个候选最大 token 数)
        temperature: Sampling temperature, higher=more diverse (采样温度，越高越多样)
        
    Returns / 返回:
        List of (sql, score) tuples sorted by score descending
        按分数降序排列的 (sql, score) 元组列表
    """
    system_msg = (
        "You are a SQL expert. Given a database schema and a natural language question, "
        "generate the correct SQL query. Output only the SQL query."
    )

    instruction = (
        "Given the following database schema and question, "
        "generate the SQL query that answers the question."
    )
    user_input = f"{instruction}\n\n{schema}\n\n[QUESTION]\n{question}"

    prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
[SQL]
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    candidates = []
    
    with torch.no_grad():
        # Generate multiple candidates using sampling
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            num_return_sequences=num_candidates,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        
        # Extract sequences and compute scores
        sequences = outputs.sequences
        
        for i in range(num_candidates):
            generated = tokenizer.decode(sequences[i], skip_special_tokens=False)
            
            # Extract SQL
            if "[SQL]" in generated:
                sql_part = generated.split("[SQL]")[-1]
                if "<|im_end|>" in sql_part:
                    sql_part = sql_part.split("<|im_end|>")[0]
                sql = sql_part.strip()
            else:
                sql = generated.strip()
            
            # Compute sequence score (average log prob)
            if hasattr(outputs, 'scores') and outputs.scores:
                # Get log probs for generated tokens
                seq_scores = []
                for j, score in enumerate(outputs.scores):
                    if j < len(sequences[i]) - input_length:
                        token_id = sequences[i][input_length + j]
                        log_probs = torch.log_softmax(score[i], dim=-1)
                        seq_scores.append(log_probs[token_id].item())
                avg_score = sum(seq_scores) / len(seq_scores) if seq_scores else 0.0
            else:
                avg_score = 0.0
            
            candidates.append((sql, avg_score))
    
    # Remove duplicates and sort by score
    seen = set()
    unique_candidates = []
    for sql, score in candidates:
        sql_normalized = " ".join(sql.lower().split())
        if sql_normalized not in seen:
            seen.add(sql_normalized)
            unique_candidates.append((sql, score))
    
    unique_candidates.sort(key=lambda x: x[1], reverse=True)
    
    return unique_candidates


def validate_sql_syntax(sql: str) -> Tuple[bool, str]:
    """
    Validate SQL syntax using sqlparse.
    
    Args:
        sql: SQL query string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        import sqlparse
        parsed = sqlparse.parse(sql)
        if not parsed or not parsed[0].tokens:
            return False, "Empty or invalid SQL"
        
        # Basic validation - check for common SQL keywords
        sql_upper = sql.upper()
        has_select = "SELECT" in sql_upper
        has_from = "FROM" in sql_upper
        
        if not has_select:
            return False, "Missing SELECT keyword"
        if not has_from and "COUNT(" not in sql_upper:
            return False, "Missing FROM keyword"
            
        return True, ""
    except Exception as e:
        return False, str(e)


def execute_sql_on_schema(
    sql: str,
    schema: str,
    sample_data: Optional[Dict[str, List[Dict]]] = None
) -> Tuple[bool, Any, str]:
    """
    Execute SQL query on a mock database created from schema.
    
    Args:
        sql: SQL query to execute
        schema: Schema text (used to create mock tables)
        sample_data: Optional dict of {table_name: [row_dicts]} for test data
        
    Returns:
        Tuple of (success, result, error_message)
    """
    try:
        import duckdb
        
        # Create in-memory database
        conn = duckdb.connect(":memory:")
        
        # Parse schema to create tables
        tables_created = _create_tables_from_schema(conn, schema, sample_data)
        
        if not tables_created:
            return False, None, "Failed to create tables from schema"
        
        # Execute the query
        try:
            result = conn.execute(sql).fetchall()
            conn.close()
            return True, result, ""
        except Exception as e:
            conn.close()
            return False, None, f"Execution error: {str(e)}"
            
    except ImportError:
        return False, None, "DuckDB not installed"
    except Exception as e:
        return False, None, f"Error: {str(e)}"


def _create_tables_from_schema(conn, schema: str, sample_data: Optional[Dict] = None) -> bool:
    """
    Create tables in DuckDB from schema text.
    
    This parses the [TABLES] section and creates corresponding tables.
    """
    try:
        lines = schema.split("\n")
        current_table = None
        columns = []
        tables = {}
        
        in_tables_section = False
        
        for line in lines:
            line = line.strip()
            
            if line == "[TABLES]":
                in_tables_section = True
                continue
            elif line.startswith("[") and line.endswith("]"):
                # Save previous table
                if current_table and columns:
                    tables[current_table] = columns
                in_tables_section = False
                current_table = None
                columns = []
                continue
            
            if not in_tables_section:
                continue
                
            if line.endswith(":"):
                # Save previous table
                if current_table and columns:
                    tables[current_table] = columns
                current_table = line[:-1].strip()
                columns = []
            elif line and current_table:
                # Parse column: "column_name (PK)" or "column_name (FK)" or just "column_name"
                col_name = line.split("(")[0].strip()
                if col_name:
                    columns.append(col_name)
        
        # Save last table
        if current_table and columns:
            tables[current_table] = columns
        
        # Create tables in DuckDB
        for table_name, cols in tables.items():
            # Create table with TEXT columns (simplest approach)
            col_defs = ", ".join([f'"{col}" TEXT' for col in cols])
            create_sql = f'CREATE TABLE "{table_name}" ({col_defs})'
            conn.execute(create_sql)
            
            # Insert sample data if provided
            if sample_data and table_name in sample_data:
                for row in sample_data[table_name]:
                    values = ", ".join([f"'{row.get(col, '')}'" for col in cols])
                    insert_sql = f'INSERT INTO "{table_name}" VALUES ({values})'
                    try:
                        conn.execute(insert_sql)
                    except:
                        pass  # Skip invalid rows
        
        return len(tables) > 0
        
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False


def generate_sql_with_egd(
    model,
    tokenizer,
    question: str,
    schema: str,
    num_candidates: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    sample_data: Optional[Dict[str, List[Dict]]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate SQL using Execution-Guided Decoding (EGD).
    使用执行引导解码 (EGD) 生成 SQL。
    
    Function / 功能:
        Main EGD function. Generates multiple candidates, validates each
        through syntax checking and execution, selects the best one.
        EGD 主函数。生成多个候选，通过语法检查和执行验证每个候选，选择最佳的。
    
    Called by / 调用者:
        - pipeline.ipynb: EGD testing section (EGD 测试部分)
        - evaluate_with_egd(): Batch EGD evaluation (批量 EGD 评估)
    
    Pipeline / 流程:
        1. Generate k SQL candidates using sampling (使用采样生成 k 个候选)
        2. Validate syntax of each candidate (验证每个候选的语法)
        3. Try to execute each on mock DB from schema (在根据模式创建的模拟数据库上执行)
        4. Return best: executed > syntax_valid > highest_score (返回最佳候选)
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        question: Natural language question
        schema: Database schema
        num_candidates: Number of candidates to generate
        max_new_tokens: Maximum tokens per candidate
        temperature: Sampling temperature
        sample_data: Optional sample data for execution testing
        verbose: Whether to print progress
        
    Returns:
        Dict with:
        - 'sql': Best SQL query
        - 'method': How it was selected ('executed', 'syntax_valid', 'highest_score')
        - 'candidates': List of all candidates with their status
        - 'executed': Whether the selected SQL was successfully executed
    """
    if verbose:
        print("=" * 60)
        print("EXECUTION-GUIDED DECODING (EGD)")
        print("=" * 60)
        print(f"\nQuestion: {question}")
        print(f"Generating {num_candidates} candidates...")
    
    # Step 1: Generate candidates
    candidates = generate_sql_candidates(
        model, tokenizer, question, schema,
        num_candidates=num_candidates,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    
    if verbose:
        print(f"\nGenerated {len(candidates)} unique candidates")
    
    # Step 2: Validate and rank candidates
    candidate_results = []
    best_executed = None
    best_syntax_valid = None
    
    for i, (sql, score) in enumerate(candidates):
        result = {
            "sql": sql,
            "score": score,
            "syntax_valid": False,
            "executed": False,
            "execution_result": None,
            "error": None,
        }
        
        # Check syntax
        syntax_valid, syntax_error = validate_sql_syntax(sql)
        result["syntax_valid"] = syntax_valid
        
        if not syntax_valid:
            result["error"] = syntax_error
            candidate_results.append(result)
            continue
        
        # Track first syntax-valid candidate
        if best_syntax_valid is None:
            best_syntax_valid = sql
        
        # Try to execute
        executed, exec_result, exec_error = execute_sql_on_schema(sql, schema, sample_data)
        result["executed"] = executed
        result["execution_result"] = exec_result
        
        if not executed:
            result["error"] = exec_error
        elif best_executed is None:
            best_executed = sql
        
        candidate_results.append(result)
        
        if verbose:
            status = "✓ Executed" if executed else ("✓ Valid syntax" if syntax_valid else "✗ Invalid")
            print(f"  [{i+1}] {status}: {sql[:60]}...")
    
    # Step 3: Select best candidate
    if best_executed:
        selected_sql = best_executed
        method = "executed"
    elif best_syntax_valid:
        selected_sql = best_syntax_valid
        method = "syntax_valid"
    elif candidates:
        selected_sql = candidates[0][0]  # Highest score
        method = "highest_score"
    else:
        selected_sql = ""
        method = "none"
    
    if verbose:
        print(f"\nSelected ({method}): {selected_sql}")
    
    return {
        "sql": selected_sql,
        "method": method,
        "candidates": candidate_results,
        "executed": method == "executed",
    }


def evaluate_with_egd(
    model,
    tokenizer,
    eval_data: List[Dict],
    max_samples: Optional[int] = None,
    num_candidates: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate model using Execution-Guided Decoding.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        eval_data: Evaluation dataset
        max_samples: Maximum samples to evaluate
        num_candidates: Candidates per sample
        verbose: Print progress
        
    Returns:
        Evaluation results with EGD metrics
    """
    if max_samples:
        eval_data = eval_data[:max_samples]
    
    results = []
    correct = 0
    egd_improved = 0  # Cases where EGD found executable SQL
    
    if verbose:
        print(f"\nEvaluating {len(eval_data)} samples with EGD (k={num_candidates})...")
    
    for i, example in enumerate(eval_data):
        # Generate with EGD
        egd_result = generate_sql_with_egd(
            model, tokenizer,
            example["question"],
            example["schema"],
            num_candidates=num_candidates,
            verbose=False
        )
        
        pred_sql = egd_result["sql"]
        
        # Compare with gold
        gold_norm = normalize_sql(example["sql"])
        pred_norm = normalize_sql(pred_sql)
        is_match = gold_norm == pred_norm
        
        if is_match:
            correct += 1
        
        if egd_result["executed"]:
            egd_improved += 1
        
        results.append({
            "question": example["question"],
            "gold_sql": example["sql"],
            "pred_sql": pred_sql,
            "match": is_match,
            "egd_method": egd_result["method"],
            "egd_executed": egd_result["executed"],
        })
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(eval_data)}] EM: {100*correct/(i+1):.1f}%, Executed: {100*egd_improved/(i+1):.1f}%")
    
    accuracy = 100 * correct / len(eval_data) if eval_data else 0
    exec_rate = 100 * egd_improved / len(eval_data) if eval_data else 0
    
    if verbose:
        print(f"\n Final Results:")
        print(f"  Exact Match: {accuracy:.2f}%")
        print(f"  Execution Rate: {exec_rate:.2f}%")
    
    return {
        "accuracy": accuracy,
        "execution_rate": exec_rate,
        "correct": correct,
        "executed": egd_improved,
        "total": len(eval_data),
        "results": results,
    }

