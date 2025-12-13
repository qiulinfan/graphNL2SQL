"""
Testing Utilities for NL2SQL Evaluation.
NL2SQL 评估测试工具模块

This module encapsulates all testing and evaluation functions for use in 
notebooks and scripts.
本模块封装了所有测试和评估相关函数，供 notebook 和脚本使用。

Main Functions / 主要功能:
- generate_sql(): Generate SQL from question and schema / 从问题和模式生成 SQL
- load_finetuned_model(): Load model with LoRA adapters / 加载带 LoRA 的模型
- evaluate_model(): Evaluate on dataset with Exact Match (EM) / 在数据集上评估精确匹配
- evaluate_with_execution(): Evaluate with both EM and Execution Match (EX) / 同时评估 EM 和执行匹配
- generate_sql_with_egd(): Execution-Guided Decoding / 执行引导解码
- evaluate_with_egd(): Evaluate using EGD method / 使用 EGD 方法评估
- test_all_checkpoints(): Test all checkpoints and report EM/EX / 测试所有checkpoint并报告EM/EX

Metrics / 评估指标:
- Exact Match (EM): Normalized SQL string comparison / 规范化 SQL 字符串比较
- Execution Match (EX): Compare execution results on mock DB / 在模拟数据库上比较执行结果

Called by / 调用者:
- pipeline.ipynb: Testing section / 测试部分
- inference.py: Command-line inference / 命令行推理
"""

import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

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
    use_egd: bool = False,
    egd_candidates: int = 5,
) -> str:
    """
    Generate SQL from question and schema.
    从问题和模式生成 SQL 查询。
    
    Function / 功能:
        Core inference function. Formats input as chat prompt, generates SQL
        using the model, and extracts the SQL from the response.
        Optionally uses EGD for better accuracy.
        核心推理函数。将输入格式化为对话提示，使用模型生成 SQL，并从响应中提取 SQL。
        可选使用 EGD 以提高准确率。
    
    Called by / 调用者:
        - evaluate_model(): For batch evaluation (批量评估时调用)
        - evaluate_with_execution(): For EM+EX evaluation (EM+EX 评估时调用)
        - run_quick_test(): For single-sample testing (单样本测试时调用)
    
    Args / 参数:
        model: The language model (语言模型)
        tokenizer: The tokenizer (分词器)
        question: Natural language question (自然语言问题)
        schema: Database schema in text format (文本格式的数据库模式)
        max_new_tokens: Maximum tokens to generate (最大生成 token 数)
        temperature: Sampling temperature (采样温度)
        do_sample: Whether to use sampling (是否使用采样)
        use_egd: Whether to use Execution-Guided Decoding (是否使用 EGD)
        egd_candidates: Number of candidates for EGD (EGD 候选数量)
        
    Returns / 返回:
        str: Generated SQL query (生成的 SQL 查询)
    """
    # Use EGD if requested
    # 如果请求则使用 EGD
    if use_egd:
        egd_result = generate_sql_with_egd(
            model, tokenizer, question, schema,
            num_candidates=egd_candidates,
            max_new_tokens=max_new_tokens,
            verbose=False
        )
        return egd_result["sql"]
    
    # Try to use improved prompt templates
    try:
        from prompt_templates import PromptStyle, get_system_message, format_for_inference
        use_improved_prompt = True
    except ImportError:
        use_improved_prompt = False
    
    if use_improved_prompt:
        # Use improved prompt system
        system_msg = get_system_message(PromptStyle.DETAILED)
        user_input = format_for_inference(
            schema=schema,
            question=question,
            style=PromptStyle.DETAILED,
        )
    else:
        # Fall back to simple prompt
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
    use_egd: bool = False,
    egd_candidates: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset using Exact Match metric.
    使用精确匹配指标在数据集上评估模型。
    
    Function / 功能:
        Iterates through evaluation data, generates SQL for each example,
        compares with gold SQL using normalized comparison, and computes accuracy.
        Optionally uses EGD for better accuracy.
        遍历评估数据，为每个样本生成 SQL，使用规范化比较与标准 SQL 对比，计算准确率。
        可选使用 EGD 以提高准确率。
    
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
        use_egd: Whether to use Execution-Guided Decoding (是否使用 EGD)
        egd_candidates: Number of candidates for EGD (EGD 候选数量)
        
    Returns / 返回:
        Dict with keys: accuracy, correct, total, results
        包含 accuracy, correct, total, results 的字典
    """
    if max_samples:
        eval_data = eval_data[:max_samples]
    
    results = []
    correct = 0
    
    if verbose:
        mode = "EGD" if use_egd else "Standard"
        print(f"Evaluating on {len(eval_data)} samples ({mode} mode)...")
    
    for i, example in enumerate(eval_data):
        # Generate SQL (with optional EGD)
        pred_sql = generate_sql(
            model, tokenizer,
            example["question"],
            example["schema"],
            max_new_tokens=max_new_tokens,
            use_egd=use_egd,
            egd_candidates=egd_candidates,
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


def compare_execution_results(
    result1: Any,
    result2: Any,
) -> bool:
    """
    Compare two SQL execution results for equivalence.
    比较两个 SQL 执行结果是否等价。
    
    Function / 功能:
        Compares query results as sets (ignoring row order) to determine
        if two different SQL queries produce equivalent results.
        将查询结果作为集合比较（忽略行顺序），判断两个不同的 SQL 是否产生等价结果。
    
    Called by / 调用者:
        - evaluate_with_execution(): For execution match evaluation (执行匹配评估)
    
    Args / 参数:
        result1: First execution result (rows as tuples) (第一个执行结果)
        result2: Second execution result (rows as tuples) (第二个执行结果)
        
    Returns / 返回:
        bool: True if results are equivalent (结果等价返回 True)
    """
    if result1 is None or result2 is None:
        return False
    
    # Convert to sets of tuples for order-independent comparison
    # 转换为元组集合以进行顺序无关的比较
    try:
        # Handle case where results might be lists of lists or tuples
        set1 = set(tuple(row) if isinstance(row, (list, tuple)) else (row,) for row in result1)
        set2 = set(tuple(row) if isinstance(row, (list, tuple)) else (row,) for row in result2)
        return set1 == set2
    except (TypeError, ValueError):
        # If unhashable, fall back to sorted comparison
        # 如果不可哈希，退回到排序比较
        try:
            sorted1 = sorted([str(row) for row in result1])
            sorted2 = sorted([str(row) for row in result2])
            return sorted1 == sorted2
        except:
            return False


def evaluate_with_execution(
    model,
    tokenizer,
    eval_data: List[Dict],
    max_samples: Optional[int] = None,
    max_new_tokens: int = 256,
    verbose: bool = True,
    use_egd: bool = False,
    egd_candidates: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate model using both Exact Match (EM) and Execution Match (EX).
    使用精确匹配 (EM) 和执行匹配 (EX) 评估模型。
    
    Function / 功能:
        Evaluates generated SQL queries using two metrics:
        使用两种指标评估生成的 SQL 查询：
        1. Exact Match (EM): Normalized string comparison (精确匹配：规范化字符串比较)
        2. Execution Match (EX): Compare execution results on mock DB (执行匹配：在模拟数据库上比较执行结果)
        
        EX is more forgiving as different SQL can produce same results.
        执行匹配更宽容，因为不同的 SQL 可以产生相同的结果。
        
        Optionally uses EGD for better accuracy.
        可选使用 EGD 以提高准确率。
    
    Called by / 调用者:
        - pipeline.ipynb: Comprehensive evaluation (综合评估)
    
    Args / 参数:
        model: The language model (语言模型)
        tokenizer: The tokenizer (分词器)
        eval_data: List of dicts with 'question', 'schema', 'sql' keys
                   包含 question, schema, sql 键的字典列表
        max_samples: Maximum samples to evaluate (最大评估样本数)
        max_new_tokens: Max tokens per generation (每次生成的最大 token 数)
        verbose: Print progress updates (是否打印进度)
        use_egd: Whether to use Execution-Guided Decoding (是否使用 EGD)
        egd_candidates: Number of candidates for EGD (EGD 候选数量)
        
    Returns / 返回:
        Dict with keys:
        返回字典包含：
        - exact_match_accuracy: EM accuracy (精确匹配准确率)
        - execution_match_accuracy: EX accuracy (执行匹配准确率)
        - exact_match_count: Number of EM correct (精确匹配正确数)
        - execution_match_count: Number of EX correct (执行匹配正确数)
        - total: Total samples (总样本数)
        - results: Detailed per-sample results (每个样本的详细结果)
    """
    try:
        import duckdb
        has_duckdb = True
    except ImportError:
        has_duckdb = False
        if verbose:
            print("Warning: DuckDB not installed. Execution match will be skipped.")
            print("Install with: pip install duckdb")
    
    if max_samples:
        eval_data = eval_data[:max_samples]
    
    results = []
    em_correct = 0  # Exact match count
    ex_correct = 0  # Execution match count
    exec_errors = 0  # Count of execution failures
    
    if verbose:
        mode = "EGD" if use_egd else "Standard"
        print(f"Evaluating {len(eval_data)} samples with EM + EX ({mode} mode)...")
        print("-" * 60)
    
    for i, example in enumerate(eval_data):
        # Generate SQL (with optional EGD)
        pred_sql = generate_sql(
            model, tokenizer,
            example["question"],
            example["schema"],
            max_new_tokens=max_new_tokens,
            use_egd=use_egd,
            egd_candidates=egd_candidates,
        )
        
        gold_sql = example["sql"]
        schema = example["schema"]
        
        # Exact Match (EM)
        gold_norm = normalize_sql(gold_sql)
        pred_norm = normalize_sql(pred_sql)
        is_em = gold_norm == pred_norm
        
        if is_em:
            em_correct += 1
        
        # Execution Match (EX)
        # If EM is True, EX should also be True (same SQL = same results)
        # 如果 EM 为 True，EX 也应该为 True（相同 SQL = 相同结果）
        is_ex = is_em  # Start with EM result
        gold_result = None
        pred_result = None
        gold_error = None
        pred_error = None
        
        # Only try execution if EM is False (to find additional matches)
        # 只有当 EM 为 False 时才尝试执行（以发现额外的匹配）
        if not is_em and has_duckdb:
            # Execute gold SQL
            gold_success, gold_result, gold_error = execute_sql_on_schema(gold_sql, schema)
            
            # Execute predicted SQL
            pred_success, pred_result, pred_error = execute_sql_on_schema(pred_sql, schema)
            
            if gold_success and pred_success:
                # Compare results
                is_ex = compare_execution_results(gold_result, pred_result)
            elif not gold_success and not pred_success:
                # Both failed - could be schema parsing issue
                exec_errors += 1
                # Store error info for debugging
                if verbose and exec_errors <= 3:  # Only show first few errors
                    print(f"    [Debug] Both SQLs failed execution:")
                    print(f"      Gold error: {gold_error[:100] if gold_error else 'None'}")
                    print(f"      Pred error: {pred_error[:100] if pred_error else 'None'}")
            # else: One succeeded, one failed - not a match (is_ex stays False)
        
        if is_ex:
            ex_correct += 1
        
        result_entry = {
            "question": example["question"],
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "exact_match": is_em,
            "execution_match": is_ex,
            "gold_exec_result": gold_result,
            "pred_exec_result": pred_result,
            "gold_exec_error": gold_error,
            "pred_exec_error": pred_error,
        }
        results.append(result_entry)
        
        if verbose and (i + 1) % 10 == 0:
            em_acc = 100 * em_correct / (i + 1)
            ex_acc = 100 * ex_correct / (i + 1)
            print(f"  [{i+1}/{len(eval_data)}] EM: {em_acc:.1f}%, EX: {ex_acc:.1f}%")
    
    em_accuracy = 100 * em_correct / len(eval_data) if eval_data else 0
    ex_accuracy = 100 * ex_correct / len(eval_data) if eval_data else 0
    
    if verbose:
        print("-" * 60)
        print(f"Final Results:")
        print(f"  Exact Match (EM):     {em_accuracy:.2f}% ({em_correct}/{len(eval_data)})")
        print(f"  Execution Match (EX): {ex_accuracy:.2f}% ({ex_correct}/{len(eval_data)})")
        if exec_errors > 0:
            print(f"  Execution Errors:     {exec_errors} (schema parsing issues)")
        
        # Show improvement from EX over EM
        ex_only = sum(1 for r in results if r["execution_match"] and not r["exact_match"])
        if ex_only > 0:
            print(f"\n  EX found {ex_only} additional correct queries beyond EM")
    
    return {
        "exact_match_accuracy": em_accuracy,
        "execution_match_accuracy": ex_accuracy,
        "exact_match_count": em_correct,
        "execution_match_count": ex_correct,
        "execution_errors": exec_errors,
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
    temperature: float = 0.3,  # Reduced from 0.7 to improve quality
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
        # Strategy: First candidate uses greedy (best quality), rest use sampling (diversity)
        # 策略：第一个候选使用贪婪解码（最佳质量），其余使用采样（多样性）
        
        # Generate first candidate with greedy decoding (temperature=0, do_sample=False)
        # 使用贪婪解码生成第一个候选（最高质量）
        outputs_greedy = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        
        # Generate remaining candidates with sampling (if num_candidates > 1)
        # 使用采样生成其余候选（如果候选数 > 1）
        if num_candidates > 1:
            outputs_sampled = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=num_candidates - 1,
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
            # Combine sequences: greedy first, then sampled
            # 合并序列：贪婪解码的在前，然后是采样的
            sequences = torch.cat([outputs_greedy.sequences, outputs_sampled.sequences], dim=0)
            # Combine scores for score computation
            # 合并分数用于计算
            all_scores = outputs_greedy.scores if hasattr(outputs_greedy, 'scores') else []
            if hasattr(outputs_sampled, 'scores') and outputs_sampled.scores:
                # Note: scores structure might be different, we'll compute from sequences
                pass
        else:
            sequences = outputs_greedy.sequences
        
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
            
            # Compute sequence score
            # For greedy (i=0), assign highest score; for sampled, use decreasing score
            # 对于贪婪解码（i=0），分配最高分数；对于采样，使用递减分数
            if i == 0:
                # Greedy candidate gets highest score (1.0)
                # 贪婪候选获得最高分数（1.0）
                avg_score = 1.0
            else:
                # For sampled candidates, use position-based score (earlier = higher)
                # 对于采样候选，使用基于位置的分数（越早 = 越高）
                avg_score = 1.0 - (i - 1) * 0.1  # Decreasing score for later candidates
            
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
    Enhanced to handle edge cases and provide better error messages.
    """
    try:
        lines = schema.split("\n")
        current_table = None
        columns = []
        tables = {}
        
        in_tables_section = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            if line == "[TABLES]":
                in_tables_section = True
                continue
            elif line.startswith("[") and line.endswith("]"):
                # Save previous table before leaving section
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
                # Handle table names with quotes or special characters
                if current_table.startswith('"') and current_table.endswith('"'):
                    current_table = current_table[1:-1]
                columns = []
            elif line and current_table:
                # Parse column: handle various formats
                # "column_name (PK)" or "column_name (FK)" or "    column_name" or just "column_name"
                # Remove leading indentation (4 spaces or tabs)
                col_line = line.lstrip(" \t")
                
                # Extract column name (before any parentheses or markers)
                if "(" in col_line:
                    col_name = col_line.split("(")[0].strip()
                else:
                    col_name = col_line.strip()
                
                # Handle quoted column names
                if col_name.startswith('"') and col_name.endswith('"'):
                    col_name = col_name[1:-1]
                
                # Skip if empty or looks like a comment
                if col_name and not col_name.startswith("#") and not col_name.startswith("--"):
                    columns.append(col_name)
        
        # Save last table
        if current_table and columns:
            tables[current_table] = columns
        
        if len(tables) == 0:
            # Try fallback: look for any table-like patterns
            # This handles schemas that might not have [TABLES] section
            for line in lines:
                line = line.strip()
                if ":" in line and not line.startswith("[") and not line.startswith("#"):
                    parts = line.split(":")
                    if len(parts) == 2:
                        table_name = parts[0].strip()
                        col_info = parts[1].strip()
                        if table_name and col_info:
                            if table_name not in tables:
                                tables[table_name] = []
                            # Try to extract column name
                            col_name = col_info.split("(")[0].strip()
                            if col_name:
                                tables[table_name].append(col_name)
        
        if len(tables) == 0:
            return False
        
        # Create tables in DuckDB
        for table_name, cols in tables.items():
            if not cols:
                # Skip tables with no columns
                continue
                
            try:
                # Create table with TEXT columns (simplest approach)
                # Escape table and column names properly
                safe_table_name = table_name.replace('"', '""')
                col_defs = ", ".join([f'"{col.replace(chr(34), chr(34)+chr(34))}" TEXT' for col in cols])
                create_sql = f'CREATE TABLE "{safe_table_name}" ({col_defs})'
                conn.execute(create_sql)
                
                # Insert sample data if provided
                if sample_data and table_name in sample_data:
                    for row in sample_data[table_name]:
                        try:
                            # Build values list, handling missing columns
                            values = []
                            for col in cols:
                                val = row.get(col, '')
                                # Escape single quotes in values
                                val_str = str(val).replace("'", "''")
                                values.append(f"'{val_str}'")
                            if values:
                                values_str = ", ".join(values)
                                insert_sql = f'INSERT INTO "{safe_table_name}" VALUES ({values_str})'
                                conn.execute(insert_sql)
                        except Exception as e:
                            # Skip invalid rows silently
                            pass
            except Exception as e:
                # Log but continue with other tables
                print(f"Warning: Failed to create table '{table_name}': {e}")
                continue
        
        return len([t for t in tables.values() if t]) > 0
        
    except Exception as e:
        print(f"Error creating tables from schema: {e}")
        return False


def generate_sql_with_egd(
    model,
    tokenizer,
    question: str,
    schema: str,
    num_candidates: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 0.3,  # Reduced from 0.7 to improve quality
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
    executed_candidates = []  # List of (sql, score) for executed candidates
    syntax_valid_candidates = []  # List of (sql, score) for syntax-valid candidates
    
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
        
        # Track syntax-valid candidates with their scores
        syntax_valid_candidates.append((sql, score))
        
        # Try to execute
        executed, exec_result, exec_error = execute_sql_on_schema(sql, schema, sample_data)
        result["executed"] = executed
        result["execution_result"] = exec_result
        
        if not executed:
            result["error"] = exec_error
        else:
            # Track executed candidates with their scores
            executed_candidates.append((sql, score))
        
        candidate_results.append(result)
        
        if verbose:
            status = "✓ Executed" if executed else ("✓ Valid syntax" if syntax_valid else "✗ Invalid")
            print(f"  [{i+1}] {status}: {sql[:60]}...")
    
    # Step 3: Select best candidate
    # Priority: 
    #   1. First candidate (greedy) if syntax valid (highest quality)
    #   2. Executed candidate with highest score
    #   3. Syntax-valid candidate with highest score
    #   4. Highest score candidate
    # 优先级：
    #   1. 第一个候选（贪婪解码）如果语法正确（最高质量）
    #   2. 可执行的候选中分数最高的
    #   3. 语法正确的候选中分数最高的
    #   4. 分数最高的候选
    
    # Check if first candidate (greedy, highest quality) is syntax valid
    # 检查第一个候选（贪婪解码，最高质量）是否语法正确
    if candidates and len(candidates) > 0:
        first_sql, first_score = candidates[0]
        # Find the result for first candidate
        first_result = next((r for r in candidate_results if r["sql"] == first_sql), None)
        if first_result and first_result["syntax_valid"]:
            # Prefer first candidate if it's syntax valid (even if execution failed)
            # 如果第一个候选语法正确，优先选择它（即使执行失败）
            # Execution failure might be due to schema parsing issues, not SQL correctness
            # 执行失败可能是由于 schema 解析问题，而不是 SQL 正确性问题
            selected_sql = first_sql
            method = "first_valid" if first_result["executed"] else "first_syntax_valid"
        elif executed_candidates:
            # Select executed candidate with highest score
            executed_candidates.sort(key=lambda x: x[1], reverse=True)
            selected_sql = executed_candidates[0][0]
            method = "executed"
        elif syntax_valid_candidates:
            # Select syntax-valid candidate with highest score
            syntax_valid_candidates.sort(key=lambda x: x[1], reverse=True)
            selected_sql = syntax_valid_candidates[0][0]
            method = "syntax_valid"
        else:
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
    Evaluate model using Execution-Guided Decoding with EM and EX metrics.
    使用执行引导解码评估模型，包含精确匹配和执行匹配指标。
    
    Function / 功能:
        Combines EGD generation with comprehensive evaluation:
        结合 EGD 生成与综合评估：
        1. Generate SQL using EGD (k candidates, select best executable)
           使用 EGD 生成 SQL（k 个候选，选择最佳可执行的）
        2. Compute Exact Match (EM) - string comparison
           计算精确匹配 (EM) - 字符串比较
        3. Compute Execution Match (EX) - result comparison
           计算执行匹配 (EX) - 结果比较
    
    Called by / 调用者:
        - pipeline.ipynb: EGD evaluation section (EGD 评估部分)
    
    Args / 参数:
        model: The language model (语言模型)
        tokenizer: The tokenizer (分词器)
        eval_data: Evaluation dataset (评估数据集)
        max_samples: Maximum samples to evaluate (最大评估样本数)
        num_candidates: Candidates per sample (每个样本的候选数)
        verbose: Print progress (是否打印进度)
        
    Returns / 返回:
        Dict with EM accuracy, EX accuracy, execution rate, and detailed results
        包含 EM 准确率、EX 准确率、执行率和详细结果的字典
    """
    try:
        import duckdb
        has_duckdb = True
    except ImportError:
        has_duckdb = False
        if verbose:
            print("Warning: DuckDB not installed. Execution match will be skipped.")
    
    if max_samples:
        eval_data = eval_data[:max_samples]
    
    results = []
    em_correct = 0       # Exact match count
    ex_correct = 0       # Execution match count
    egd_executed = 0     # Cases where EGD found executable SQL
    
    if verbose:
        print(f"\nEvaluating {len(eval_data)} samples with EGD (k={num_candidates})...")
        print("-" * 60)
    
    for i, example in enumerate(eval_data):
        gold_sql = example["sql"]
        schema = example["schema"]
        
        # Generate with EGD
        egd_result = generate_sql_with_egd(
            model, tokenizer,
            example["question"],
            schema,
            num_candidates=num_candidates,
            verbose=False
        )
        
        pred_sql = egd_result["sql"]
        
        # Exact Match (EM)
        gold_norm = normalize_sql(gold_sql)
        pred_norm = normalize_sql(pred_sql)
        is_em = gold_norm == pred_norm
        
        if is_em:
            em_correct += 1
        
        if egd_result["executed"]:
            egd_executed += 1
        
        # Execution Match (EX) - compare gold vs pred results
        is_ex = False
        gold_result = None
        pred_result = None
        
        if has_duckdb:
            gold_success, gold_result, _ = execute_sql_on_schema(gold_sql, schema)
            pred_success, pred_result, _ = execute_sql_on_schema(pred_sql, schema)
            
            if gold_success and pred_success:
                is_ex = compare_execution_results(gold_result, pred_result)
        
        if is_ex:
            ex_correct += 1
        
        results.append({
            "question": example["question"],
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "exact_match": is_em,
            "execution_match": is_ex,
            "egd_method": egd_result["method"],
            "egd_executed": egd_result["executed"],
        })
        
        if verbose and (i + 1) % 10 == 0:
            em_acc = 100 * em_correct / (i + 1)
            ex_acc = 100 * ex_correct / (i + 1)
            exec_rate = 100 * egd_executed / (i + 1)
            print(f"  [{i+1}/{len(eval_data)}] EM: {em_acc:.1f}%, EX: {ex_acc:.1f}%, Exec: {exec_rate:.1f}%")
    
    em_accuracy = 100 * em_correct / len(eval_data) if eval_data else 0
    ex_accuracy = 100 * ex_correct / len(eval_data) if eval_data else 0
    exec_rate = 100 * egd_executed / len(eval_data) if eval_data else 0
    
    if verbose:
        print("-" * 60)
        print(f"Final Results:")
        print(f"  Exact Match (EM):     {em_accuracy:.2f}% ({em_correct}/{len(eval_data)})")
        print(f"  Execution Match (EX): {ex_accuracy:.2f}% ({ex_correct}/{len(eval_data)})")
        print(f"  EGD Execution Rate:   {exec_rate:.2f}% ({egd_executed}/{len(eval_data)})")
        
        # Show improvement
        ex_only = sum(1 for r in results if r["execution_match"] and not r["exact_match"])
        if ex_only > 0:
            print(f"\n  EX found {ex_only} additional correct queries beyond EM")
    
    return {
        "exact_match_accuracy": em_accuracy,
        "execution_match_accuracy": ex_accuracy,
        "execution_rate": exec_rate,
        "exact_match_count": em_correct,
        "execution_match_count": ex_correct,
        "egd_executed_count": egd_executed,
        "total": len(eval_data),
        "results": results,
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load data from JSONL file.
    从JSONL文件加载数据。
    
    Args / 参数:
        file_path: Path to JSONL file (JSONL文件路径)
        
    Returns / 返回:
        List of dictionaries (字典列表)
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data


# =============================================================================
# CHECKPOINT EVALUATION
# =============================================================================

def test_all_checkpoints(
    checkpoint_dir: str,
    eval_data: List[Dict],
    base_model_name: Optional[str] = None,
    max_samples: Optional[int] = None,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    use_egd: bool = False,
    egd_candidates: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Test all checkpoints in a directory and report EM and EX accuracy for each.
    测试目录中的所有checkpoint，并报告每个的EM和EX准确率。
    
    Function / 功能:
        Scans checkpoint directory, loads each checkpoint, evaluates on eval_data,
        and returns a summary sorted by performance.
        扫描checkpoint目录，加载每个checkpoint，在eval_data上评估，并返回按性能排序的摘要。
    
    Args / 参数:
        checkpoint_dir: Directory containing checkpoints (e.g., "./checkpoints/phase2_spider")
                       包含checkpoints的目录（例如："./checkpoints/phase2_spider"）
        eval_data: List of dicts with 'question', 'schema', 'sql' keys
                  包含 question, schema, sql 键的字典列表
        base_model_name: Base model name (auto-detected if None)
                        基础模型名称（如果为None则自动检测）
        max_samples: Maximum samples to evaluate per checkpoint
                     每个checkpoint评估的最大样本数
        load_in_4bit: Whether to use 4-bit quantization
                      是否使用4位量化
        load_in_8bit: Whether to use 8-bit quantization
                      是否使用8位量化
        use_egd: Whether to use Execution-Guided Decoding
                是否使用执行引导解码
        egd_candidates: Number of candidates for EGD
                       EGD的候选数量
        verbose: Print progress updates
                是否打印进度更新
    
    Returns / 返回:
        Dict with keys:
        返回字典包含：
        - summary: List of dicts with checkpoint name, step, EM, EX (按性能排序)
                  包含checkpoint名称、步数、EM、EX的字典列表（按性能排序）
        - best_em: Best checkpoint by EM accuracy (按EM准确率的最佳checkpoint)
        - best_ex: Best checkpoint by EX accuracy (按EX准确率的最佳checkpoint)
        - all_results: Dict mapping checkpoint path to full evaluation results
                        将checkpoint路径映射到完整评估结果的字典
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    if max_samples:
        eval_data = eval_data[:max_samples]
    
    # Find all checkpoints
    # 查找所有checkpoints
    checkpoints = []
    
    # Pattern 1: checkpoint-{step} directories
    # 模式1：checkpoint-{step} 目录
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            # Extract step number
            # 提取步数
            match = re.search(r'checkpoint-(\d+)', item.name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, item))
    
    # Pattern 2: "final" checkpoint
    # 模式2："final" checkpoint
    final_path = checkpoint_dir / "final"
    if final_path.exists() and final_path.is_dir():
        checkpoints.append((float('inf'), final_path))  # Sort last
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort by step number
    # 按步数排序
    checkpoints.sort(key=lambda x: x[0])
    
    if verbose:
        print("=" * 80)
        print(f"Testing {len(checkpoints)} checkpoints from {checkpoint_dir}")
        print("=" * 80)
        print()
    
    all_results = {}
    summary = []
    
    for idx, (step, checkpoint_path) in enumerate(checkpoints, 1):
        checkpoint_name = checkpoint_path.name
        step_str = "final" if step == float('inf') else str(step)
        
        if verbose:
            print(f"[{idx}/{len(checkpoints)}] Testing checkpoint: {checkpoint_name} (Step {step_str})")
            print("-" * 80)
        
        try:
            # Load model
            # 加载模型
            if verbose:
                print("  Loading model...")
            model, tokenizer = load_finetuned_model(
                str(checkpoint_path),
                base_model_name=base_model_name,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )
            
            # Evaluate
            # 评估
            if verbose:
                print(f"  Evaluating on {len(eval_data)} samples...")
            results = evaluate_with_execution(
                model, tokenizer, eval_data,
                max_samples=None,  # Already limited above
                verbose=True,  # Show progress during evaluation
                use_egd=use_egd,
                egd_candidates=egd_candidates,
            )
            
            em_acc = results["exact_match_accuracy"]
            ex_acc = results["execution_match_accuracy"]
            em_count = results["exact_match_count"]
            ex_count = results["execution_match_count"]
            total = results["total"]
            
            summary.append({
                "checkpoint": checkpoint_name,
                "step": step_str,
                "path": str(checkpoint_path),
                "em_accuracy": em_acc,
                "ex_accuracy": ex_acc,
                "em_count": em_count,
                "ex_count": ex_count,
                "total": total,
            })
            
            all_results[str(checkpoint_path)] = results
            
            if verbose:
                print(f"  ✓ EM: {em_acc:.2f}% ({em_count}/{total})")
                print(f"  ✓ EX: {ex_acc:.2f}% ({ex_count}/{total})")
                print()
            
            # Clean up memory
            # 清理内存
            if verbose:
                print("  Cleaning up memory...")
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if verbose:
                print()
            
        except Exception as e:
            if verbose:
                print(f"  ❌ Error: {e}")
                print()
            summary.append({
                "checkpoint": checkpoint_name,
                "step": step_str,
                "path": str(checkpoint_path),
                "em_accuracy": 0.0,
                "ex_accuracy": 0.0,
                "em_count": 0,
                "ex_count": 0,
                "total": total if 'total' in locals() else 0,
                "error": str(e),
            })
    
    # Sort by EX accuracy (primary), then EM accuracy (secondary)
    # 按EX准确率排序（主要），然后按EM准确率排序（次要）
    summary.sort(key=lambda x: (x["ex_accuracy"], x["em_accuracy"]), reverse=True)
    
    # Find best checkpoints
    # 查找最佳checkpoints
    best_em = max(summary, key=lambda x: x["em_accuracy"]) if summary else None
    best_ex = max(summary, key=lambda x: x["ex_accuracy"]) if summary else None
    
    # Print summary
    # 打印摘要
    if verbose:
        print("=" * 80)
        print("SUMMARY - All Checkpoints (Sorted by EX, then EM)")
        print("=" * 80)
        print(f"{'Rank':<6} {'Step':<10} {'Checkpoint':<30} {'EM %':<10} {'EX %':<10}")
        print("-" * 80)
        
        for rank, item in enumerate(summary, 1):
            step = item["step"]
            checkpoint = item["checkpoint"][:28]  # Truncate if too long
            em = item["em_accuracy"]
            ex = item["ex_accuracy"]
            
            marker = ""
            if item == best_ex:
                marker = " ⭐ (Best EX)"
            elif item == best_em:
                marker = " 🏆 (Best EM)"
            
            print(f"{rank:<6} {step:<10} {checkpoint:<30} {em:>6.2f}%   {ex:>6.2f}%{marker}")
        
        print("-" * 80)
        print()
        
        if best_ex:
            print(f"🏆 Best by EX: {best_ex['checkpoint']} (Step {best_ex['step']})")
            print(f"   EM: {best_ex['em_accuracy']:.2f}%, EX: {best_ex['ex_accuracy']:.2f}%")
            print(f"   Path: {best_ex['path']}")
            print()
        
        if best_em and best_em != best_ex:
            print(f"⭐ Best by EM: {best_em['checkpoint']} (Step {best_em['step']})")
            print(f"   EM: {best_em['em_accuracy']:.2f}%, EX: {best_em['ex_accuracy']:.2f}%")
            print(f"   Path: {best_em['path']}")
            print()
    
    return {
        "summary": summary,
        "best_em": best_em,
        "best_ex": best_ex,
        "all_results": all_results,
    }

