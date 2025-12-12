"""
Testing Utilities for NL2SQL Evaluation.

This module encapsulates all testing and evaluation functions for use in notebooks
and scripts. It provides a clean interface for:
- Loading fine-tuned models with LoRA adapters
- Generating SQL from natural language questions
- Evaluating models on datasets (Spider, WikiSQL)
- Interactive testing
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
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        question: Natural language question
        schema: Database schema in text format
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        
    Returns:
        Generated SQL query string
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
    
    Handles:
    - Case insensitivity (SELECT = select)
    - Whitespace normalization (multiple spaces -> single space)
    - Quote unification (double quotes -> single quotes)
    """
    sql = sql.lower()                    # 忽略大小写
    sql = sql.replace('"', "'")          # 统一引号为单引号
    sql = " ".join(sql.split())          # 规范化空格
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
    Evaluate model on a dataset.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        eval_data: List of evaluation examples with 'question', 'schema', 'sql' keys
        max_samples: Maximum number of samples to evaluate (None = all)
        max_new_tokens: Maximum tokens to generate per sample
        verbose: Whether to print progress
        
    Returns:
        Dictionary with evaluation results including accuracy, correct count,
        total count, and detailed results
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

