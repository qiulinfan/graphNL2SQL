"""
Advanced Prompt Templates for NL2SQL.

Provides multiple prompt styles with varying levels of detail and guidance.
"""

from typing import Optional, List, Dict
from enum import Enum


class PromptStyle(str, Enum):
    """Prompt style options."""
    SIMPLE = "simple"  # Current simple format
    DETAILED = "detailed"  # Detailed with guidelines
    FEW_SHOT = "few_shot"  # With examples
    EXPERT = "expert"  # Most detailed with best practices


def get_system_message(style: PromptStyle = PromptStyle.DETAILED) -> str:
    """
    Get system message based on style.
    
    Args:
        style: Prompt style (simple, detailed, few_shot, expert)
    
    Returns:
        System message string
    """
    if style == PromptStyle.SIMPLE:
        return (
            "You are a SQL expert. Given a database schema and a natural language question, "
            "generate the correct SQL query. Output only the SQL query."
        )
    
    elif style == PromptStyle.DETAILED:
        return """You are an expert SQL developer specializing in natural language to SQL translation.

Your task is to generate accurate, efficient SQL queries from natural language questions.

Key Requirements:
- Generate valid SQL syntax that executes without errors
- Use correct table and column names from the provided schema
- Handle JOINs correctly when multiple tables are involved
- **Use table aliases for JOINs** (e.g., `FROM table1 AS t1 JOIN table2 AS t2`)
- **Always qualify column names with table names/aliases** (e.g., `t1.column_name` instead of just `column_name`)
- Apply appropriate WHERE conditions based on the question
- Use correct aggregation functions (COUNT, SUM, AVG, MAX, MIN) when needed
- Include ORDER BY when sorting is requested
- Use LIMIT when the question asks for a specific number of results

Output only the SQL query without any explanation or markdown formatting."""

    elif style == PromptStyle.EXPERT:
        return """You are a senior SQL expert with deep expertise in database query optimization and natural language understanding.

Your mission is to translate natural language questions into precise, efficient SQL queries.

Core Principles:
1. **Accuracy First**: The SQL must correctly answer the question
2. **Schema Adherence**: Use exact table and column names from the schema
3. **Proper Joins**: 
   - Use INNER JOIN for required relationships
   - Use LEFT JOIN when all records from one table are needed
   - Match foreign keys correctly
   - **Always use table aliases** (e.g., `FROM singer AS s JOIN concert AS c`)
   - **Always qualify column names** with table aliases (e.g., `s.name`, `c.date`)
4. **Condition Logic**:
   - Translate question constraints to WHERE clauses accurately
   - Handle date/number comparisons correctly
   - Use appropriate operators (=, >, <, >=, <=, !=, LIKE, IN)
5. **Aggregations**:
   - COUNT(*) for counting records
   - SUM/AVG/MAX/MIN for numeric aggregations
   - Use GROUP BY when aggregating by categories
6. **Ordering & Limits**:
   - ORDER BY for sorting (ASC/DESC)
   - LIMIT for top-N queries
7. **Subqueries**: Use when needed for complex filtering or calculations

Common Pitfalls to Avoid:
- Don't use columns that don't exist in the schema
- Don't forget to join tables when accessing columns from multiple tables
- Don't mix up table aliases
- Don't use unqualified column names in multi-table queries (always use `table.column` or `alias.column`)
- Don't forget to define table aliases when using JOINs
- Don't use incorrect aggregation syntax
- Don't forget WHERE conditions mentioned in the question

Output Requirements:
- Output ONLY the SQL query
- No explanations, no markdown, no code blocks
- Use proper SQL formatting (keywords in uppercase recommended)
- Ensure the query is syntactically valid"""

    else:  # FEW_SHOT
        return """You are a SQL expert. Given a database schema and a natural language question, generate the correct SQL query.

I will provide you with examples to guide your understanding. Follow the same pattern for new questions.

Output only the SQL query without any explanation."""


def get_instruction(
    style: PromptStyle = PromptStyle.DETAILED,
    include_guidelines: bool = True
) -> str:
    """
    Get instruction text based on style.
    
    Args:
        style: Prompt style
        include_guidelines: Whether to include detailed guidelines
    
    Returns:
        Instruction string
    """
    if style == PromptStyle.SIMPLE:
        return (
            "Given the following database schema and question, "
            "generate the SQL query that answers the question."
        )
    
    base_instruction = (
        "Given the following database schema and question, "
        "generate the SQL query that answers the question."
    )
    
    if not include_guidelines or style == PromptStyle.SIMPLE:
        return base_instruction
    
    guidelines = """
    
Guidelines:
- Read the schema carefully to understand available tables and columns
- Identify which tables and columns are relevant to the question
- Determine if JOINs are needed (when accessing columns from multiple tables)
- **When using JOINs:**
  - Always define table aliases (e.g., `FROM singer AS s`)
  - Always qualify column names with aliases (e.g., `s.name`, `s.age`)
  - This prevents ambiguity and improves query clarity
- Translate question constraints into WHERE conditions
- Apply aggregations (COUNT, SUM, etc.) when the question asks for counts or totals
- Use ORDER BY for sorting requests
- Use LIMIT for "top N" or "first N" requests
- Ensure all column and table names match the schema exactly"""
    
    return base_instruction + guidelines


def format_few_shot_examples(
    examples: List[Dict[str, str]],
    max_examples: int = 2
) -> str:
    """
    Format few-shot examples for the prompt.
    
    Args:
        examples: List of dicts with 'schema', 'question', 'sql' keys
        max_examples: Maximum number of examples to include
    
    Returns:
        Formatted examples string
    """
    if not examples or max_examples == 0:
        return ""
    
    examples_text = "\n\n[EXAMPLES]\n\n"
    
    for i, ex in enumerate(examples[:max_examples], 1):
        schema = ex.get("schema", "")
        question = ex.get("question", "")
        sql = ex.get("sql", "")
        
        examples_text += f"""Example {i}:
[SCHEMA]
{schema}

[QUESTION]
{question}

[SQL]
{sql}

"""
    
    examples_text += "[END EXAMPLES]\n\n"
    return examples_text


def build_prompt(
    schema: str,
    question: str,
    style: PromptStyle = PromptStyle.DETAILED,
    system_message: Optional[str] = None,
    instruction: Optional[str] = None,
    schema_links: Optional[str] = None,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    include_guidelines: bool = True,
) -> str:
    """
    Build a complete prompt for NL2SQL.
    
    Args:
        schema: Database schema text
        question: Natural language question
        style: Prompt style
        system_message: Custom system message (overrides style)
        instruction: Custom instruction (overrides style)
        schema_links: Schema linking annotations (optional)
        few_shot_examples: List of example dicts for few-shot learning
        include_guidelines: Whether to include detailed guidelines
    
    Returns:
        Complete prompt string
    """
    # Get system message
    if system_message is None:
        system_message = get_system_message(style)
    
    # Get instruction
    if instruction is None:
        instruction = get_instruction(style, include_guidelines)
    
    # Build user input
    parts = []
    
    # Add instruction
    parts.append(instruction)
    
    # Add few-shot examples if provided
    if few_shot_examples:
        examples_text = format_few_shot_examples(few_shot_examples)
        parts.append(examples_text)
    
    # Add schema
    parts.append("[SCHEMA]")
    parts.append(schema)
    
    # Add schema links if provided
    if schema_links:
        parts.append("")
        parts.append(schema_links)
    
    # Add question
    parts.append("")
    parts.append("[QUESTION]")
    parts.append(question)
    
    user_input = "\n".join(parts)
    
    return user_input


def format_for_training(
    schema: str,
    question: str,
    sql: str,
    style: PromptStyle = PromptStyle.DETAILED,
    schema_links: Optional[str] = None,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, str]:
    """
    Format a training example with improved prompt.
    
    Args:
        schema: Database schema text
        question: Natural language question
        sql: Target SQL query
        style: Prompt style
        schema_links: Schema linking annotations
        few_shot_examples: Few-shot examples (optional)
    
    Returns:
        Dict with 'input' and 'output' keys
    """
    input_text = build_prompt(
        schema=schema,
        question=question,
        style=style,
        schema_links=schema_links,
        few_shot_examples=few_shot_examples,
    )
    
    return {
        "input": input_text,
        "output": f"[SQL]\n{sql}",
        "schema": schema,
        "question": question,
        "sql": sql,
    }


def format_for_inference(
    schema: str,
    question: str,
    style: PromptStyle = PromptStyle.DETAILED,
    schema_links: Optional[str] = None,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Format prompt for inference (without SQL output).
    
    Args:
        schema: Database schema text
        question: Natural language question
        style: Prompt style
        schema_links: Schema linking annotations
        few_shot_examples: Few-shot examples (optional)
    
    Returns:
        Formatted prompt string
    """
    return build_prompt(
        schema=schema,
        question=question,
        style=style,
        schema_links=schema_links,
        few_shot_examples=few_shot_examples,
    )

