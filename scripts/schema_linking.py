"""
Schema Linking: Map natural language question words to database schema elements.

Schema linking identifies:
1. Table mentions in the question
2. Column mentions in the question  
3. Value-to-column mappings (e.g., "25 years old" → age column)
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class SchemaLink:
    """Represents a link between question text and schema element."""
    question_text: str  # The text in the question
    schema_element: str  # table.column or just table
    link_type: str  # "table", "column", "value"
    confidence: float  # 0.0 to 1.0
    span: Tuple[int, int]  # Character span in question


def extract_table_mentions(
    question: str,
    graph,
    threshold: float = 0.6
) -> List[SchemaLink]:
    """
    Identify table names mentioned in the question.
    
    Uses fuzzy matching and embedding similarity to find table mentions.
    """
    links = []
    question_lower = question.lower()
    
    # Get all table names
    table_names = list(graph.tables.keys())
    if not table_names:
        return links
    
    # Direct string matching (exact or substring)
    for table_name in table_names:
        table_lower = table_name.lower()
        
        # Exact match
        if table_lower in question_lower:
            start = question_lower.find(table_lower)
            links.append(SchemaLink(
                question_text=question[start:start+len(table_name)],
                schema_element=table_name,
                link_type="table",
                confidence=1.0,
                span=(start, start + len(table_name))
            ))
        # Substring match (e.g., "singers" contains "singer")
        elif any(word in table_lower for word in question_lower.split() if len(word) > 3):
            # Find matching word
            for word in question_lower.split():
                if len(word) > 3 and word in table_lower or table_lower in word:
                    start = question_lower.find(word)
                    links.append(SchemaLink(
                        question_text=word,
                        schema_element=table_name,
                        link_type="table",
                        confidence=0.8,
                        span=(start, start + len(word))
                    ))
    
    # Embedding-based matching if available
    if HAS_SENTENCE_TRANSFORMERS and len(table_names) > 0:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get embeddings for question words and table names
            question_words = [w for w in question_lower.split() if len(w) > 3]
            if not question_words:
                return links
            
            question_embeddings = model.encode(question_words, convert_to_numpy=True)
            table_embeddings = model.encode([t.lower() for t in table_names], convert_to_numpy=True)
            
            # Compute similarities
            similarities = np.dot(question_embeddings, table_embeddings.T)
            
            for i, word in enumerate(question_words):
                for j, table_name in enumerate(table_names):
                    sim = float(similarities[i, j])
                    if sim >= threshold:
                        # Check if not already added
                        if not any(l.schema_element == table_name and 
                                  l.question_text == word for l in links):
                            start = question_lower.find(word)
                            links.append(SchemaLink(
                                question_text=word,
                                schema_element=table_name,
                                link_type="table",
                                confidence=sim,
                                span=(start, start + len(word))
                            ))
        except Exception:
            pass  # Fall back to string matching only
    
    return links


def extract_column_mentions(
    question: str,
    graph,
    threshold: float = 0.6
) -> List[SchemaLink]:
    """
    Identify column names mentioned in the question.
    
    Matches question words to column names using fuzzy matching and embeddings.
    """
    links = []
    question_lower = question.lower()
    
    # Collect all columns
    columns = []
    for table_name, table in graph.tables.items():
        for col in table.columns:
            columns.append((table_name, col.name))
    
    if not columns:
        return links
    
    # Direct string matching
    for table_name, col_name in columns:
        col_lower = col_name.lower()
        
        # Exact match
        if col_lower in question_lower:
            start = question_lower.find(col_lower)
            links.append(SchemaLink(
                question_text=question[start:start+len(col_name)],
                schema_element=f"{table_name}.{col_name}",
                link_type="column",
                confidence=1.0,
                span=(start, start + len(col_name))
            ))
        # Word-level matching (e.g., "age" matches "Age")
        else:
            col_words = col_lower.replace('_', ' ').split()
            for col_word in col_words:
                if len(col_word) > 2 and col_word in question_lower:
                    start = question_lower.find(col_word)
                    links.append(SchemaLink(
                        question_text=col_word,
                        schema_element=f"{table_name}.{col_name}",
                        link_type="column",
                        confidence=0.7,
                        span=(start, start + len(col_word))
                    ))
    
    # Embedding-based matching
    if HAS_SENTENCE_TRANSFORMERS and len(columns) > 0:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            question_words = [w for w in question_lower.split() if len(w) > 2]
            if not question_words:
                return links
            
            # Get column name variations (full name and individual words)
            col_texts = []
            col_info = []
            for table_name, col_name in columns:
                col_lower = col_name.lower()
                col_texts.append(col_lower)
                col_info.append((table_name, col_name, col_lower))
                # Also add individual words
                for word in col_lower.replace('_', ' ').split():
                    if len(word) > 2:
                        col_texts.append(word)
                        col_info.append((table_name, col_name, word))
            
            question_embeddings = model.encode(question_words, convert_to_numpy=True)
            col_embeddings = model.encode(col_texts, convert_to_numpy=True)
            
            similarities = np.dot(question_embeddings, col_embeddings.T)
            
            for i, word in enumerate(question_words):
                for j, (table_name, col_name, col_text) in enumerate(col_info):
                    sim = float(similarities[i, j])
                    if sim >= threshold:
                        # Check if not already added
                        if not any(l.schema_element == f"{table_name}.{col_name}" and 
                                  l.question_text == word for l in links):
                            start = question_lower.find(word)
                            links.append(SchemaLink(
                                question_text=word,
                                schema_element=f"{table_name}.{col_name}",
                                link_type="column",
                                confidence=sim,
                                span=(start, start + len(word))
                            ))
        except Exception:
            pass
    
    return links


def extract_value_mentions(
    question: str,
    graph,
    threshold: float = 0.5
) -> List[SchemaLink]:
    """
    Identify values in the question and map them to likely columns.
    
    Examples:
    - "25 years old" → age column
    - "John" → name column
    - "2020" → year/date column
    """
    links = []
    question_lower = question.lower()
    
    # Extract potential values (numbers, quoted strings, capitalized words)
    # Numbers
    numbers = re.findall(r'\b\d+\b', question)
    # Quoted strings
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', question)
    quoted = [q[0] or q[1] for q in quoted]
    # Capitalized words (likely proper nouns)
    capitalized = re.findall(r'\b[A-Z][a-z]+\b', question)
    
    all_values = numbers + quoted + capitalized
    
    # Get all columns with their types
    columns = []
    for table_name, table in graph.tables.items():
        for col in table.columns:
            columns.append((table_name, col.name, col.dtype))
    
    if not columns or not all_values:
        return links
    
    # Match values to columns based on type and name
    for value in all_values:
        value_lower = value.lower()
        
        # Try to match to columns
        for table_name, col_name, dtype in columns:
            col_lower = col_name.lower()
            confidence = 0.0
            
            # Type-based matching
            if dtype == "number" and value.isdigit():
                confidence = 0.6
            elif dtype == "text" and not value.isdigit():
                confidence = 0.5
            
            # Name-based matching (e.g., "age" column for "25 years old")
            if "age" in col_lower and ("year" in value_lower or "old" in value_lower):
                confidence = 0.9
            elif "name" in col_lower and value[0].isupper():
                confidence = 0.8
            elif "year" in col_lower and value.isdigit() and len(value) == 4:
                confidence = 0.8
            elif "date" in col_lower and ("/" in value or "-" in value):
                confidence = 0.7
            
            if confidence >= threshold:
                start = question_lower.find(value_lower)
                if start == -1:
                    start = question.find(value)
                links.append(SchemaLink(
                    question_text=value,
                    schema_element=f"{table_name}.{col_name}",
                    link_type="value",
                    confidence=confidence,
                    span=(start, start + len(value)) if start >= 0 else (0, len(value))
                ))
    
    return links


def perform_schema_linking(
    question: str,
    graph,
    include_tables: bool = True,
    include_columns: bool = True,
    include_values: bool = True,
    table_threshold: float = 0.6,
    column_threshold: float = 0.6,
    value_threshold: float = 0.5,
) -> List[SchemaLink]:
    """
    Perform complete schema linking for a question.
    
    Identifies all mentions of tables, columns, and values in the question
    and links them to schema elements.
    
    Args:
        question: Natural language question
        graph: SchemaGraph object
        include_tables: Whether to extract table mentions
        include_columns: Whether to extract column mentions
        include_values: Whether to extract value-to-column mappings
        table_threshold: Similarity threshold for table matching
        column_threshold: Similarity threshold for column matching
        value_threshold: Confidence threshold for value matching
    
    Returns:
        List of SchemaLink objects
    """
    all_links = []
    
    if include_tables:
        table_links = extract_table_mentions(question, graph, table_threshold)
        all_links.extend(table_links)
    
    if include_columns:
        column_links = extract_column_mentions(question, graph, column_threshold)
        all_links.extend(column_links)
    
    if include_values:
        value_links = extract_value_mentions(question, graph, value_threshold)
        all_links.extend(value_links)
    
    # Remove duplicates (same span and schema element)
    seen = set()
    unique_links = []
    for link in all_links:
        key = (link.span, link.schema_element, link.link_type)
        if key not in seen:
            seen.add(key)
            unique_links.append(link)
    
    # Sort by confidence (highest first)
    unique_links.sort(key=lambda x: x.confidence, reverse=True)
    
    return unique_links


def format_schema_links_for_prompt(links: List[SchemaLink]) -> str:
    """
    Format schema links as text to include in the prompt.
    
    Example output:
    [SCHEMA LINKS]
    Question word "singer" → table: singer
    Question word "name" → column: singer.name
    Question word "25" → column: singer.age (value)
    """
    if not links:
        return ""
    
    lines = ["[SCHEMA LINKS]"]
    
    # Group by type
    tables = [l for l in links if l.link_type == "table"]
    columns = [l for l in links if l.link_type == "column"]
    values = [l for l in links if l.link_type == "value"]
    
    for link in tables:
        lines.append(f'Question word "{link.question_text}" → table: {link.schema_element} (confidence: {link.confidence:.2f})')
    
    for link in columns:
        lines.append(f'Question word "{link.question_text}" → column: {link.schema_element} (confidence: {link.confidence:.2f})')
    
    for link in values:
        lines.append(f'Question word "{link.question_text}" → column: {link.schema_element} (value, confidence: {link.confidence:.2f})')
    
    return "\n".join(lines)

