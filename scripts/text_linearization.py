"""
Text Linearization for Schema Graphs.
模式图的文本线性化模块

This module converts Hybrid Graph representations into text format
that can be fed to LLMs for NL2SQL tasks.
本模块将混合图表示转换为可供 LLM 使用的文本格式。

Supports multiple linearization styles / 支持多种线性化样式:
- basic: Simple table(columns) format / 简单的表(列)格式
- detailed: Includes types, PK/FK markers / 包含类型、主键/外键标记
- typed: Full type annotations / 完整类型注解
- structured: Section tags for clear boundaries / 带节段标签的清晰结构

Called by / 调用者:
- prepare_training_data.py: Creates training prompts (创建训练提示)
- training_utils.py: Formats schema for model input (格式化模式作为模型输入)
"""

try:
    from .schema_graph import SchemaGraph, ColumnNode, TableNode, EdgeType
except ImportError:
    from schema_graph import SchemaGraph, ColumnNode, TableNode, EdgeType


def linearize_basic(graph: SchemaGraph) -> str:
    """
    Basic text linearization.

    Format:
        Schema Graph:
        Table: Student(id, name, age)
        Table: Course(cid, title, student_id)
        Foreign Key: Student.id -> Course.student_id
    """
    lines = ["Schema Graph:"]

    # Add tables with columns
    for table_name, table in graph.tables.items():
        col_names = [col.name for col in table.columns]
        lines.append(f"Table: {table_name}({', '.join(col_names)})")

    # Add foreign key relationships
    fk_edges = graph.get_foreign_key_edges()
    if fk_edges:
        for edge in fk_edges:
            lines.append(f"Foreign Key: {edge.source} -> {edge.target}")

    return "\n".join(lines)


def linearize_detailed(graph: SchemaGraph) -> str:
    """
    Detailed text linearization with type information.

    Format:
        Schema:
        Table: Student
          Columns: id (number, PK), name (text), age (number)
        Table: Course
          Columns: cid (number, PK), title (text), student_id (number, FK->Student.id)
        Relations:
          Student.id = Course.student_id
    """
    lines = ["Schema:"]

    # Add tables with detailed column info
    for table_name, table in graph.tables.items():
        lines.append(f"Table: {table_name}")

        col_strs = []
        for col in table.columns:
            markers = []
            if col.is_primary_key:
                markers.append("PK")
            if col.is_foreign_key and col.fk_reference:
                markers.append(f"FK->{col.fk_reference[0]}.{col.fk_reference[1]}")

            if markers:
                col_strs.append(f"{col.name} ({col.dtype}, {', '.join(markers)})")
            else:
                col_strs.append(f"{col.name} ({col.dtype})")

        lines.append(f"  Columns: {', '.join(col_strs)}")

    # Add relations section
    fk_edges = graph.get_foreign_key_edges()
    if fk_edges:
        lines.append("Relations:")
        for edge in fk_edges:
            lines.append(f"  {edge.source} = {edge.target}")

    return "\n".join(lines)


def linearize_typed(graph: SchemaGraph) -> str:
    """
    Typed graph linearization (Extension 2 from README).

    Format:
        [table] Student
          [column_primary] id
          [column] name
          [column] age
        [table] Course
          [column_primary] cid
          [column] title
          [column_foreign] student_id
        [foreign_key_edge] Course.student_id -> Student.id
    """
    lines = []

    # Add tables with typed columns
    for table_name, table in graph.tables.items():
        lines.append(f"[table] {table_name}")

        for col in table.columns:
            if col.is_primary_key:
                col_type = "column_primary"
            elif col.is_foreign_key:
                col_type = "column_foreign"
            else:
                col_type = "column"

            lines.append(f"  [{col_type}] {col.name}")

    # Add foreign key edges
    fk_edges = graph.get_foreign_key_edges()
    for edge in fk_edges:
        lines.append(f"[foreign_key_edge] {edge.source} -> {edge.target}")

    return "\n".join(lines)


def linearize_compact(graph: SchemaGraph) -> str:
    """
    Compact single-line linearization for shorter context.

    Format:
        Tables: Student(id*, name, age) | Course(cid*, title, student_id^Student.id)

    * = primary key, ^ref = foreign key reference
    """
    table_strs = []

    for table_name, table in graph.tables.items():
        col_strs = []
        for col in table.columns:
            col_str = col.name
            if col.is_primary_key:
                col_str += "*"
            if col.is_foreign_key and col.fk_reference:
                col_str += f"^{col.fk_reference[0]}.{col.fk_reference[1]}"
            col_strs.append(col_str)

        table_strs.append(f"{table_name}({', '.join(col_strs)})")

    return "Tables: " + " | ".join(table_strs)


def linearize_structured(
    graph: SchemaGraph,
    include_db_id: bool = True,
    semantic_links: list[tuple[str, str, float]] = None
) -> str:
    """
    Structured text linearization with clear section tags and indentation.

    This format is designed for better LLM comprehension with:
    - [SECTION] tags for clear boundaries
    - Double newlines between sections
    - 4-space indentation for hierarchical structure (columns under tables)
    - (PK) and (FK) markers inline with columns

    Format:
        [DATABASE]
        farm

        [TABLES]
        city:
            City_ID (PK)
            Official_Name
            Status
        farm:
            Farm_ID (PK)
            Host_city_ID (FK)

        [FOREIGN KEYS]
        farm.Host_city_ID -> city.City_ID

        [SEMANTIC LINKS]
        farm.Year ≈ farm_competition.Year
    """
    sections = []

    # [DATABASE] section
    if include_db_id and graph.db_id:
        sections.append(f"[DATABASE]\n{graph.db_id}")

    # [TABLES] section
    table_lines = ["[TABLES]"]
    for table_name, table in graph.tables.items():
        table_lines.append(f"{table_name}:")
        for col in table.columns:
            # Build column string with markers
            col_str = f"    {col.name}"
            markers = []
            if col.is_primary_key:
                markers.append("PK")
            if col.is_foreign_key:
                markers.append("FK")
            if markers:
                col_str += f" ({', '.join(markers)})"
            table_lines.append(col_str)
    sections.append("\n".join(table_lines))

    # [FOREIGN KEYS] section
    fk_edges = graph.get_foreign_key_edges()
    if fk_edges:
        fk_lines = ["[FOREIGN KEYS]"]
        for edge in fk_edges:
            fk_lines.append(f"{edge.source} -> {edge.target}")
        sections.append("\n".join(fk_lines))

    # [SEMANTIC LINKS] section
    if semantic_links:
        sem_lines = ["[SEMANTIC LINKS]"]
        for col1, col2, score in semantic_links:
            sem_lines.append(f"{col1} ≈ {col2}")
        sections.append("\n".join(sem_lines))

    # Join sections with double newlines
    return "\n\n".join(sections)


def linearize_wikisql_structured(graph: SchemaGraph, table_data: dict = None) -> str:
    """
    WikiSQL-specific structured linearization.

    Since WikiSQL has single tables with no FKs, we use a simplified format.

    Format:
        [TABLES]
        table:
            Player
            No.
            Nationality
            Position
    """
    if table_data and "header" in table_data:
        headers = table_data["header"]
        lines = ["[TABLES]", "table:"]
        for header in headers:
            lines.append(f"    {header}")
        return "\n".join(lines)

    # Fallback to graph-based
    if "table" in graph.tables:
        lines = ["[TABLES]", "table:"]
        for col in graph.tables["table"].columns:
            lines.append(f"    {col.name}")
        return "\n".join(lines)

    return "[TABLES]\ntable:\n    (unknown)"


def linearize_wikisql(graph: SchemaGraph, table_data: dict = None) -> str:
    """
    WikiSQL-specific linearization that includes column headers.

    Since WikiSQL has single tables, we use a simpler format.

    Format:
        Table columns: Player, No., Nationality, Position, Years in Toronto
    """
    if table_data and "header" in table_data:
        headers = table_data["header"]
        return f"Table columns: {', '.join(headers)}"

    # Fallback to graph-based
    if "table" in graph.tables:
        col_names = [col.name for col in graph.tables["table"].columns]
        return f"Table columns: {', '.join(col_names)}"

    return "Table columns: (unknown)"


def linearize_for_training(
    graph: SchemaGraph,
    style: str = "structured",
    include_db_id: bool = True,
    include_semantic_links: bool = False,
    semantic_threshold: float = 0.7
) -> str:
    """
    Main linearization function for training data preparation.

    Args:
        graph: SchemaGraph to linearize
        style: Linearization style - "basic", "detailed", "typed", "compact", "structured"
        include_db_id: Whether to include database ID
        include_semantic_links: Whether to compute semantic links between columns
        semantic_threshold: Cosine similarity threshold for semantic links

    Returns:
        Linearized schema text
    """
    # Handle structured style separately (it has different signature)
    if style == "structured":
        semantic_links = None
        if include_semantic_links and len(graph.tables) > 1:
            semantic_links = compute_semantic_links(graph, threshold=semantic_threshold)
        return linearize_structured(
            graph,
            include_db_id=include_db_id,
            semantic_links=semantic_links
        )

    # Legacy linearizers
    linearizers = {
        "basic": linearize_basic,
        "detailed": linearize_detailed,
        "typed": linearize_typed,
        "compact": linearize_compact,
    }

    if style not in linearizers:
        raise ValueError(f"Unknown style: {style}. Choose from {list(linearizers.keys()) + ['structured']}")

    schema_text = linearizers[style](graph)

    # Add semantic links if requested
    if include_semantic_links and len(graph.tables) > 1:
        semantic_links = compute_semantic_links(graph, threshold=semantic_threshold)
        for col1, col2, score in semantic_links:
            schema_text += f"\nSemantic Link: {col1} ≈ {col2}"

    if include_db_id:
        return f"Database: {graph.db_id}\n{schema_text}"

    return schema_text


def compute_semantic_links(
    graph: SchemaGraph,
    threshold: float = 0.7
) -> list[tuple[str, str, float]]:
    """
    Compute semantic links between columns using embedding similarity.

    Returns list of (table.column1, table.column2, similarity_score) tuples
    for columns that are semantically similar but not FK-linked.

    Uses sophisticated rules for composite word comparison:
    - Filters out weak semantic words (id, key, code, etc.)
    - For composite vs single: ANY part match → similar
    - For composite vs composite: BOTH parts must match
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("Warning: sentence-transformers not installed, skipping semantic links")
        return []

    # Weak semantic words that don't carry meaning on their own
    WEAK_WORDS = {
        'id', 'key', 'code', 'number', 'uuid', 'guid', 'no', 'idx', 'index',
        'pk', 'fk', 'ref', 'num', 'seq', 'type', 'flag', 'status'
    }

    def is_composite(word: str) -> bool:
        """Check if word contains underscore or space."""
        return '_' in word or ' ' in word

    def split_word(word: str) -> list[str]:
        """Split composite word into parts."""
        return word.replace(' ', '_').lower().split('_')

    def get_meaningful_parts(word: str) -> list[str]:
        """Get meaningful parts, filtering out weak semantic words."""
        parts = split_word(word)
        meaningful = [p for p in parts if p and p not in WEAK_WORDS]
        # If all parts are weak, return original parts
        return meaningful if meaningful else parts

    # Collect all columns
    columns = []
    for table_name, table in graph.tables.items():
        for col in table.columns:
            columns.append((table_name, col.name))

    if len(columns) < 2:
        return []

    # Get existing FK pairs to exclude
    fk_pairs = set()
    for edge in graph.edges:
        if edge.edge_type == EdgeType.FOREIGN_KEY:
            fk_pairs.add((edge.source, edge.target))
            fk_pairs.add((edge.target, edge.source))

    # Build vocabulary of all unique meaningful parts for embedding
    all_parts = set()
    for _, col_name in columns:
        all_parts.update(get_meaningful_parts(col_name))

    # Compute embeddings for all unique parts
    model = SentenceTransformer('all-MiniLM-L6-v2')
    part_list = list(all_parts)
    if not part_list:
        return []

    embeddings = model.encode(part_list, convert_to_numpy=True)
    part_to_embedding = {part: embeddings[i] for i, part in enumerate(part_list)}

    def get_similarity(part1: str, part2: str) -> float:
        """Compute cosine similarity between two parts."""
        if part1 not in part_to_embedding or part2 not in part_to_embedding:
            return 0.0
        e1, e2 = part_to_embedding[part1], part_to_embedding[part2]
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

    def are_semantically_similar(col1: str, col2: str) -> tuple[bool, float]:
        """
        Check if two column names are semantically similar.

        Rules:
        - Single vs Single: direct comparison
        - Composite vs Single: ANY part of composite matches single → similar
        - Composite vs Composite: BOTH first AND second parts must match
        """
        parts1 = get_meaningful_parts(col1)
        parts2 = get_meaningful_parts(col2)

        # Case 1: Both single (or reduced to single after filtering)
        if len(parts1) == 1 and len(parts2) == 1:
            sim = get_similarity(parts1[0], parts2[0])
            return sim >= threshold, sim

        # Case 2: One composite, one single
        if len(parts1) > 1 and len(parts2) == 1:
            # ANY part of col1 matches col2
            best_sim = max(get_similarity(p, parts2[0]) for p in parts1)
            return best_sim >= threshold, best_sim

        if len(parts1) == 1 and len(parts2) > 1:
            # ANY part of col2 matches col1
            best_sim = max(get_similarity(parts1[0], p) for p in parts2)
            return best_sim >= threshold, best_sim

        # Case 3: Both composite - BOTH parts must match
        if len(parts1) >= 2 and len(parts2) >= 2:
            first_sim = get_similarity(parts1[0], parts2[0])
            second_sim = get_similarity(parts1[-1], parts2[-1])
            avg_sim = (first_sim + second_sim) / 2
            return (first_sim >= threshold and second_sim >= threshold), avg_sim

        return False, 0.0

    # Find similar pairs
    semantic_links = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            table1, col1 = columns[i]
            table2, col2 = columns[j]

            # Skip same-table pairs
            if table1 == table2:
                continue

            # Skip FK-linked pairs
            col1_full = f"{table1}.{col1}"
            col2_full = f"{table2}.{col2}"
            if (col1_full, col2_full) in fk_pairs:
                continue

            # Check semantic similarity with new rules
            is_similar, score = are_semantically_similar(col1, col2)
            if is_similar:
                semantic_links.append((col1_full, col2_full, score))

    # Sort by similarity and return top links
    semantic_links.sort(key=lambda x: -x[2])
    return semantic_links[:5]  # Limit to top 5


def format_training_example(
    question: str,
    schema_text: str,
    sql: str = None,
    include_instruction: bool = True,
    use_structured_format: bool = True,
    schema_graph = None,
    include_schema_linking: bool = False,
    prompt_style: str = "detailed",  # "simple", "detailed", "expert", "few_shot"
) -> dict:
    """
    Format a complete training example with schema, question, and SQL.

    Returns a dict with 'input' and 'output' suitable for instruction tuning.

    Args:
        question: Natural language question
        schema_text: Linearized schema text
        sql: Target SQL query
        include_instruction: Whether to include instruction prefix
        use_structured_format: Use [QUESTION] and [SQL] section tags (recommended)
        schema_graph: SchemaGraph object (required if include_schema_linking=True)
        include_schema_linking: Whether to add schema linking annotations
        prompt_style: Prompt style - "simple", "detailed", "expert", or "few_shot"
    """
    # Use new prompt templates if style is not "simple"
    if prompt_style != "simple" and use_structured_format:
        try:
            from .prompt_templates import PromptStyle, format_for_training
        except ImportError:
            try:
                from prompt_templates import PromptStyle, format_for_training
            except ImportError:
                # Fall back to simple format if prompt_templates not available
                prompt_style = "simple"
        
        if prompt_style != "simple":
            # Perform schema linking if requested
            schema_links_text = None
            if include_schema_linking and schema_graph is not None:
                try:
                    from .schema_linking import perform_schema_linking, format_schema_links_for_prompt
                except ImportError:
                    from schema_linking import perform_schema_linking, format_schema_links_for_prompt
                
                links = perform_schema_linking(question, schema_graph)
                schema_links_text = format_schema_links_for_prompt(links)
            
            # Map string to enum
            style_map = {
                "simple": PromptStyle.SIMPLE,
                "detailed": PromptStyle.DETAILED,
                "expert": PromptStyle.EXPERT,
                "few_shot": PromptStyle.FEW_SHOT,
            }
            style = style_map.get(prompt_style, PromptStyle.DETAILED)
            
            result = format_for_training(
                schema=schema_text,
                question=question,
                sql=sql or "",
                style=style,
                schema_links=schema_links_text,
            )
            return result
    
    # Fall back to original simple format
    # Perform schema linking if requested
    schema_links_text = ""
    if include_schema_linking and schema_graph is not None:
        try:
            from .schema_linking import perform_schema_linking, format_schema_links_for_prompt
        except ImportError:
            from schema_linking import perform_schema_linking, format_schema_links_for_prompt
        
        links = perform_schema_linking(question, schema_graph)
        schema_links_text = format_schema_links_for_prompt(links)
        if schema_links_text:
            schema_links_text = "\n\n" + schema_links_text
    
    if use_structured_format:
        # New structured format with section tags
        if include_instruction:
            instruction = (
                "Given the following database schema and question, "
                "generate the SQL query that answers the question."
            )
            input_text = f"{instruction}\n\n{schema_text}{schema_links_text}\n\n[QUESTION]\n{question}"
        else:
            input_text = f"{schema_text}{schema_links_text}\n\n[QUESTION]\n{question}"

        result = {
            "input": input_text,
            "schema": schema_text,
            "question": question,
        }

        if sql:
            result["output"] = f"[SQL]\n{sql}"
            result["sql"] = sql
    else:
        # Legacy format (for backward compatibility)
        if include_instruction:
            instruction = (
                "Given the following database schema and question, "
                "generate the SQL query that answers the question."
            )
            input_text = f"{instruction}\n\n{schema_text}\n\nQuestion: {question}"
        else:
            input_text = f"{schema_text}\n\nQuestion: {question}"

        result = {
            "input": input_text,
            "schema": schema_text,
            "question": question,
        }

        if sql:
            result["output"] = sql
            result["sql"] = sql

    return result


# Example usage
if __name__ == "__main__":
    from schema_graph import parse_spider_schema

    # Test schema
    test_schema = {
        "db_id": "university",
        "table_names_original": ["Student", "Course", "Enrollment"],
        "column_names_original": [
            [-1, "*"],
            [0, "student_id"],
            [0, "name"],
            [0, "age"],
            [1, "course_id"],
            [1, "title"],
            [1, "credits"],
            [2, "enrollment_id"],
            [2, "student_id"],
            [2, "course_id"],
            [2, "grade"]
        ],
        "column_types": ["text", "number", "text", "number", "number", "text", "number", "number", "number", "number", "text"],
        "primary_keys": [1, 4, 7],
        "foreign_keys": [[8, 1], [9, 4]]
    }

    graph = parse_spider_schema(test_schema)

    print("=" * 60)
    print("STRUCTURED LINEARIZATION (NEW - RECOMMENDED)")
    print("=" * 60)
    print(linearize_structured(graph, include_db_id=True))

    print("\n" + "=" * 60)
    print("BASIC LINEARIZATION (LEGACY)")
    print("=" * 60)
    print(linearize_basic(graph))

    print("\n" + "=" * 60)
    print("DETAILED LINEARIZATION (LEGACY)")
    print("=" * 60)
    print(linearize_detailed(graph))

    print("\n" + "=" * 60)
    print("TYPED LINEARIZATION (LEGACY)")
    print("=" * 60)
    print(linearize_typed(graph))

    print("\n" + "=" * 60)
    print("COMPACT LINEARIZATION (LEGACY)")
    print("=" * 60)
    print(linearize_compact(graph))

    print("\n" + "=" * 60)
    print("FULL TRAINING EXAMPLE (STRUCTURED FORMAT)")
    print("=" * 60)
    schema_text = linearize_for_training(graph, style="structured")
    example = format_training_example(
        question="What are the names of students enrolled in the course with title 'Database Systems'?",
        schema_text=schema_text,
        sql="SELECT Student.name FROM Student JOIN Enrollment ON Student.student_id = Enrollment.student_id JOIN Course ON Enrollment.course_id = Course.course_id WHERE Course.title = 'Database Systems'"
    )
    print(f"INPUT:\n{example['input']}\n")
    print(f"OUTPUT:\n{example['output']}")
