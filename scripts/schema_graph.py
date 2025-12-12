"""
Hybrid Graph Schema Representation for NL2SQL.
NL2SQL 混合图模式表示模块

This module implements the Table&Column-level Hybrid Graph design:
本模块实现表-列级别的混合图设计：

- Nodes: Tables and Columns
  节点：表和列
- Edges:
  边：
  - table-column containment edges (表-列包含边)
  - foreign key connections between columns (列之间的外键连接)
  - table-table edges for cross-table relationships (跨表关系边)

Called by / 调用者:
- text_linearization.py: Converts graph to text format (将图转换为文本格式)
- prepare_training_data.py: Creates training data (创建训练数据)
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class NodeType(Enum):
    TABLE = "table"
    COLUMN = "column"


class EdgeType(Enum):
    TABLE_COLUMN = "table_column"      # Table contains column
    FOREIGN_KEY = "foreign_key"        # FK relationship between columns
    TABLE_TABLE = "table_table"        # Cross-table relationship (derived from FK)
    INTRA_TABLE = "intra_table"        # Columns within same table


@dataclass
class ColumnNode:
    """Represents a column in the schema graph."""
    name: str
    table_name: str
    dtype: str = "text"
    is_primary_key: bool = False
    is_foreign_key: bool = False
    # Reference to another column if this is a FK
    fk_reference: Optional[tuple[str, str]] = None  # (table, column)

    @property
    def full_name(self) -> str:
        return f"{self.table_name}.{self.name}"

    def __hash__(self):
        return hash(self.full_name)


@dataclass
class TableNode:
    """Represents a table in the schema graph."""
    name: str
    columns: list[ColumnNode] = field(default_factory=list)

    def add_column(self, column: ColumnNode) -> None:
        self.columns.append(column)

    def get_primary_keys(self) -> list[ColumnNode]:
        return [c for c in self.columns if c.is_primary_key]

    def get_foreign_keys(self) -> list[ColumnNode]:
        return [c for c in self.columns if c.is_foreign_key]

    def __hash__(self):
        return hash(self.name)


@dataclass
class Edge:
    """Represents an edge in the schema graph."""
    source: str  # Node identifier (table name or table.column)
    target: str  # Node identifier
    edge_type: EdgeType

    def __hash__(self):
        return hash((self.source, self.target, self.edge_type))


@dataclass
class SchemaGraph:
    """
    Hybrid Graph representation of a database schema.

    Combines table-level and column-level representations with:
    - Table nodes and column nodes
    - Table-column containment edges
    - Foreign key edges between columns
    - Table-table edges for cross-table relationships
    """
    db_id: str
    tables: dict[str, TableNode] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_table(self, table: TableNode) -> None:
        """Add a table and create table-column edges."""
        self.tables[table.name] = table

        # Create table-column containment edges
        for column in table.columns:
            self.edges.append(Edge(
                source=table.name,
                target=column.full_name,
                edge_type=EdgeType.TABLE_COLUMN
            ))

    def add_foreign_key(
        self,
        from_table: str,
        from_column: str,
        to_table: str,
        to_column: str
    ) -> None:
        """Add a foreign key relationship and update nodes."""
        # Update column node to mark as FK
        if from_table in self.tables:
            for col in self.tables[from_table].columns:
                if col.name == from_column:
                    col.is_foreign_key = True
                    col.fk_reference = (to_table, to_column)
                    break

        # Add column-column FK edge
        self.edges.append(Edge(
            source=f"{from_table}.{from_column}",
            target=f"{to_table}.{to_column}",
            edge_type=EdgeType.FOREIGN_KEY
        ))

        # Add table-table edge
        self.edges.append(Edge(
            source=from_table,
            target=to_table,
            edge_type=EdgeType.TABLE_TABLE
        ))

    def add_primary_key(self, table_name: str, column_name: str) -> None:
        """Mark a column as primary key."""
        if table_name in self.tables:
            for col in self.tables[table_name].columns:
                if col.name == column_name:
                    col.is_primary_key = True
                    break

    def get_all_columns(self) -> list[ColumnNode]:
        """Get all columns across all tables."""
        columns = []
        for table in self.tables.values():
            columns.extend(table.columns)
        return columns

    def get_foreign_key_edges(self) -> list[Edge]:
        """Get all foreign key edges."""
        return [e for e in self.edges if e.edge_type == EdgeType.FOREIGN_KEY]

    def get_table_relationships(self) -> list[tuple[str, str]]:
        """Get all table-to-table relationships."""
        return [
            (e.source, e.target)
            for e in self.edges
            if e.edge_type == EdgeType.TABLE_TABLE
        ]

    def __repr__(self) -> str:
        return (
            f"SchemaGraph(db_id='{self.db_id}', "
            f"tables={len(self.tables)}, edges={len(self.edges)})"
        )


def parse_wikisql_schema(table_data: dict) -> SchemaGraph:
    """
    Parse WikiSQL table format into SchemaGraph.
    将 WikiSQL 表格式解析为 SchemaGraph。
    
    Function / 功能:
        Converts WikiSQL raw table data to our graph representation.
        将 WikiSQL 原始表数据转换为图表示。
    
    Called by / 调用者:
        - prepare_training_data.py: When processing WikiSQL dataset
          处理 WikiSQL 数据集时调用
    
    Args / 参数:
        table_data: WikiSQL table dict with keys: id, header, types, rows
                    WikiSQL 表字典，包含 id, header, types, rows 等键
    
    Returns / 返回:
        SchemaGraph: Graph representation of the table schema
                     表模式的图表示
    
    Note: WikiSQL has single-table queries, no foreign keys.
    注意：WikiSQL 只有单表查询，没有外键。
    """
    table_id = table_data.get("id", "table")
    headers = table_data.get("header", [])
    types = table_data.get("types", ["text"] * len(headers))

    # Use page_title as table name if available, otherwise "table"
    page_title = table_data.get("page_title", "")
    if page_title:
        # Clean up page title to be a valid table name
        table_name = page_title.replace(" ", "_").replace("-", "_")
        # Remove special characters
        table_name = "".join(c for c in table_name if c.isalnum() or c == "_")
    else:
        table_name = "table"

    graph = SchemaGraph(db_id=table_id)

    # Create single table with columns
    table = TableNode(name=table_name)

    for i, (header, dtype) in enumerate(zip(headers, types)):
        column = ColumnNode(
            name=header,
            table_name=table_name,
            dtype=dtype,
            is_primary_key=(i == 0)  # Assume first column is PK for WikiSQL
        )
        table.add_column(column)

    graph.add_table(table)

    return graph


def parse_spider_schema(schema_data: dict) -> SchemaGraph:
    """
    Parse Spider schema format into SchemaGraph.
    将 Spider 模式格式解析为 SchemaGraph。
    
    Function / 功能:
        Converts Spider database schema (with multiple tables and foreign keys) 
        to our hybrid graph representation.
        将 Spider 数据库模式（包含多表和外键）转换为混合图表示。
    
    Called by / 调用者:
        - prepare_training_data.py: When processing Spider dataset
          处理 Spider 数据集时调用
    
    Args / 参数:
        schema_data: Spider schema dict from tables.json
                     来自 tables.json 的 Spider 模式字典
    
    Returns / 返回:
        SchemaGraph: Graph with tables, columns, and FK relationships
                     包含表、列和外键关系的图
    
    Spider schema format (from tables.json):
    {
        "db_id": "concert_singer",
        "table_names_original": ["stadium", "singer", ...],
        "column_names_original": [[-1, "*"], [0, "Stadium_ID"], ...],
        "primary_keys": [1, 5, 11, 14],
        "foreign_keys": [[12, 1], [13, 5], ...]
    }
    """
    db_id = schema_data.get("db_id", "unknown")
    table_names = schema_data.get("table_names_original", [])
    column_info = schema_data.get("column_names_original", [])
    column_types = schema_data.get("column_types", [])
    primary_keys = schema_data.get("primary_keys", [])
    foreign_keys = schema_data.get("foreign_keys", [])

    graph = SchemaGraph(db_id=db_id)

    # Create tables
    tables_dict: dict[int, TableNode] = {}
    for idx, table_name in enumerate(table_names):
        tables_dict[idx] = TableNode(name=table_name)

    # Add columns to tables
    # column_info format: [table_idx, column_name]
    # table_idx = -1 means it's the special "*" column, skip it
    column_to_table: dict[int, tuple[str, str]] = {}  # col_idx -> (table_name, col_name)

    for col_idx, (table_idx, col_name) in enumerate(column_info):
        if table_idx == -1:  # Skip "*" column
            continue

        table_name = table_names[table_idx]
        dtype = column_types[col_idx] if col_idx < len(column_types) else "text"

        column = ColumnNode(
            name=col_name,
            table_name=table_name,
            dtype=dtype,
            is_primary_key=(col_idx in primary_keys)
        )
        tables_dict[table_idx].add_column(column)
        column_to_table[col_idx] = (table_name, col_name)

    # Add tables to graph
    for table in tables_dict.values():
        graph.add_table(table)

    # Add foreign key relationships
    for from_col_idx, to_col_idx in foreign_keys:
        if from_col_idx in column_to_table and to_col_idx in column_to_table:
            from_table, from_col = column_to_table[from_col_idx]
            to_table, to_col = column_to_table[to_col_idx]
            graph.add_foreign_key(from_table, from_col, to_table, to_col)

    return graph


def parse_create_table_schema(create_statements: str, db_id: str = "database") -> SchemaGraph:
    """
    Parse CREATE TABLE statements into SchemaGraph.

    Handles format like:
    CREATE TABLE department (creation VARCHAR, department_id VARCHAR);
    CREATE TABLE management (department_id VARCHAR, head_id VARCHAR);

    Args:
        create_statements: String containing one or more CREATE TABLE statements
        db_id: Database identifier

    Returns:
        SchemaGraph instance
    """
    import re

    graph = SchemaGraph(db_id=db_id)

    # Pattern to match CREATE TABLE statements
    table_pattern = r'CREATE\s+TABLE\s+(\w+)\s*\(([^)]+)\)'

    # Find all CREATE TABLE statements
    matches = re.findall(table_pattern, create_statements, re.IGNORECASE)

    # Track columns that might be foreign keys (contain _id suffix)
    potential_fk_columns: list[tuple[str, str]] = []  # (table_name, column_name)
    primary_key_candidates: dict[str, str] = {}  # table_name -> pk_column

    for table_name, columns_str in matches:
        table = TableNode(name=table_name)

        # Parse column definitions
        columns = [c.strip() for c in columns_str.split(',')]

        for i, col_def in enumerate(columns):
            # Extract column name and type
            parts = col_def.strip().split()
            if not parts:
                continue

            col_name = parts[0]
            col_type = parts[1] if len(parts) > 1 else "text"

            # Check for PRIMARY KEY keyword
            is_pk = 'PRIMARY' in col_def.upper() and 'KEY' in col_def.upper()

            # Normalize type names
            col_type_lower = col_type.lower()
            if col_type_lower in ('varchar', 'char', 'text', 'string'):
                normalized_type = "text"
            elif col_type_lower in ('int', 'integer', 'bigint', 'smallint', 'number'):
                normalized_type = "number"
            elif col_type_lower in ('real', 'float', 'double', 'decimal'):
                normalized_type = "number"
            else:
                normalized_type = col_type_lower

            column = ColumnNode(
                name=col_name,
                table_name=table_name,
                dtype=normalized_type,
                is_primary_key=is_pk
            )
            table.add_column(column)

            # Track potential FK columns (those ending with _id)
            if col_name.lower().endswith('_id'):
                potential_fk_columns.append((table_name, col_name))

            # First column or column ending with _id could be PK
            if i == 0 and not is_pk:
                primary_key_candidates[table_name] = col_name

        graph.add_table(table)

    # Infer foreign keys based on column name matching
    # If table A has column "user_id" and table "user" exists with "id" column,
    # then A.user_id -> user.id is a foreign key
    for from_table, fk_col in potential_fk_columns:
        # Try to find referenced table
        # e.g., department_id -> department, head_id -> head
        base_name = fk_col.lower()
        if base_name.endswith('_id'):
            referenced_table_name = base_name[:-3]  # Remove _id

            # Check if such a table exists
            for table_name in graph.tables:
                if table_name.lower() == referenced_table_name:
                    # Find the primary key or id column in referenced table
                    ref_table = graph.tables[table_name]
                    ref_col = None

                    # Look for PK or 'id' column
                    for col in ref_table.columns:
                        if col.is_primary_key:
                            ref_col = col.name
                            break
                        if col.name.lower() == 'id' or col.name.lower() == fk_col.lower():
                            ref_col = col.name

                    if ref_col and from_table != table_name:
                        graph.add_foreign_key(from_table, fk_col, table_name, ref_col)
                    break

    return graph


def create_schema_graph(schema_data: dict, dataset_type: str = "spider") -> SchemaGraph:
    """
    Factory function to create SchemaGraph from raw schema data.

    Args:
        schema_data: Raw schema dictionary
        dataset_type: "spider" or "wikisql"

    Returns:
        SchemaGraph instance
    """
    if dataset_type == "wikisql":
        return parse_wikisql_schema(schema_data)
    elif dataset_type == "spider":
        return parse_spider_schema(schema_data)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test with Spider-like schema
    test_spider_schema = {
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
        "foreign_keys": [[8, 1], [9, 4]]  # Enrollment.student_id -> Student.student_id, Enrollment.course_id -> Course.course_id
    }

    graph = parse_spider_schema(test_spider_schema)
    print(f"Created graph: {graph}")
    print(f"\nTables:")
    for name, table in graph.tables.items():
        print(f"  {name}:")
        for col in table.columns:
            pk_marker = " (PK)" if col.is_primary_key else ""
            fk_marker = f" (FK -> {col.fk_reference})" if col.is_foreign_key else ""
            print(f"    - {col.name}: {col.dtype}{pk_marker}{fk_marker}")

    print(f"\nForeign Key Relationships:")
    for edge in graph.get_foreign_key_edges():
        print(f"  {edge.source} -> {edge.target}")

    print(f"\nTable Relationships:")
    for src, tgt in graph.get_table_relationships():
        print(f"  {src} -- {tgt}")
