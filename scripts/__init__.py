"""
GraphNL2SQL Scripts Package.

This package contains all the utility modules for the NL2SQL project:
- schema_graph: Graph-based schema representation
- text_linearization: Graph to text conversion
- training_utils: Training pipeline utilities
- testing_utils: Evaluation and testing utilities
"""

from .schema_graph import (
    SchemaGraph,
    TableNode,
    ColumnNode,
    Edge,
    EdgeType,
    NodeType,
    parse_spider_schema,
    parse_wikisql_schema,
    create_schema_graph,
)

from .text_linearization import (
    linearize_basic,
    linearize_detailed,
    linearize_typed,
    linearize_compact,
    linearize_structured,
    linearize_for_training,
)

from .training_utils import (
    TrainingConfig,
    load_config_from_json,
    save_config_to_json,
    load_datasets,
    load_model_and_tokenizer,
    setup_lora,
    train_phase1_wikisql,
    train_phase2_spider,
    init_wandb,
)

from .testing_utils import (
    generate_sql,
    load_finetuned_model,
    evaluate_model,
    run_quick_test,
    run_interactive_test,
    # EGD functions
    generate_sql_candidates,
    generate_sql_with_egd,
    evaluate_with_egd,
    validate_sql_syntax,
)

__all__ = [
    # Schema Graph
    "SchemaGraph", "TableNode", "ColumnNode", "Edge", "EdgeType", "NodeType",
    "parse_spider_schema", "parse_wikisql_schema", "create_schema_graph",
    # Text Linearization
    "linearize_basic", "linearize_detailed", "linearize_typed", 
    "linearize_compact", "linearize_structured", "linearize_for_training",
    # Training
    "TrainingConfig", "load_config_from_json", "save_config_to_json",
    "load_datasets", "load_model_and_tokenizer", "setup_lora",
    "train_phase1_wikisql", "train_phase2_spider", "init_wandb",
    # Testing
    "generate_sql", "load_finetuned_model", "evaluate_model",
    "run_quick_test", "run_interactive_test",
]

