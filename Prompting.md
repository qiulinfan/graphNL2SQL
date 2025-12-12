# Data downloading and precessing

Please read the README.md in the repository. It is a description of what we should do in this project.
During our coding, please write any code in the root directory. And please ignore the directories: /idea, /progress, /proposal


OK good. Now we start from: 

- Getting the dataset, and transform all data into our "Table&Column-level Hybrid Graph" design, and do text linearization, exporting the data into training set

Already good, but a few things can be better:
1. in the [input] and [schema], we see:

Database: 2-13797388-8 Schema: Table: table Columns: Date (text, PK), Opponent (text), Result (text), Score (text), Record (text), Streak (text) Question: Who was the opponent with a score of 109–108?

A few things:
(1) is the "Database: 2-13797388-8" really needed? it does not means anything
(2) Do you think we need some newlines and tabs actually, for the structure?

2. The [output] and [sql] seem to be the same thing





Ok, good.
Since spider is not working, we rather just comment them up. Can you find a better spider source?
And Can you Add synthetic_text_to_sql also as an option?
Notice: for each dataset, we only want at most 50000 data; otherwise it is too time-consuming to download and train.


Found two great alternatives:

Dataset	Total	Multi-table	JOINs
gretelai/synthetic_text_to_sql	100k	21k (21%)	20k
NumbersStation/NSText2SQL	289k	120k (41.5%)	48k


New Dataset Options
Dataset	Flag	Size	Multi-table	JOINs
gretelai/synthetic_text_to_sql	--gretelai	100k	21%	20k
NumbersStation/NSText2SQL	--nstext2sql	289k	41%	48k
sql-create-context	--sql-context	78k	2%	2k
Usage Examples



WikiSQL + Gretelai (recommended for good mix)

python prepare_training_data.py --style basic --gretelai



All multi-table datasets

```
python prepare_training_data.py --style basic --gretelai --nstext2sql
```





With semantic links

```
python prepare_training_data.py --style basic --gretelai --semantic
```



These datasets are not suitable for the first round training, actually.
We might still, need Spider.
Here is actually a download link:
https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view

For the advanded datasets, they can still serve as an option, but disabled by default.

The default dataset: just WikiSQL and Spider.



We will require: 

All data from Spider

20000 data from WikiSQL







## Wiki 和 Spider 的结构

**WikiSQL:**

- **Train:** ~56,355 examples
- **Dev:** ~8,421 examples
- **Multi-table:** 0% (single-table only by design)
- **With JOIN:** 0%

**Spider:**

- **Train:** 7,000 examples
- **Dev:** 1,034 examples
- **Multi-table:** ~100% (166 databases with complex schemas)
- **With JOIN:** ~26-38% of queries







Spider Dataset Summary

| Metric                  | Train         | Dev         |
| :---------------------- | :------------ | :---------- |
| **Total examples**      | 7,000         | 1,034       |
| **Multi-table schemas** | 100%          | 100%        |
| **Queries with JOIN**   | 39.8% (2,783) | 39.5% (408) |

Complex SQL Patterns (Train Set)

| Pattern           | Count | Percentage |
| :---------------- | :---- | :--------- |
| Nested subqueries | 1,019 | 14.6%      |
| GROUP BY          | 1,775 | 25.4%      |
| HAVING            | 427   | 6.1%       |
| ORDER BY          | 1,628 | 23.3%      |
| UNION             | 67    | 1.0%       |

**Total databases:** 166 (cross-domain schemas)



I found that, using WikiSQL mixed with Spider, is actually not a good choice. This is because: WikiSQL has too many data, which are very simple. It can let model forget about the complex patterns learned in Spider.



## 采样和训练计划

Our plan is actually:

- Sample 5000 data from WikiSQL, train for one-epoch as a warm-up -> in order to let model quickly learn the pattern of SQL.

	For the sampling: evenly choose different patterns.

- Then, we train on Spider for 3 epoches

For even pattern distribution, WikiSQL queries fall into these categories:

| Pattern                     | Example                  |
| :-------------------------- | :----------------------- |
| Simple SELECT               | SELECT col FROM table    |
| WHERE with comparisons      | WHERE col = 'value'      |
| Aggregation (COUNT/SUM/AVG) | SELECT COUNT(col)        |
| MIN/MAX                     | SELECT MAX(col)          |
| Combined (AGG + WHERE)      | SELECT AVG(col) WHERE... |

So, we plan to just use a little subset of WikiSQL, and the total Spider. Does this make sense?





Done! Here's the summary:

## Data Preparation Complete

### Generated Training Data

| File                | Examples | Description                 |
| :------------------ | :------- | :-------------------------- |
| wikisql_train.jsonl | 4,293    | Balanced WikiSQL (warm-up)  |
| wikisql_dev.jsonl   | 999      | Balanced WikiSQL dev        |
| spider_train.jsonl  | 7,000    | Full Spider (main training) |
| spider_dev.jsonl    | 1,034    | Spider dev                  |
| train.jsonl         | 11,293   | Combined training set       |
| dev.jsonl           | 2,033    | Combined dev set            |

### WikiSQL Pattern Distribution (Balanced)

```
where_only: 715 (16.7%)
count:      714 (16.6%)
sum:        714 (16.6%)
avg:        714 (16.6%)
max:        714 (16.6%)
min:        714 (16.6%)
select_only:  8 (0.2%)  ← limited by dataset
```

### Usage for Your Training Plan

**Phase 1 - WikiSQL warmup (1 epoch):**

```bash
# Data already prepared in wikisql_train.jsonl (4,293 balanced examples)
```

**Phase 2 - Spider training (3 epochs):**

```bash
# Data already prepared in spider_train.jsonl (7,000 examples)
```

**Regenerate data with different settings:**

```bash
# Balanced WikiSQL + full Spider
python prepare_training_data.py --wikisql-balanced 5000 --spider

# Spider only
python prepare_training_data.py --spider --skip-wikisql
```





### 调整 semantic link 的形式

````
Database: farm\nSchema Graph:\nTable: city(City_ID, Official_Name, Status, Area_km_2, Population, Census_Ranking)\nTable: farm(Farm_ID, Year, Total_Horses, Working_Horses, Total_Cattle, Oxen, Bulls, Cows, Pigs, Sheep_and_Goats)\nTable: farm_competition(Competition_ID, Year, Theme, Host_city_ID, Hosts)\nTable: competition_record(Competition_ID, Farm_ID, Rank)\nForeign Key: farm_competition.Host_city_ID -> city.City_ID\nForeign Key: competition_record.Farm_ID -> farm.Farm_ID\nForeign Key: competition_record.Competition_ID -> farm_competition.Competition_ID\nSemantic Link: (Competition_ID) ≈ (Farm_ID)\nSemantic Link: (Year) ≈ (Year)

Database: farm\nSchema Graph:\nTable: city(City_ID, Official_Name, Status, Area_km_2, Population, Census_Ranking)\nTable: farm(Farm_ID, Year, Total_Horses, Working_Horses, Total_Cattle, Oxen, Bulls, Cows, Pigs, Sheep_and_Goats)\nTable: farm_competition(Competition_ID, Year, Theme, Host_city_ID, Hosts)\nTable: competition_record(Competition_ID, Farm_ID, Rank)\nForeign Key: farm_competition.Host_city_ID -> city.City_ID\nForeign Key: competition_record.Farm_ID -> farm.Farm_ID\nForeign Key: competition_record.Competition_ID -> farm_competition.Competition_ID\nSemantic Link: (Competition_ID) ≈ (Farm_ID)\nSemantic Link: (Year) ≈ (Year)

Database: farm\nSchema Graph:\nTable: city(City_ID, Official_Name, Status, Area_km_2, Population, Census_Ranking)\nTable: farm(Farm_ID, Year, Total_Horses, Working_Horses, Total_Cattle, Oxen, Bulls, Cows, Pigs, Sheep_and_Goats)\nTable: farm_competition(Competition_ID, Year, Theme, Host_city_ID, Hosts)\nTable: competition_record(Competition_ID, Farm_ID, Rank)\nForeign Key: farm_competition.Host_city_ID -> city.City_ID\nForeign Key: competition_record.Farm_ID -> farm.Farm_ID\nForeign Key: competition_record.Competition_ID -> farm_competition.Competition_ID\nSemantic Link: (Competition_ID) ≈ (Farm_ID)\nSemantic Link: (Year) ≈ (Year)
````



I have three questions:

1. Why does each Table has the same name?
2. I feel like semantic link should include "which table is this column from"
3. (Competition_ID) ≈ (Farm_ID) is correct?



I feel like, one more important thing:

- when there is already the foreign key correspondence (strong relation), do not establish the same semantic link (weak realation)



#### 难绷 claude 处理 semantic 太近的问题

I added a filter in compute_semantic_links() that checks if both columns are _id columns with different base names:

```python
# Skip pairs where both are just ID columns (spurious matches)
if col1_name.endswith('_id') and col2_name.endswith('_id'):
    # Only allow if the base names are actually similar
    base1 = col1_name.replace('_id', '')
    base2 = col2_name.replace('_id', '')
    if base1 != base2:
        continue
```

**Logic:**

| Column 1       | Column 2 | Base 1      | Base 2 | Result                  |
| :------------- | :------- | :---------- | :----- | :---------------------- |
| Competition_ID | Farm_ID  | competition | farm   | **SKIP** (different)    |
| user_id        | user_id  | user        | user   | ALLOW (same)            |
| Year           | Year     | -           | -      | ALLOW (not _id columns) |

This filters out spurious matches where the embedding model thinks Competition_ID ≈ Farm_ID just because they both end in _ID.



我:
Don't you feel like this method is WAY too shallow?

- There actually, can be a pair of x_id and y_id that are semantically close!



I propose: separate "single" and "composite" words



First, propose a list of "weak semantic words":

```
id, key, code, number, uuid, guid, no, idx, index
```

(Including upper/lower cases)

And below is the rule for the case of composite words:

- For two words (x,y), if x has a "_" or " ", but y does not, i.e. one of them is composite: then compare the semantic meaning of the two parts of x, respectively to the semantic meaning of y. If either one is close enough -> then x and y are semantically close
	- special case: if any part of x is in the "weak semantic words", then we only view x as a single word, that is, the other part. This is to prevent things like "object_id = id". This is not true.
- For two words (x,y), if they both have "_" or " ", i.e. they are both composite words, then MUST compare respective the first part of x to the first part of y, and then the second part of x to the second part of y. Only if they are both close enough, we then say x and y are semantically close.





我是天才.

Done! Here's the new semantic link algorithm:

1. Weak Semantic Words (filtered out)

```python
WEAK_WORDS = {'id', 'key', 'code', 'number', 'uuid', 'guid', 'no', 'idx', 'index', 
              'pk', 'fk', 'ref', 'num', 'seq', 'type', 'flag', 'status'}
```

2. Comparison Rules

| Structure                  | Rule                  | Example                     |
| :------------------------- | :-------------------- | :-------------------------- |
| **Single vs Single**       | Direct compare        | competition vs farm         |
| **Composite vs Single**    | ANY part matches      | birth_date vs date → ✓      |
| **Composite vs Composite** | BOTH parts must match | user_name vs user_email → ✗ |

3. Examples

| Columns                   | After Filter               | Result               |
| :------------------------ | :------------------------- | :------------------- |
| Competition_ID vs Farm_ID | competition vs farm        | ✗ Different entities |
| user_id vs user           | user vs user               | ✓ Same               |
| object_id vs id           | object vs id               | ✗ Different          |
| customer_id vs client_id  | customer vs client         | ✓ Semantically close |
| start_time vs end_time    | [start,time] vs [end,time] | ✗ start≠end          |



### 最终修改

I am a bit concerned about whether this form of graph can be good enough for the 7B llm to understand.

For humans, it is intuiative, only if you view \n into a real "newline"

Does it actually understand that, the \n seperate things?

Database: farm\nSchema Graph:\nTable: city(City_ID, Official_Name, Status, Area_km_2, Population, Census_Ranking)\nTable: farm(Farm_ID, Year, Total_Horses, Working_Horses, Total_Cattle, Oxen, Bulls, Cows, Pigs, Sheep_and_Goats)\nTable: farm_competition(Competition_ID, Year, Theme, Host_city_ID, Hosts)\nTable: competition_record(Competition_ID, Farm_ID, Rank)\nForeign Key: farm_competition.Host_city_ID -> city.City_ID\nForeign Key: competition_record.Farm_ID -> farm.Farm_ID\nForeign Key: competition_record.Competition_ID -> farm_competition.Competition_ID\nSemantic Link: farm.Year ≈ farm_competition.Yea



Thus, I plan to add section tags: [], and bullet / indentation

```
[DATABASE] 
farm

[TABLES]
city:
    City_ID
    Official_Name
    Status
    Area_km_2
    Population
    Census_Ranking
farm:
    Host_city_ID
    Farm_ID
    Year
    Total_Horses
    Working_Horses
farm_competition:
    Competition_ID
    Year
    Theme
    Host_city_ID
    Hosts

[FOREIGN KEYS]
farm.Host_city_ID -> city.City_ID
farm_competition.Host_city_ID -> city.City_ID

[SEMANTIC LINKS]
farm.Year ≈ farm_competition.Year

```

We use:

- [] and \n\n for section parting
- In one section, \n to show a new item
- For second-level itemization, i.e. columns below the tables: we use four spaces, as indentation, to show difference

Is that good?





# Training





OK, OK. It is time we GET to train!

I have actually, already written some code, but in an ipynb file. I hope I can put them in .py files in order to make it clearer.

Notice: my code is incomplete.



```
## Step 2: Import Libraries
import torch
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from datasets import load_dataset, Dataset
import os

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    

## Step 3: Configuration
# Model - WikiSQL finetuned T5-base
MODEL_NAME = "mrm8488/t5-base-finetuned-wikiSQL"
OUTPUT_DIR = "./t5_wikisql_lora_finetuned"

# LoRA hyperparameters
LORA_R = 16              # Rank - higher = more capacity, more memory
LORA_ALPHA = 32          # Scaling factor (typically 2x r)
LORA_DROPOUT = 0.1       # Dropout for regularization

# Training hyperparameters
EPOCHS = 3
BATCH_SIZE = 8           # T5-base is smaller, can use larger batch
GRADIENT_ACCUMULATION = 2  # Effective batch = BATCH_SIZE * GRADIENT_ACCUMULATION
LEARNING_RATE = 1e-4
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128   # SQL queries are typically shorter

# WandB Configuration
WANDB_PROJECT = "t5-wikisql-lora-finetuning"
WANDB_RUN_NAME = "t5-base-wikisql-lora"




## Step 4: Load Model and Tokenizer
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
# T5-base is smaller (~220M params), so standard loading works well
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

print(f"Model loaded. Parameters: {model.num_parameters():,}")



## Step 5: Apply LoRA
# Define LoRA configuration
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q", "v", "k", "o"],  # T5 attention modules
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()




## Step 6: Prepare Your Dataset
dataset = Dataset.from_pandas(df)
subset_size = int(len(dataset) * 0.33)
dataset = dataset.shuffle(seed=42).select(range(subset_size))

print(f"Dataset size: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
print(f"\nSample:")
print(dataset[0])




def format_input(question: str, schema: str) -> str:
    """Format input for the T5 model.

    Based on your data format:
    Schema: Table: table_xxx Columns: col1 (type), col2 (type)
    """
    return f"translate English to SQL: {question} | {schema}"


def preprocess_function(examples):
    """Tokenize inputs and labels."""
    # Format inputs using question and schema
    inputs = [
        format_input(q, s)
        for q, s in zip(examples["question"], examples["schema"])
    ]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LENGTH,
        padding="max_length",
        truncation=True,
    )

    # Tokenize targets (sql column, renamed from 'output')
    labels = tokenizer(
        examples["sql"],  # This was 'output' in your original data
        max_length=MAX_TARGET_LENGTH,
        padding="max_length",
        truncation=True,
    )

    # Replace padding with -100 for loss calculation
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Preprocess dataset
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
)

# Split into train/eval
split = processed_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]


print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")


## Step 7: Training
# Training hyperparameters (UPDATED for large dataset)
EPOCHS = 3
BATCH_SIZE = 8                  # Increase if GPU memory allows (8-16 for T5-base, 4-8 for T5-large)
GRADIENT_ACCUMULATION = 8         # Effective batch = 8 * 4 = 32
LEARNING_RATE = 3e-4              # Can be slightly higher with larger dataset
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 256

# Initialize WandB
wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    config={
        "model_name": MODEL_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "max_source_length": MAX_SOURCE_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
    }
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

# Training arguments with WandB logging
# training_args = Seq2SeqTrainingArguments(
#     output_dir=OUTPUT_DIR,
#     num_train_epochs=EPOCHS,
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE,
#     gradient_accumulation_steps=GRADIENT_ACCUMULATION,
#     learning_rate=LEARNING_RATE,
#     warmup_ratio=0.1,
#     weight_decay=0.01,
#     logging_steps=10,
#     eval_strategy="steps",
#     eval_steps=50,
#     save_steps=50,
#     save_total_limit=3,
#     load_best_model_at_end=True,
#     predict_with_generate=True,
#     generation_max_length=MAX_TARGET_LENGTH,
#     fp16=torch.cuda.is_available(),
#     # WandB Integration
#     report_to="wandb",
#     run_name=WANDB_RUN_NAME,
# )

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.05,            # Reduced - with more steps, less warmup needed
    weight_decay=0.01,

    # Logging - less frequent with large dataset
    logging_steps=10,            # Was 10, now every 100 steps

    # Evaluation - less frequent to speed up training
    eval_strategy="steps",
    eval_steps=500,               # Was 50, now every 500 steps

    # Saving - less frequent
    save_steps=500,               # Was 50, now every 500 steps
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # Add this to track best model
    greater_is_better=False,            # Lower loss is better

    predict_with_generate=False,  # DISABLE for faster training (generation is slow)
    generation_max_length=MAX_TARGET_LENGTH,
    fp16=torch.cuda.is_available(),

    # WandB Integration
    report_to="wandb",
    run_name=WANDB_RUN_NAME,

    # Optional: Gradient checkpointing for memory efficiency
    # gradient_checkpointing=True,  # Uncomment if OOM errors
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


# Start training!
trainer.train()

# Save the final model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# Finish WandB run
wandb.finish()
print("WandB run finished. Check your dashboard for loss graphs!")

```

## 调整训练结构
<<<<<<< Updated upstream

There are a few things I want to mention, and should modify.

- First, the model is too small. I think we should use a ~5B llm model as basis.
- Secondly, this version of training cut down on training data, but after finishing the data preprocessing module, we do not hope to lose any training data. We will stricly use up our training data: 1 epoch of WikiSQL for Heating, and 3 epoches of Spider.
- Also, this training scipt does not seem to store checkpoint models, but we use do it (maybe after one epoch?)
=======

There are a few things I want to mention, and should modify.

- First, the model is too small. I think we should use a ~5B llm model as basis.
- Secondly, this version of training cut down on training data, but after finishing the data preprocessing module, we do not hope to lose any training data. We will stricly use up our training data: 1 epoch of WikiSQL for Heating, and 3 epoches of Spider.
- Also, this training scipt does not seem to store checkpoint models, but we use do it (maybe after one epoch?)





OK, OK, this is good. But I also hope that I can modify parameters in the .ipynb file that runs the commands, and also, check the number of parameters (like: trainable params: 3,538,944 || all params: 226,442,496 || trainable%: 1.5628) in the .ipynb file.

Also, the  training dataset size, columns, etc





# Testing

Adding save the model to load the model and test it.
here is reference

```
from peft import PeftModel

def load_finetuned_model(adapter_path: str):
    """Load the finetuned LoRA model."""
    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    return model, tokenizer


# Example usage:
# model, tokenizer = load_finetuned_model("./t5_wikisql_lora_finetuned")


# Clone Spider dataset
!rm -rf spider
!git clone https://github.com/taoyds/spider.git



import json
import os
from collections import defaultdict

def linearize_schema_simple(db_schema):
    """Simple schema format: col1, col2, col3 (WikiSQL style)"""
    table_names = db_schema["table_names_original"]
    column_names = db_schema["column_names_original"]

    # Collect all columns
    all_columns = []
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx >= 0:
            all_columns.append(col_name)

    return ", ".join(all_columns)


def load_spider_for_wikisql_model(spider_path="./spider"):
    """Load Spider dataset formatted for WikiSQL model."""

    with open(os.path.join(spider_path, "tables.json"), "r") as f:
        tables_data = json.load(f)

    with open(os.path.join(spider_path, "train_spider.json"), "r") as f:
        train_data = json.load(f)

    with open(os.path.join(spider_path, "dev.json"), "r") as f:
        dev_data = json.load(f)

    db_schemas = {db["db_id"]: db for db in tables_data}

    def process_examples(data):
        examples = []
        for item in data:
            db_id = item["db_id"]
            if db_id not in db_schemas:
                continue
            schema_text = linearize_schema_simple(db_schemas[db_id])
            examples.append({
                "question": item["question"],
                "schema": schema_text,
                "sql": item["query"]
            })
        return examples

    train_examples = process_examples(train_data)
    dev_examples = process_examples(dev_data)

    print(f"Train: {len(train_examples)}, Dev: {len(dev_examples)}")
    return Dataset.from_list(train_examples), Dataset.from_list(dev_examples)


# Uncomment to load Spider dataset:
# dataset, eval_raw = load_spider_for_wikisql_model("./spider")
# print(f"\nSample:")
# print(f"Question: {dataset[0]['question']}")
# print(f"Schema: {dataset[0]['schema'][:100]}...")
# print(f"SQL: {dataset[0]['sql']}")
```

>>>>>>> Stashed changes
