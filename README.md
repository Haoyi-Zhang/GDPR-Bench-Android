# GDPR-Bench-Android

**A Comprehensive Benchmark for Evaluating GDPR Compliance Detection in Android Applications**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Tasks](#tasks)
- [Methods](#methods)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**GDPR-Bench-Android** is a comprehensive benchmark designed to evaluate the ability of Large Language Models (LLMs) and automated analysis methods to detect GDPR (General Data Protection Regulation) compliance violations in Android applications. The benchmark provides:

- **Annotated Dataset**: A curated collection of real-world Android malware and spyware applications with annotated GDPR violations
- **Multi-granularity Tasks**: Two evaluation tasks covering different granularities (file, module, line-level) and code snippet classification
- **Multiple Methods**: Support for LLM-based, agentic (ReAct, RAG), and static analysis (AST-based) approaches
- **Comprehensive Evaluation**: Detailed metrics and comparison tools for assessing model performance

### Key Features

✅ **Real-world Dataset**: 15 Android RAT/spyware applications with expert-annotated GDPR violations  
✅ **Multi-level Analysis**: File-level, module-level, and line-level violation detection  
✅ **Flexible Framework**: Easy integration of new models and methods  
✅ **Automated Evaluation**: Built-in evaluation scripts with multiple metrics  
✅ **Extensible**: Support for LLMs, agentic methods, and rule-based approaches

---

## 🚀 Features

### 1. **Comprehensive Dataset**
- **15 Android Applications**: Real-world RAT (Remote Access Trojan) and spyware apps
- **Multi-label Annotations**: Each code sample annotated with violated GDPR articles
- **Multi-granularity Coverage**: File, module, and line-level annotations

### 2. **Two Evaluation Tasks**
- **Task 1**: Multi-granularity GDPR violation detection (file, module, line level)
- **Task 2**: Code snippet classification for GDPR article identification

### 3. **Multiple Detection Methods**
- **LLM-based**: GPT-4o, Claude, Gemini, Qwen, DeepSeek, etc.
- **Agentic**: ReAct agent with GDPR tools, RAG-based retrieval
- **Static Analysis**: AST-based pattern matching with formal rules

### 4. **Automated Evaluation**
- Accuracy@K metrics for Task 1
- Precision, Recall, F1-score for Task 2
- Confusion matrices and comparison tables
- CSV exports for further analysis

---

## 📁 Project Structure

```
GDPR-Bench-Android/
├── repos/                          # Android application repositories
│   ├── AhMyth-Android-RAT/
│   ├── Android_Spy_App/
│   ├── AndroRAT/
│   └── ... (15 applications total)
│
├── methods/                        # Detection method implementations
│   ├── base_method.py             # Abstract base class for methods
│   ├── method_factory.py          # Factory for creating method instances
│   ├── config.py                  # Method configurations
│   ├── react_method.py            # ReAct agent implementation
│   ├── react_tools.py             # GDPR tools for ReAct
│   ├── rag_method.py              # RAG-based method
│   ├── formal_ast_method.py       # AST-based static analysis
│   ├── formal_gdpr_detector.py    # GDPR rule detector
│   ├── multilang_ast_parser.py    # Multi-language AST parser
│   └── README.md                  # Methods documentation
│
├── GDPR_dataset.json              # Raw annotated dataset
├── task1_dataset.json             # Task 1 formatted dataset
├── task2_dataset.json             # Task 2 formatted dataset
│
├── create_task1_dataset.py        # Convert raw data to Task 1 format
├── create_task2_dataset.py        # Convert raw data to Task 2 format
├── predict.py                     # Main prediction script
├── evaluate_model.py              # Model evaluation script
│
├── task1_predictions/             # Task 1 prediction results
├── task2_predictions/             # Task 2 prediction results
├── task1_eval_results/            # Task 1 evaluation results
├── task2_eval_results/            # Task 2 evaluation results
├── evaluation_comparison/         # Comparison analysis results
│
├── logs/                          # Execution logs
└── README.md                      # This file
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/GDPR-Bench-Android.git
cd GDPR-Bench-Android
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `numpy` - Numerical computing
- `pandas` - Data manipulation and analysis
- `scikit-learn` - Machine learning metrics
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `requests` - HTTP library for API calls

**Method-specific Dependencies:**

For **LLM-based methods**:
```bash
pip install openai anthropic google-generativeai
```

For **ReAct Agent**:
```bash
pip install langchain langgraph langchain-openai
```

For **RAG Method**:
```bash
pip install langchain chromadb sentence-transformers
```

For **AST-based Analysis**:
```bash
pip install javalang tree-sitter tree-sitter-java tree-sitter-kotlin
```

### Step 3: Configure API Keys

**Option 1: Environment Variables (Recommended)**

```bash
# On Linux/Mac
export OPENAI_API_KEY='your-api-key-here'
export OPENAI_API_BASE='https://api.openai.com/v1'  # Optional: custom endpoint

# On Windows (Command Prompt)
set OPENAI_API_KEY=your-api-key-here

# On Windows (PowerShell)
$env:OPENAI_API_KEY='your-api-key-here'
```

**Option 2: .env File**

```bash
# Copy the example configuration file
cp config.env.example .env

# Edit .env and add your API keys
# Note: .env is automatically ignored by git
```

**For Multiple Model Providers:**

```bash
export OPENAI_API_KEY='your-openai-key'
export ANTHROPIC_API_KEY='your-anthropic-key'
export GOOGLE_API_KEY='your-google-key'
```

---

## ⚡ Quick Start

### 1. Prepare the Dataset

Generate Task 1 and Task 2 datasets from the raw annotated data:

```bash
# Create Task 1 dataset (multi-granularity detection)
python create_task1_dataset.py GDPR_dataset.json task1_dataset.json

# Create Task 2 dataset (snippet classification)
python create_task2_dataset.py GDPR_dataset.json task2_dataset.json
```

### 2. Configure API Keys

Set your API key (see [Installation](#installation) for details):

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 3. Run Predictions

**Using LLM models:**

```bash
# Single model, single task
python predict.py --models=gpt-4o --tasks=1

# Multiple models, both tasks
python predict.py --models=gpt-4o,claude-sonnet-4-5-20250929 --tasks=all

# All models on both tasks
python predict.py --models=all --tasks=all
```

**Using Alternative Methods:**

```bash
# ReAct agent
python predict.py --models=react --tasks=all

# AST-based static analysis
python predict.py --models=ast-based --tasks=all

# RAG-based retrieval
python predict.py --models=rag --tasks=all
```

**Advanced Options:**

```bash
# Exclude specific apps
python predict.py --models=gpt-4o --tasks=1 --exclude-apps=Android_Spy_App,Dash

# Incremental mode (resume from existing predictions)
python predict.py --models=gpt-4o --tasks=1 --incremental

# Disable monitor screen clearing
python predict.py --models=gpt-4o --tasks=1 --disable-monitor-clear
```

### 4. Evaluate Results

```bash
# Evaluate specific model
python evaluate_model.py --models=gpt-4o --task=1

# Evaluate all models
python evaluate_model.py --models=all --task=all
```

### 5. View Results

Evaluation results are saved in:
- `task1_eval_results/` - Task 1 detailed results
- `task2_eval_results/` - Task 2 detailed results
- `evaluation_comparison/` - Unified comparison tables and analysis

Check the main comparison file:
```bash
cat evaluation_comparison/all_models_comparison.txt
```

---

## 📊 Dataset

### Overview

The GDPR-Bench-Android dataset consists of **15 real-world Android RAT and spyware applications** with expert-annotated GDPR violations. Each code snippet is annotated with:
- **Repository URL**: Source repository
- **App Name**: Application name
- **Commit ID**: Specific commit
- **Code Snippet**: The actual code
- **Violated Article(s)**: GDPR article number(s)
- **Annotation Note**: Description of the violation

### Covered GDPR Articles

The dataset covers **23 GDPR articles** commonly violated in Android applications:

| Article | Description |
|---------|-------------|
| 5 | Principles of processing (lawfulness, fairness, transparency) |
| 6 | Lawfulness of processing |
| 7 | Conditions for consent |
| 8 | Conditions for child's consent |
| 9 | Processing of special categories of personal data |
| 12 | Transparent information and communication |
| 13 | Information when data are collected |
| 14 | Information when data not obtained from subject |
| 15 | Right of access |
| 16 | Right to rectification |
| 17 | Right to erasure |
| 18 | Right to restriction of processing |
| 19 | Notification obligation |
| 21 | Right to object |
| 25 | Data protection by design and default |
| 30 | Records of processing activities |
| 32 | Security of processing |
| 33 | Personal data breach notification |
| 35 | Data protection impact assessment |
| 44 | General principle for transfers |
| 46 | Transfers with appropriate safeguards |
| 58 | Powers of supervisory authorities |
| 83 | Administrative fines |

### Dataset Statistics

```json
{
  "total_applications": 15,
  "total_violations": 17500+,
  "unique_files": 500+,
  "granularity_levels": 3,
  "articles_covered": 23
}
```

### Data Format

**Raw Dataset (`GDPR_dataset.json`):**
```json
[
  {
    "app_name": "Android_Spy_App",
    "repo_url": "https://github.com/abhinavsuthar/Android_Spy_App",
    "Commit_ID": "d524cef38b5526861a724f3f9b08b8b9a4a3d06a",
    "violated_article": 6,
    "code_snippet": "manager.openCamera(camerId, stateCallback, null);",
    "code_snippet_path": "app/src/main/java/me/hawkshaw/test/MainActivity2.java: line 202",
    "annotation_note": "Camera access without lawful basis for processing"
  }
]
```

---

## 🎯 Tasks

### Task 1: Multi-Granularity GDPR Violation Detection

**Objective**: Given a file from an Android application, detect GDPR violations at three granularity levels:

1. **File-level**: Identify violated articles for the entire file
2. **Module-level**: Identify violations for specific modules/classes
3. **Line-level**: Identify violations for specific code lines

**Input Format:**
```json
{
  "repo_url": "...",
  "app_name": "...",
  "Commit_ID": "...",
  "file_level_violations": [
    {"file_path": "path/to/File.java", "violated_articles": []}
  ],
  "module_level_violations": [
    {"file_path": "...", "module_name": "ClassName", "violated_articles": []}
  ],
  "line_level_violations": [
    {"file_path": "...", "line_spans": "10-15", "violated_articles": [], "violation_description": "..."}
  ]
}
```

**Evaluation Metrics**: Accuracy@K (K=1,2,3,4,5) for each granularity level

### Task 2: Code Snippet Classification

**Objective**: Given a code snippet, classify which GDPR article(s) it violates

**Input Format:**
```json
{
  "repo_url": "...",
  "app_name": "...",
  "Commit_ID": "...",
  "code_snippet_path": "...",
  "code_snippet": "...",
  "violated_articles": []
}
```

**Evaluation Metrics**: Accuracy, Precision, Recall, F1-score (macro-averaged), Confusion Matrix

---

## 🔬 Methods

### 1. LLM-based Methods

Direct prompting of Large Language Models with GDPR context and code.

**Supported Models:**
- `gpt-4o` - OpenAI GPT-4 Optimized
- `o1` - OpenAI O1 (reasoning model)
- `claude-sonnet-4-5-20250929` - Anthropic Claude Sonnet 4.5
- `claude-3-7-sonnet-20250219` - Anthropic Claude 3.7 Sonnet
- `gemini-2.5-pro-thinking` - Google Gemini 2.5 Pro
- `qwen2.5-72b-instruct` - Qwen 2.5 72B
- `deepseek-r1` - DeepSeek R1

**Configuration:**
```python
# In methods/config.py
API_URL = "https://api.nuwaapi.com/v1/chat/completions"
API_KEY = "your-api-key"
```

### 2. ReAct Agent

Agentic approach using Reasoning and Acting framework with specialized GDPR tools.

**Tools:**
- `gdpr_lookup`: Look up GDPR article definitions
- `code_search`: Search for sensitive API patterns
- `rule_check`: Check code against formal GDPR rules

**Usage:**
```bash
python predict.py --models=react --tasks=all
```

**Configuration:**
```python
# In methods/config.py
METHOD_CONFIGS = {
    'react': {
        'model': 'gpt-4o',
        'max_iterations': 5,
        'temperature': 0.0,
        'timeout': 300
    }
}
```

### 3. RAG (Retrieval-Augmented Generation)

Retrieval-based method using GDPR knowledge base with semantic search.

**Features:**
- Vector store with GDPR articles and case law
- Semantic similarity search
- Context-aware LLM reasoning

**Usage:**
```bash
python predict.py --models=rag --tasks=all
```

### 4. AST-based Static Analysis

Rule-based method using Abstract Syntax Tree analysis with formal GDPR violation patterns.

**Features:**
- Multi-language support (Java, Kotlin, PHP, JavaScript, Python)
- Pattern-based violation detection
- Data flow and control flow analysis
- Fast, deterministic analysis

**Usage:**
```bash
python predict.py --models=ast-based --tasks=all
```

**Configuration:**
```python
# In methods/config.py
METHOD_CONFIGS = {
    'ast-based': {
        'languages': ['java', 'kotlin', 'php', 'javascript', 'python'],
        'enable_data_flow': True,
        'enable_control_flow': True,
        'strict_mode': False
    }
}
```

### Adding New Methods

To add a new detection method:

1. Create a new file in `methods/` (e.g., `my_method.py`)
2. Inherit from `BaseMethod` and implement required methods:
   ```python
   from methods.base_method import BaseMethod
   
   class MyMethod(BaseMethod):
       def initialize(self):
           # Setup your method
           pass
       
       def predict_file_level(self, file_path, code, **kwargs):
           # Implement file-level prediction
           return [6, 7, 32]  # Example: violated articles
       
       def predict_module_level(self, file_path, module_name, code, **kwargs):
           # Implement module-level prediction
           return [6, 7]
       
       def predict_line_level(self, file_path, line_spans, code, description, **kwargs):
           # Implement line-level prediction
           return [32]
       
       def predict_snippet(self, snippet, snippet_path="", **kwargs):
           # Implement snippet prediction
           return [6, 7, 32]
   ```

3. Register in `methods/method_factory.py`:
   ```python
   from methods.my_method import MyMethod
   
   class MethodFactory:
       @staticmethod
       def create(method_name: str, config: dict) -> BaseMethod:
           if method_name == 'my-method':
               return MyMethod(config)
           # ... other methods
   ```

4. Add configuration in `methods/config.py`:
   ```python
   METHOD_CONFIGS = {
       'my-method': {
           'param1': 'value1',
           'param2': 'value2'
       }
   }
   ```

5. Use your method:
   ```bash
   python predict.py --models=my-method --tasks=all
   ```

---

## 💻 Usage

### Command-Line Interface

**Basic Syntax:**
```bash
python predict.py --models=<models> --tasks=<tasks> [options]
```

**Parameters:**
- `--models`: Comma-separated model names or `all`
- `--tasks`: Comma-separated task numbers (1,2) or `all`
- `--exclude-apps`: Comma-separated app names to exclude
- `--incremental`: Resume from existing predictions
- `--disable-monitor-clear`: Disable screen clearing in monitor

**Examples:**

```bash
# Single model, single task
python predict.py --models=gpt-4o --tasks=1

# Multiple models, multiple tasks
python predict.py --models=gpt-4o,claude-sonnet-4-5-20250929 --tasks=1,2

# All models on all tasks
python predict.py --models=all --tasks=all

# Exclude specific apps
python predict.py --models=gpt-4o --tasks=1 --exclude-apps=Android_Spy_App,Dash

# Incremental mode (resume incomplete predictions)
python predict.py --models=gpt-4o --tasks=1 --incremental

# Combine multiple options
python predict.py --models=react,ast-based --tasks=all --exclude-apps=Dash --incremental
```

### Programmatic Usage

```python
from methods.method_factory import MethodFactory
from methods.config import get_method_config

# Create method instance
config = get_method_config('react')
method = MethodFactory.create('react', config)

# Analyze code
code = """
public void collectLocation() {
    LocationManager manager = (LocationManager) getSystemService(LOCATION_SERVICE);
    Location location = manager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
    sendToServer(location);
}
"""

# Predict violations
articles = method.predict_snippet(code, "LocationCollector.java")
print(f"Violated GDPR articles: {articles}")

# Cleanup
method.cleanup()
```

### Real-time Monitoring

The prediction script includes a real-time monitoring interface:

```
================================================================================
Real-time Model Prediction Progress Monitor
================================================================================
Task 1:
  [gpt-4o                  ] Status: processing Progress: 45/150 Current: Android_Spy_App - MainActivity.java...
  [react                   ] Status: processing Progress: 32/150 Current: AndroRAT - ServiceHandler.java...
Task 2:
  [gpt-4o                  ] Status: done       Progress: 500/500 Current: Complete
  [react                   ] Status: processing Progress: 421/500 Current: snippet_421...
--------------------------------------------------------------------------------
Output files:
  task1_predictions/gpt-4o_task1_predictions.json: 45 records
  task1_predictions/react_task1_predictions.json: 32 records
  task2_predictions/gpt-4o_task2_predictions.json: 500 records
  task2_predictions/react_task2_predictions.json: 421 records
--------------------------------------------------------------------------------
Press Ctrl+C to gracefully exit and auto-save progress.
================================================================================
```

---

## 📈 Evaluation

### Running Evaluation

```bash
# Evaluate specific models
python evaluate_model.py --models=gpt-4o,react --task=1

# Evaluate all models
python evaluate_model.py --models=all --task=all

# Evaluate only Task 1
python evaluate_model.py --models=all --task=1

# Evaluate only Task 2
python evaluate_model.py --models=all --task=2
```

### Evaluation Metrics

**Task 1: Multi-granularity Detection**

- **Accuracy@K**: Proportion of correct articles in top-K predictions
- Computed separately for each granularity level (file, module, line)
- K ∈ {1, 2, 3, 4, 5}

**Task 2: Snippet Classification**

- **Accuracy**: Overall classification accuracy
- **Precision**: Macro-averaged precision across all articles
- **Recall**: Macro-averaged recall across all articles
- **F1-score**: Macro-averaged F1-score
- **Confusion Matrix**: Per-article confusion matrix (normalized)

### Output Files

**Task 1 Results:**
```
task1_eval_results/
├── gpt-4o_task1_eval.txt          # Detailed evaluation
├── react_task1_eval.txt
├── ast-based_task1_eval.txt
└── all_models_task1_eval.txt      # Aggregated results
```

**Task 2 Results:**
```
task2_eval_results/
├── gpt-4o_task2_eval.txt          # Detailed evaluation
├── gpt-4o_task2_confusion.png     # Confusion matrix
├── react_task2_eval.txt
├── react_task2_confusion.png
└── all_models_task2_eval.txt      # Aggregated results
```

**Comparison Analysis:**
```
evaluation_comparison/
├── all_models_comparison.txt       # Unified comparison table ⭐
├── evaluation_summary.txt          # Summary report
├── task1_file_level_violations.csv
├── task1_module_level_violations.csv
├── task1_line_level_violations.csv
└── task2_metrics.csv
```

### Sample Evaluation Output

**Task 1 (File-level):**
```
==== file_level_violations ====
Accuracy@1: 0.4521
Accuracy@2: 0.6234
Accuracy@3: 0.7156
Accuracy@4: 0.7689
Accuracy@5: 0.8012
```

**Task 2:**
```
Accuracy: 0.6523
Macro-Precision: 0.6789
Macro-Recall: 0.6432
Macro-F1: 0.6567

Per-class metrics:
  Article 5: Precision=0.72, Recall=0.68, F1=0.70
  Article 6: Precision=0.75, Recall=0.71, F1=0.73
  Article 7: Precision=0.69, Recall=0.65, F1=0.67
  ...
```

---

## 📊 Results

### Performance Comparison

The following table shows the performance of different methods on the GDPR-Bench-Android benchmark:

**Task 1: File-Level Violations (Accuracy@1)**

| Method | File-Level | Module-Level | Line-Level |
|--------|-----------|-------------|-----------|
| GPT-4o | 0.4521 | 0.3892 | 0.3456 |
| Claude Sonnet 4.5 | 0.4678 | 0.4012 | 0.3589 |
| ReAct Agent | 0.5123 | 0.4567 | 0.4123 |
| AST-based | 0.3890 | 0.3456 | 0.3012 |

**Task 2: Snippet Classification**

| Method | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| GPT-4o | 0.6523 | 0.6789 | 0.6432 | 0.6567 |
| Claude Sonnet 4.5 | 0.6678 | 0.6912 | 0.6589 | 0.6723 |
| ReAct Agent | 0.7012 | 0.7234 | 0.6890 | 0.7045 |
| AST-based | 0.5890 | 0.6123 | 0.5678 | 0.5876 |

*Note: These are example results. Actual performance may vary based on model versions and configurations.*

### Key Findings

1. **Agentic methods** (ReAct) tend to outperform direct LLM prompting due to iterative reasoning and tool usage
2. **Line-level detection** is significantly more challenging than file-level detection
3. **AST-based methods** provide fast, deterministic results but may have lower coverage
4. **Larger models** (e.g., 70B+ parameters) generally perform better but at higher computational cost

---

## 🤝 Contributing

We welcome contributions to GDPR-Bench-Android! Here are some ways you can contribute:

### Adding New Methods

1. Fork the repository
2. Create a new method class in `methods/`
3. Implement the `BaseMethod` interface
4. Add tests and documentation
5. Submit a pull request

### Expanding the Dataset

1. Identify new Android applications with GDPR violations
2. Follow the annotation guidelines (see `docs/annotation_guidelines.md`)
3. Submit annotated data with detailed violation descriptions

### Improving Documentation

- Fix typos or unclear sections
- Add usage examples
- Translate documentation to other languages

### Reporting Issues

If you encounter bugs or have feature requests:
1. Check existing issues to avoid duplicates
2. Provide detailed reproduction steps
3. Include system information and logs

---


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Thanks to all contributors who helped build this benchmark
- Android application repositories used in this dataset
- GDPR legal experts who provided domain knowledge
- Open-source community for tools and libraries

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

## 🔗 Resources

- [GDPR Official Text](https://gdpr-info.eu/)
- [Android Security Best Practices](https://developer.android.com/privacy-and-security/security-best-practices)
- [LangChain Documentation](https://python.langchain.com/)
- [Tree-sitter Parsers](https://tree-sitter.github.io/tree-sitter/)

---

<div align="center">
  <b>⭐ If you find this project useful, please consider giving it a star! ⭐</b>
</div>
