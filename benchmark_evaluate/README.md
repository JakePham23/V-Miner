# MinerU Benchmark Evaluation Utility

This folder contains a comprehensive, academically reputable benchmark evaluation utility for Document Layout Analysis and OCR (PDF-to-Markdown) systems. 

It is designed to compare a folder of **Ground Truth** (GT) markdown files with a folder of **Predicted** (Test) markdown files.

## Academic & Industry Credibility
To ensure the results are 100% reliable for scientific papers and commercial reports, this tool relies on standard metrics:

### 1. Traditional Metrics (Text & Table)
* **Normalized Edit Distance (Similarity):** Uses `rapidfuzz` (C++ optimized Levenshtein distance) to measure structural text similarity.
* **CER & WER (Character & Word Error Rate):** Standard OCR metrics. Computed via `jiwer`.
* **BLEU (1-4):** Standard metric for text generation/reconstruction, measuring n-gram overlap. Uses `nltk`.
* **METEOR:** Captures synonymy and stemming accuracy. Uses `nltk`.
* **TEDS (Tree Edit Distance-based Similarity):** The official metric introduced by IBM (CVPR 2020) and used by OpenDataLab's OmniDocBench. It converts tables to HTML and calculates tree-edit distance on both structure (`TEDS-Str`) and content (`TEDS-Con`).

### 2. Semantic Metrics (LLM-as-a-Judge)
Modeled after modern evaluation pipelines (e.g., Marker), this runs an LLM to evaluate text, layout reading order, tables, and formulas.

## Installation

```bash
pip install -r requirements.txt
```

*(Note: The script automatically downloads necessary NLTK data like `punkt` and `wordnet` at runtime.)*

## Usage

The `evaluate.py` script requires two directories: `--gt-dir` (ground truth) and `--test-dir` (predictions). It will recursively scan and pair files by name.

```bash
python evaluate.py --gt-dir /path/to/gt_md --test-dir /path/to/pred_md --output-dir ./results --mode [MODE]
```

### Modes

1. **`traditional`** (Default): Fast, zero API cost, purely algorithmic.
   ```bash
   python evaluate.py --gt-dir ../OmniDocBench/compare/gt_md --test-dir ../OmniDocBench/compare/pred_md --output-dir ./results --mode traditional
   ```

2. **`llm`**: Runs the LLM-as-a-judge semantic evaluation.
   ```bash
   python evaluate.py --gt-dir ../OmniDocBench/compare/gt_md --test-dir ../OmniDocBench/compare/pred_md --output-dir ./results --mode llm --llm-provider gemini
   ```

3. **`all`**: Runs both traditional and LLM metrics.

### LLM Options
* `--llm-provider`: `gemini` (default), `openai`, or `ollama`.
* `--llm-model`: The model to use (e.g. `gemini-1.5-flash`, `gpt-4o`).
* `--llm-api-key`: Your API key. Defaults to `GEMINI_API_KEY` in environment.
* `--llm-base-url`: Optional base URL (e.g., for local Ollama `http://localhost:11434/v1`).

## Outputs

The tool generates two files in the `--output-dir`:
* `eval_report.json`: Granular, document-by-document metrics.
* `eval_report.md`: A beautiful Markdown summary table.
