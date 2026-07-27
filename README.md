# Novig Support Ticket Triage

## Project Overview

This project is an LLM-based support ticket triage system for the Novig take-home assignment. It classifies incoming support tickets, determines urgency, decides whether each ticket is safe for automatic draft generation, and generates draft responses when appropriate.

The repository also includes evaluation tools for:

- the labeled training set, where predictions are compared against ground-truth labels, and
- the unlabeled evaluation set, where two model configurations are compared using agreement metrics.

## Repository Structure

```text
.
├── eval.py
├── requirements.txt
├── README.md
├── .env
├── novig_files/
├── prompts/
├── results/
└── src/
```

Key files and directories:

- `src/` - Main source code for the triage pipeline.
- `src/config.py` - Named model configurations. Each config defines the classification and drafting models for a run.
- `src/classify_tickets.py` - Runs the full pipeline on a JSONL dataset: classification, optional draft generation, prediction writing, and labeled evaluation when labels are available.
- `src/draft.py` - Draft-response generation logic. It loads `prompts/draft.md` and uses the shared LLM interface.
- `src/evaluate_predictions.py` - Evaluation framework. Supports labeled accuracy evaluation and unlabeled agreement evaluation.
- `src/llm.py` - Provider-agnostic LLM interface. This is the only module that calls OpenAI or Anthropic SDKs directly.
- `prompts/` - Prompt files loaded at runtime.
- `prompts/classification.md` - Classification system prompt.
- `prompts/draft.md` - Draft-response system prompt.
- `novig_files/` - Assignment data files, including `tickets_train.jsonl`, `tickets_eval.jsonl`, and taxonomy documentation.
- `results/` - Timestamped run outputs, including predictions, reports, and config snapshots.
- `eval.py` - Unlabeled evaluation-set runner. Runs two configs and reports agreement metrics.
- `requirements.txt` - Python dependencies.
- `.env` - Local environment variables for API keys. This file should not be committed.

## Installation

From a fresh clone:

```bash
git clone <repo-url>
cd novig_takehome
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```bash
touch .env
```

Add the API keys for the providers you plan to use:

```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

You only need the key for the provider used by the selected config. For example, an OpenAI-only config requires `OPENAI_API_KEY`; a Claude config requires `ANTHROPIC_API_KEY`.

## Configuration

Model configurations are defined in `src/config.py`:

```python
CONFIGS = {
    "claude-sonnet": {
        "classification_model": "claude-sonnet-5",
        "draft_model": "claude-sonnet-5",
    },
    "gpt-4.1-mini": {
        "classification_model": "gpt-4.1-mini",
        "draft_model": "gpt-4.1-mini",
    },
}
```

Each config controls the model used for:

- `classification_model` - the model used to classify the ticket and decide whether drafting is allowed.
- `draft_model` - the model used to generate a draft response when `should_draft` is `true`.

Select a config with `--config`:

```bash
python3 src/classify_tickets.py novig_files/tickets_train.jsonl --config gpt-4.1-mini
```

If `--config` is omitted, the first config listed in `CONFIGS` is used as the default. If an unknown config name is provided, the command exits with a clear error listing the available configs.

The LLM provider is inferred from the configured model name in `src/llm.py`. OpenAI models such as `gpt-*` use the OpenAI SDK; Anthropic models such as `claude-*` use the Anthropic SDK.

## Running the Pipeline

Generate predictions on the labeled training set:

```bash
python3 src/classify_tickets.py novig_files/tickets_train.jsonl --config gpt-4.1-mini
```

This writes predictions and a labeled evaluation report into a timestamped folder under `results/`.

Evaluate an existing prediction file against the labeled training set:

```bash
python3 src/evaluate_predictions.py novig_files/tickets_train.jsonl results/<run_dir>/category_output.jsonl
```

For detailed mismatch inspection:

```bash
python3 src/evaluate_predictions.py --verbose novig_files/tickets_train.jsonl results/<run_dir>/category_output.jsonl
```

Generate official predictions and agreement metrics on the unlabeled evaluation set:

```bash
python3 eval.py
```

By default, `eval.py` runs the first two configs from `CONFIGS`. To choose the configs explicitly:

```bash
python3 eval.py --first-config claude-sonnet --second-config gpt-4.1-mini
```

To run on a different input file:

```bash
python3 eval.py --input-path novig_files/tickets_eval.jsonl
```

## Results

All pipeline runs create timestamped folders under `results/`.

For `src/classify_tickets.py`, the folder is named with the timestamp and selected config:

```text
results/20260727_154233_gpt-4_1-mini/
```

Expected files:

- `category_output.jsonl` - Final prediction objects, one per input ticket.
- `eval_output.txt` - Labeled evaluation report when the input file contains labels.
- `config.py` - Copy of the exact config file used for the run.

For `eval.py`, the folder includes both config names:

```text
results/20260727_154233_claude-sonnet_vs_gpt-4_1-mini/
```

Expected files:

- `predictions.jsonl` - Official assignment predictions from the first config.
- `comparison_predictions_<config>.jsonl` - Predictions from the second config.
- `agreement_report.txt` - Agreement metrics between the two configs.
- `config.py` - Copy of the exact config file used for the run.

Config names are sanitized for filesystem safety, so dots may appear as underscores in result folder or file names.

## Project Architecture

The pipeline has four main stages:

1. Ticket classification

   `src/classify_tickets.py` loads `prompts/classification.md`, sends the ticket JSON to the configured classification model, parses the JSON response, and validates required fields.

2. Auto-draft decision

   The classification result includes `should_draft`. If it is `false`, the pipeline leaves `draft_response` as `null` and writes the final prediction.

3. Draft generation

   If `should_draft` is `true`, `src/draft.py` loads `prompts/draft.md` and sends both the original ticket JSON and classification result to the configured draft model. The returned text is inserted into `draft_response`.

4. Evaluation

   `src/evaluate_predictions.py` supports two modes:

   - Labeled mode compares predictions to `tickets_train.jsonl` labels and reports accuracy-style metrics.
   - Agreement mode compares two prediction files from different configs and reports agreement metrics without treating either model as ground truth.

All model calls go through `src/llm.py`, keeping the rest of the pipeline provider-agnostic.
