# MAGI — Meeting Argumentation & Generation Intelligence

A research system for LLM-based meeting discussion generation and evaluation using the AMI and ICSI corpora.

---

## Repository Structure

```
code/
├── LICENSE                  # CC BY 4.0 (code) + third-party dataset notices
├── Readme.md                # This file
├── requirements.txt         # Python dependencies
├── sampler.py               # Evaluation runner (inference + scoring)
├── stats.py                 # Aggregate mean ± SD across run folders
├── sum.py                   # Cross-condition comparison + Wilcoxon test
├── data/                    # Dataset preparation scripts (see data/README.md)
│   ├── add_theme.py
│   ├── prepare_icsi.py / prepare_helper_icsi.py
│   ├── prepare_ami.py  / prepare_helper_ami.py
│   └── dataset/             # Output JSON / PKL files
└── results/                 # output examples per condition
    ├── ami_single_4/        # AMI, single-agent (gpt-4.1)
    ├── ami_single_5/        # AMI, single-agent (gpt-5.1-chat) 
    ├── ami_multi_noaf/      # AMI, multi-agent, no argument framework (gpt-4.1)
    ├── ami_multi_af/        # AMI, multi-agent, with argument framework (gpt-4.1)
    ├── icsi_single_4/　　　　# ICSI, single-agent (gpt-4.1)
    ├── icsi_single_5/　　　　# ICSI, single-agent (gpt-5.1-chat) 
    ├── icsi_multi_noaf/     # ICSI, multi-agent, no argument framework (gpt-4.1)
    └── icsi_multi_af/       # ICSI, multi-agent, with argument framework (gpt-4.1)
```

> **Note:** Agent, model, evaluation module, and utility code is not included in this repository.  
> The scripts above depend on internal modules (`agent/`, `models/`, `evaluation/`, `utils/`, `const/`, `run_flow.py`, `run_single.py`) that must be provided separately.

---

## Experimental Conditions

| Condition key | Mode | Argument Framework | Model |
|---|---|---|---|
| `single_4` | Single-agent | No | gpt-4.1 |
| `single_5` | Single-agent | No | gpt-5.1-chat |
| `multi_noaf` | Multi-agent | No | gpt-4.1 |
| `multi_af` | Multi-agent | Yes | gpt-4.1 |

---

## File Descriptions

### `sampler.py`

Runs inference on meeting records and saves evaluation scores.

- Loads meeting data from `data/dataset/icsi.pkl` (or `extracted_ami.pkl`).
- Runs each meeting through either a **single-agent** or **multi-agent** flow.
- Scores outputs with `Sampler` (BERTScore, ROUGE, etc.) and/or `Sampler_RAGAS`.
- Saves per-meeting JSONL files under `results/{condition}/{run}/`.

#### Output filename format

```
{meeting_id}_{YYYYMMDD_HHMMSS}.jsonl          # raw inference output
```

#### Score sections

| Section | Description |
|---|---|
| `score_in_out` | Similarity between input (theme) and output |
| `score_ref_out` | Similarity between reference summary and output |
| `score_arg_div` | Argument diversity score |
| `score_opp` | Opposition/counter-argument score |
| `score_faith` | Faithfulness score |
| `score_ragas_in_out` | RAGAS score (input → output) |
| `score_ragas_ref_out` | RAGAS score (reference → output) |

---

### `stats.py`

Aggregates per-meeting scores across numbered run sub-folders (`1/`, `2/`, `3/`, …) and saves mean ± SD to `stats.json`.

#### Usage

```bash
# Default: results/ami_single_4/ → results/ami_single_4/stats.json
python stats.py

# Specify condition folder
python stats.py -f ami_multi_af
```

#### Output (`stats.json`) format

```json
{
  "ES2002a": {
    "n_runs": 3,
    "metrics": {
      "score_in_out.bertscore.f1": {"mean": 0.812, "sd": 0.005, "n": 3}
    }
  }
}
```

---

### `sum.py`

Compares evaluation results across two or more conditions and performs Wilcoxon signed-rank tests.

- Can load raw eval JSONL files or pre-computed `stats.json`.
- Prints per-metric Wilcoxon test results (p-value, effect size) between conditions.

#### Configuration (edit inside `sum.py`)

| Variable | Description |
|---|---|
| `folders` | List of condition folders to compare (e.g. `["ami_multi_noaf", "ami_multi_af"]`) |
| `single_folder` | Reference condition for Wilcoxon comparison |
| `RAGAS_eval` | `True` to use RAGAS scores, `False` for standard scores |
| `use_stats` | `True` to load from `stats.json`, `False` for raw eval files |

```bash
python sum.py
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> For PyTorch, install separately according to your CUDA version:
> ```bash
> # CUDA 12.4
> pip install torch --index-url https://download.pytorch.org/whl/cu124
> ```

### 2. Prepare datasets

See [data/README.md](data/README.md) for instructions on building `icsi.pkl` and `extracted_ami.pkl`.

### 3. Configure environment

Create a `.env` file (or set environment variables) with your Azure OpenAI credentials:

```
OPENAI_API_VERSION="2024-02-01"
CHAT_AZURE_OPENAI_API_KEY="<your-api-key>"
CHAT_AZURE_OPENAI_ENDPOINT="<your-endpoint>"
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="<deployment-name>"
```

---

## License

* This code: [CC BY-NC-ND 4.0](LICENSE) — Copyright (c) 2026 Mitsubishi Electric Corporation
* [LangGraph](https://github.com/langchain-ai/langgraph/blob/main/LICENSE): MIT License
* AMI Meeting Corpus: CC BY 4.0 (see [data/README.md](data/README.md))
* ICSI Meeting Corpus: LDC User Agreement (see [data/README.md](data/README.md))

---

## References

* Carletta, J. et al. (2005). The AMI Meeting Corpus: A Pre-announcement. *MLMI 2005*, LNCS 3869.
* Janin, A. et al. (2003). The ICSI Meeting Corpus. *ICASSP 2003*.
* [LangGraph](https://github.com/langchain-ai/langgraph)
