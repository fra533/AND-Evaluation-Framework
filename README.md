# **A Precision-Driven Evaluation Framework for Author Name Disambiguation**

This is an open-source Python implementation of the evaluation framework presented in *"A precision-driven evaluation protocol for Author Name Disambiguation"* (Cappelli, Colavizza, Peroni). It combines standard clustering metrics (Pairwise-F, B³) with structural error decomposition (Lumping Error, Splitting Error) to provide a comprehensive, interpretable assessment of Author Name Disambiguation (AND) systems.

## Why This Framework?

Standard clustering metrics (Pairwise-F, B³) evaluate aggregate quality but **don't distinguish between two qualitatively different failure modes**:

- **Lumping**: Merging publications from *different* authors into one cluster (false positive)
  - Direct contamination of author profiles
  - Hard to correct downstream
  - High cost in bibliographic applications

- **Splitting**: Fragmenting a *single* author's publications across multiple clusters (false negative)  
  - Incomplete author profiles
  - More recoverable downstream
  - Lower cost in practice

This framework makes this distinction **explicit through cluster-level metrics** and adopts a **precision-first stance**: cluster purity is prioritized over recall, reflecting real-world consequences of errors in bibliographic systems.

---

## Framework Overview

The framework evaluates AND systems through **four integrated components**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT                                        │
│  Predicted clusters (from AND system) vs. Ground-truth clusters │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  1. Pairwise Metrics    (link-based, quadratic sensitivity)    │
│  2. B³ Metrics           (instance-level, size-robust)          │
│  3. Structural Errors    (cluster-level, interpretable)         │
│     • Lumping Error (LE) → cluster purity                       │
│     • Splitting Error (SE) → author completeness               │
│  4. Composite Score      (0.3 PW + 0.5 B³ + 0.2 Struct)        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Comprehensive metrics + model selection guidance       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
git clone <repository-url>
cd evaluate_AND
pip install -r requirements.txt
```

**Requirements**:
- Python ≥ 3.9
- pandas ≥ 1.3.0
- openpyxl ≥ 3.6.0 (optional, for Excel export)

---

## Input Format

Both predictions and ground-truth use the same JSON structure:

```json
{
  "author_name_1": [
    ["paper_id_1", "paper_id_2"],
    ["paper_id_3"],
    ["paper_id_4", "paper_id_5"]
  ],
  "author_name_2": [
    ["paper_id_6", "paper_id_7", "paper_id_8"]
  ]
}
```

- **Top level**: Author name (string)
- **Value**: List of clusters (each cluster is a list of paper IDs)

**Example** (John Smith disambiguated into 3 clusters):
```json
{
  "John Smith": [
    ["arxiv_001", "scholar_002"],    # Cluster 1: John Smith (physicist)
    ["arxiv_003"],                    # Cluster 2: John Smith (mathematician, singleton)
    ["journal_004", "journal_005"]    # Cluster 3: John Smith (biologist)
  ]
}
```

> **For partial datasets** (e.g., OpenCitations): The evaluator automatically filters ground-truth to only papers present in predictions, avoiding unfair Recall penalties for missing data.

---

## Usage

### Basic Evaluation

```python
from evaluate_AND import MultiMetricEvaluator, save_json

evaluator = MultiMetricEvaluator(
    predictions_file="predictions.json",
    ground_truth_file="ground_truth.json",
    verbose=True  # Print summary after evaluation
)

results = evaluator.evaluate()
save_json(results, "results.json")
```

**Output** (printed):
```
======================================================================
📊 AUTHOR NAME DISAMBIGUATION - MULTI-METRIC EVALUATION
======================================================================

[DATASET STATISTICS]
   Total names:       1500
   Total instances:   12345
   Perfect matches:   1289

[PAIRWISE-F (Link-based)]
   Precision: 0.8523
   Recall:    0.7891
   F1:        0.8187

[B³ (Bagga & Baldwin, 2009)]
   Precision: 0.8821
   Recall:    0.8145
   F1:        0.8468

[STRUCTURAL ERRORS]
   Lumping Error:    0.0452
   Splitting Error:  0.0876
   Struct Score:     0.9336

[COMPOSITE SCORE (0.3 PW + 0.5 B³ + 0.2 Struct)]
   0.8512
======================================================================
```

### Export to Excel

```python
import argparse

# Simple export
evaluator.save_to_excel(base_output_path="reports", experiment_name="Exp1")

# With experiment metadata
args = argparse.Namespace(
    emb_type="SBERT",
    save_path="./datasets/OC",
    db_eps=0.5,
    db_min=2,
    use_citations=True
)

report_path = evaluator.save_to_excel(
    base_output_path="reports",
    experiment_name="SBERT_DBSCAN_v1",
    args=args
)
```

File is named: `{EMBEDDING}_{DATASET}_{EXPERIMENT}_{TIMESTAMP}.xlsx`

---

## Metrics Explained

### **Pairwise-F** — Link-Based Consistency

Counts publication pairs: are they co-clustered correctly?
- **Precision**: Fraction of predicted links (co-clustered pairs) that are true
- **Recall**: Fraction of true links that were predicted
- **Issue**: Quadratically sensitive to cluster size (large author lists dominate)

### **B³ (Bagga & Baldwin, 1998)** — Instance-Level Quality

For each publication, measures overlap between predicted and true cluster:
- **Precision**: Average purity of each paper's predicted cluster
- **Recall**: Average completeness of each paper's true author cluster
- **Advantage**: Robust to cluster size imbalance
- **Most reliable aggregate metric in AND**

### **Lumping Error (LE)** — Cluster Purity

$$\text{LE} = \frac{1}{|C|} \sum_{C \in C'} \frac{\max(0, g(C) - 1)}{g(C)}$$

- **Range**: [0, 1]
- **Meaning**: Fraction of ground-truth authors incorrectly mixed within each predicted cluster
- **0 = Perfect**: Every predicted cluster contains publications from only one real author
- **1 = Worst**: Every predicted cluster mixes many different authors

**Interpretation**:
- LE = 0.05 → Only 5% of predicted clusters are contaminated with wrong authors
- LE = 0.20 → Significant purity issue; clusters mix real authors too much

### **Splitting Error (SE)** — Author Completeness

$$\text{SE} = \frac{1}{|T|} \sum_{T \in T'} \frac{\max(0, p(T) - 1)}{p(T)}$$

- **Range**: [0, 1]
- **Meaning**: Fraction of predicted clusters needed to cover each true author
- **0 = Perfect**: Every real author is contained in exactly one cluster
- **1 = Worst**: Every real author is fragmented across many clusters

**Interpretation**:
- SE = 0.10 → Each author's publications are minimally fragmented
- SE = 0.40 → Many authors split across multiple clusters

---

## Results Dictionary

```python
results = evaluator.evaluate()

# Structure:
{
  "pairwise": {
    "precision": 0.8523,
    "recall": 0.7891,
    "f1": 0.8187
  },
  "b3": {
    "precision": 0.8821,  # ← Most important for AND
    "recall": 0.8145,
    "f1": 0.8468
  },
  "structural": {
    "lumping_error": 0.0452,   # ← Watch this closely
    "splitting_error": 0.0876,
    "score": 0.9336
  },
  "composite_score": 0.8512  # ← For model ranking
}
```

---

## Model Selection Strategy

**⚠️ Do NOT use Composite Score alone for model selection.**

Follow this **three-step procedure** (precision-first stance):

```
Step 1: Rank by Pairwise F1
    └─ Filter candidates with F1 < threshold

Step 2: Among remaining, rank by B³ Precision
    └─ Prioritize cluster purity (lumping avoidance)
    └─ Filter candidates with B³-P < threshold

Step 3: Among remaining, inspect Lumping Error (LE)
    └─ Final tiebreaker: pick lowest LE
    └─ Only then use Composite Score if needed
```

**Why?** The Composite Score uses symmetric weighting of LE and SE (both count equally). But in bibliographic applications, **lumping (false positives) is costlier than splitting (false negatives)**. Step 3 explicitly prioritizes purity.

---

## Interpreting Composite Score

| Score | Interpretation | Action |
|-------|---|---|
| **0.90–1.00** | Excellent | Ready for production |
| **0.85–0.90** | Very good | Accept with minor review |
| **0.80–0.85** | Good | Acceptable but optimize |
| **0.70–0.80** | Fair | Needs adjustment |
| **< 0.70** | Poor | Redesign required |

**But remember**: A small gap in Composite Score can hide a significant difference in LE. Always check Step 1–3 above.

---

## Dynamic Alignment (for Partial Datasets)

When evaluating AND systems on partial ground-truth (e.g., OpenCitations, where only indexed papers have annotations):

```python
# Internally, the framework:
# 1. Collects all papers present in predictions
# 2. Filters ground-truth to ONLY those papers
# 3. Evaluates on aligned subset
# 4. Result: Recall not penalized for unavailable data
```

This is **automatic**—just pass your files and it works correctly.

---

## Examples

### Example 1: Basic Evaluation

```bash
python example_usage.py --example 1
```

### Example 2: With Metadata + Excel Export

```bash
python example_usage.py --example 2
```

Generates Excel report with experiment parameters.

### Example 3: Batch Comparison (Multiple Experiments)

```bash
python example_usage.py --example 3
```

Runs multiple configs, prints comparison table, exports all to Excel.

### Example 4: Detailed Analysis

```bash
python example_usage.py --example 4
```

Full results with detailed breakdown.

---

## Files

- **evaluate_AND.py** – Main evaluator (production code)
- **example_usage.py** – 4 complete usage scenarios
- **example_predictions.json** – Test data (predictions)
- **example_ground_truth.json** – Test data (ground truth)
- **requirements.txt** – Dependencies
- **LICENSE** – MIT license
- **.gitignore** – Standard Python ignores
- **README.md** – This file

---

## Known Limitations

1. **Composite Score weighting**: The 0.3/0.5/0.2 weights reflect a precision-first stance. If you prioritize completeness equally, adjust γ parameter and re-inspect LE directly.

2. **Macro vs. Micro averaging**: Results use macro-averaging (average error per name). In datasets with heavy-tailed name-ambiguity distributions, a few very-ambiguous names may dominate micro-averaged scores. For detailed analysis, inspect per-name breakdowns.

3. **Large cluster bias in Pairwise-F**: Pairwise metrics still over-weight large author clusters (quadratic sensitivity). Trust B³ and LE/SE for more reliable signals.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{cappelliEvalFram,
  title={A Precision-Driven Evaluation Framework for Author Name Disambiguation},
  author={Cappelli, Francesca; Colavizza, Giovanni; Peroni, Silvio},
  journal={},
  year={}
}
```

---

## References

```bibtex
@article{bagga1998entity,
  title={Entity-based cross-document coreferencing using the Vector Space Model.},
  author={Bagga, A.; Baldwin, B.},
  journal={Proc. 36th Annual Meeting of the ACL and 17th COLING},
  year={998}
  url= {https://doi.org/10.3115/980845.980859}
}



---

## License


## Contact

**Francesca Cappelli**  
Department of Classical Philology and Italian Studies, University of Bologna  
[Email / Affiliation]

