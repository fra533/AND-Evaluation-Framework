## \#AND Evaluation Framework



Multi-metric evaluation framework for \*\*Author Name Disambiguation (AND)\*\* systems, implementing clustering metrics as per Kim et al. (2019).



Designed for BOND and compatible AND systems. Supports dynamic alignment of predictions with ground-truth, making it suitable for partial datasets (e.g., OpenCitations where ground-truth may be incomplete).



\## Features



\- \*\*Pairwise-F\*\*: Link-based precision/recall for clustering

\- \*\*B³ (Bagga \& Baldwin, 2009)\*\*: Instance-level clustering metrics

\- \*\*Structural Errors\*\*: Lumping Error (false positives) and Splitting Error (false negatives)

\- \*\*K-metric\*\*: Asymmetric penalties for errors (Kim et al., 2019)

\- \*\*Composite Score\*\*: Weighted average combining multiple metrics

\- \*\*Dynamic Alignment\*\*: Filters ground-truth to only papers present in predictions, avoiding unfair Recall penalties for missing data

\- \*\*Excel Reports\*\*: Automated export of results with experiment metadata

\- \*\*Logging\*\*: Comprehensive logging for debugging and auditing



\## Installation



```bash

pip install pandas

```



\## Data Format



\### Input JSON



Both prediction and ground-truth files use the same format:



```json

{

&#x20; "author\_name\_1": \[

&#x20;   \["paper\_id\_1", "paper\_id\_2"],

&#x20;   \["paper\_id\_3"],

&#x20;   \["paper\_id\_4", "paper\_id\_5", "paper\_id\_6"]

&#x20; ],

&#x20; "author\_name\_2": \[

&#x20;   \["paper\_id\_7", "paper\_id\_8"],

&#x20;   \["paper\_id\_9"]

&#x20; ]

}

```



\- \*\*Top level\*\*: Author name → list of predicted clusters

\- \*\*Cluster\*\*: List of paper IDs that belong together (represent one author)



\### Example



```json

{

&#x20; "John Smith": \[

&#x20;   \["arxiv\_001", "scholar\_002"],

&#x20;   \["arxiv\_003"],

&#x20;   \["journal\_004", "journal\_005"]

&#x20; ],

&#x20; "Jane Doe": \[

&#x20;   \["pub\_001", "pub\_002", "pub\_003"]

&#x20; ]

}

```



\## Usage



\### Basic Evaluation



```python

from evaluate\_BOND import MultiMetricEvaluator, save\_json



\# Initialize evaluator

evaluator = MultiMetricEvaluator(

&#x20;   predictions\_file="predictions.json",

&#x20;   ground\_truth\_file="ground\_truth.json",

&#x20;   verbose=True

)



\# Run evaluation

results = evaluator.evaluate()



\# Save results

save\_json(results, "results.json")

```



\### Export to Excel



```python

\# Simple export

evaluator.save\_to\_excel(

&#x20;   base\_output\_path="reports",

&#x20;   experiment\_name="Exp1"

)



\# With experiment metadata

import argparse



args = argparse.Namespace(

&#x20;   emb\_type="SBERT",

&#x20;   save\_path="./datasets/OC",

&#x20;   db\_eps=0.5,

&#x20;   db\_min=2,

&#x20;   use\_citations=True,

&#x20;   rel\_on="cosine"

)



evaluator.save\_to\_excel(

&#x20;   base\_output\_path="reports",

&#x20;   experiment\_name="Exp1",

&#x20;   args=args

)

```



File is named: `{EMBEDDING}\_{DATASET}\_{EXPERIMENT}\_{TIMESTAMP}.xlsx`



\### Command-line Usage



```bash

python evaluate\_BOND.py

```



Edit the `if \_\_name\_\_ == "\_\_main\_\_"` block to customize input/output paths.



\## Output Metrics



\### Pairwise-F

Link-based clustering metrics:

\- \*\*Precision\*\*: Fraction of predicted links that are correct

\- \*\*Recall\*\*: Fraction of true links that were predicted

\- \*\*F1\*\*: Harmonic mean



\### B³ (Bagga \& Baldwin, 2009)

Instance-level metrics treating each paper as an evaluation unit:

\- \*\*Precision\*\*: Average overlap between predicted and true cluster for each paper

\- \*\*Recall\*\*: Average overlap from true cluster perspective

\- \*\*F1\*\*: Harmonic mean



\### Structural Errors



\*\*Lumping Error\*\* (LE): False positives in clustering

\- Range: \[0, 1]

\- \*\*0\*\*: Every predicted cluster is pure (contains only one true author)

\- \*\*1\*\*: Every predicted cluster mixes many authors



$$\\text{LE} = \\frac{1}{|C|} \\sum\_{C \\in C'} \\frac{\\max(0, g(C) - 1)}{g(C)}$$



where $g(C)$ = number of distinct ground-truth authors in predicted cluster $C$.



\*\*Splitting Error\*\* (SE): False negatives in clustering

\- Range: \[0, 1]

\- \*\*0\*\*: Every true author appears in exactly one cluster

\- \*\*1\*\*: Every true author is split across many clusters



$$\\text{SE} = \\frac{1}{|T|} \\sum\_{T \\in T'} \\frac{\\max(0, p(T) - 1)}{p(T)}$$



where $p(T)$ = number of predicted clusters intersecting true author $T$.



\### K-metric (Kim et al., 2019)



Combines B³ scores with asymmetric penalties:

\- \*\*AAP\*\* (Asymmetric Artifact Penalty): $1 - B³\\text{-Precision}$

\- \*\*ACP\*\* (Asymmetric Cluster Penalty): $1 - B³\\text{-Recall}$

\- \*\*K\*\*: $\\text{AAP} \\times \\text{ACP}$



Higher K is better; K ∈ \[0, 1].



\### Composite Score



Weighted average combining all metrics:



$$S\_{\\text{composite}} = 0.3 \\times \\text{PW-F1} + 0.5 \\times \\text{B³-F1} + 0.2 \\times S\_{\\text{struct}}$$



where $S\_{\\text{struct}} = 1 - \\frac{\\text{LE} + \\text{SE}}{2}$ (structural score).



\## Results Dictionary



```python

{

&#x20; "pairwise": {

&#x20;   "precision": 0.85,

&#x20;   "recall": 0.80,

&#x20;   "f1": 0.8235

&#x20; },

&#x20; "b3": {

&#x20;   "precision": 0.88,

&#x20;   "recall": 0.82,

&#x20;   "f1": 0.85

&#x20; },

&#x20; "k\_metric": {

&#x20;   "aap": 0.12,

&#x20;   "acp": 0.18,

&#x20;   "k": 0.9784

&#x20; },

&#x20; "structural": {

&#x20;   "lumping\_error": 0.05,

&#x20;   "splitting\_error": 0.10,

&#x20;   "score": 0.925

&#x20; },

&#x20; "composite\_score": 0.8461

}

```



\## Algorithm: Dynamic Alignment



For partial datasets (e.g., OpenCitations, where ground-truth is only available for indexed papers):



1\. \*\*Collect available papers\*\*: All papers in predicted clusters

2\. \*\*Filter ground-truth\*\*: Remove papers not in predictions

3\. \*\*Align clusters\*\*: Evaluate only on the common subset

4\. \*\*Benefit\*\*: Recall is not penalized for papers with no ground-truth label



This prevents unfair metric scores when evaluating against incomplete ground-truth.



\## Interpreting Results



| Score | Interpretation |

|-------|---|

| \*\*0.9–1.0\*\* | Excellent disambiguation; very few errors |

| \*\*0.8–0.9\*\* | Good; acceptable for most applications |

| \*\*0.7–0.8\*\* | Fair; noticeable errors but usable |

| \*\*<0.7\*\* | Poor; significant lumping/splitting errors |



\*\*Example:\*\*

\- High Lumping Error (LE > 0.3) → predicted clusters mix too many authors → reduce clustering threshold (merge fewer papers)

\- High Splitting Error (SE > 0.3) → true authors too fragmented → increase clustering threshold (merge more papers)



\## References



\- Amigó, E., Gonzalo, J., Artiles, J., \& Verdejo, F. (2009).

&#x20; \*A comparison of extrinsic clustering evaluation metrics and an internal evaluation measure.\*

&#x20; Information Processing \& Management, 45(4), 422-430.



\- Bagga, A., \& Baldwin, B. (1998).

&#x20; \*Entity-based cross-document coreferencing using the vector space model.\*

&#x20; In Proceedings of the 36th Annual Meeting of the Association for Computational Linguistics.



\- Kim, K., et al. (2019).

&#x20; \*A large-scale author name disambiguation dataset with ground-truth annotation.\*

&#x20; arXiv preprint arXiv:1904.12122.



\## Example: Full Pipeline



```python

from evaluate\_BOND import MultiMetricEvaluator, save\_json, load\_json

import argparse



\# Load your AND system's predictions

predictions = load\_json("path/to/predictions.json")

ground\_truth = load\_json("path/to/ground\_truth.json")



\# Create evaluator

evaluator = MultiMetricEvaluator(

&#x20;   predictions\_file="predictions.json",

&#x20;   ground\_truth\_file="ground\_truth.json",

&#x20;   verbose=True

)



\# Run evaluation

results = evaluator.evaluate()



\# Print detailed results

evaluator.print\_results(detailed=True)



\# Save to JSON

save\_json(results, "evaluation\_results.json")



\# Export to Excel

args = argparse.Namespace(

&#x20;   emb\_type="SBERT",

&#x20;   save\_path="./oc\_dataset",

&#x20;   db\_eps=0.5,

&#x20;   db\_min=2

)

report\_path = evaluator.save\_to\_excel(

&#x20;   base\_output\_path="reports",

&#x20;   experiment\_name="SBERT\_DBSCAN",

&#x20;   args=args

)

print(f"Results saved to: {report\_path}")

```



\## Files



\- `evaluate\_BOND.py` – Main evaluator class

\- `README.md` – This file

\- `requirements.txt` – Python dependencies



\## License



MIT



\## Citation



If you use this evaluator in your research, please cite:



```bibtex

@software{evaluate\_BOND,

&#x20; author = {Cappelli, Francesca and Peroni, Silvio and Colavizza, Giovanni},

&#x20; title = {Author Name Disambiguation Evaluator for BOND},

&#x20; url = {https://github.com/your-repo/evaluate\_BOND},

&#x20; year = {2024}

}

```



\## Issues \& Contributions



For bug reports or feature requests, open an issue on the repository.



Pull requests are welcome; please include tests and update documentation.



\---



\*\*Author\*\*: Francesca Cappelli  

\*\*Affiliation\*\*: University of Bologna, Department of Classical Philology and Italian Studies  

\*\*Project\*\*: Author Name Disambiguation for OpenCitations (GraspOS EU Horizon Europe)

