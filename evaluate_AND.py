"""
Author Name Disambiguation Evaluator

Multi-metric evaluation framework implementing clustering metrics (Pairwise-F, B³)
and structural error measures (Lumping Error, Splitting Error).

Supports dynamic alignment of predictions with ground-truth for partial datasets
(e.g., when ground-truth is available only for papers present in OpenCitations).


"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Set

import pandas as pd


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Load JSON data from file.

    Args:
        path: File path to JSON file.

    Returns:
        Parsed JSON data as dictionary.

    Raises:
        FileNotFoundError: If file does not exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        raise


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved JSON to {path}")


class MultiMetricEvaluator:
    """
    Multi-metric evaluator for Author Name Disambiguation systems.

    Implements:
    - Pairwise-F (link-based precision/recall)
    - B³ (Bagga & Baldwin, instance-level clustering metrics)
    - Structural errors (Lumping Error, Splitting Error)
    - Composite score 

    Dynamic alignment: predictions and ground-truth are automatically aligned
    to only include papers present in both datasets. This avoids penalizing Recall
    for unavailable data (important for partial datasets like OpenCitations).

    Attributes:
        predictions_file: Path to predictions JSON file.
        ground_truth_file: Path to ground-truth JSON file.
        verbose: If True, print results after evaluation.
        results: Dictionary containing computed evaluation metrics.
        stats: Dataset-level statistics.
    """

    def __init__(
        self,
        predictions_file: str | Path,
        ground_truth_file: str | Path,
        verbose: bool = True
    ) -> None:
        """
        Initialize evaluator.

        Args:
            predictions_file: Path to JSON file with predicted clusters.
                Format: {name: [[paper_id, ...], ...]}
            ground_truth_file: Path to JSON file with ground-truth clusters.
                Format: {name: [[paper_id, ...], ...]}
            verbose: Print results summary after evaluation.
        """
        self.predictions_file = Path(predictions_file)
        self.ground_truth_file = Path(ground_truth_file)
        self.verbose = verbose
        self.results: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}

    def evaluate(self) -> Dict[str, Any]:
        """
        Execute multi-metric evaluation.

        Process:
        1. Load predictions and ground-truth from JSON files.
        2. Find common author names in both datasets.
        3. For each name, align clusters (filter GT to only include papers in predictions).
        4. Compute metrics: pairwise, B³, structural errors.
        5. Aggregate globally and compute composite score.

        Returns:
            Dictionary with keys:
            - "pairwise": {precision, recall, f1}
            - "b3": {precision, recall, f1}
            - "structural": {lumping_error, splitting_error, score}
            - "composite_score": Weighted average (0.3 PW + 0.5 B³ + 0.2 Struct)
        """
        logger.info("Starting evaluation...")

        try:
            pred = load_json(self.predictions_file)
            truth = load_json(self.ground_truth_file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load input files: {e}")
            return self._empty()

        global_stats = {
            "tp": 0, "fp": 0, "fn": 0,
            "b3p": 0.0, "b3r": 0.0,
            "le": 0.0, "se": 0.0,
            "n": 0,
            "num_names": 0
        }

        # Find common author names
        common_names = [k for k in pred if k in truth]

        if not common_names:
            logger.warning("No common names found between predictions and ground-truth.")
            self.results = self._empty()
            self.stats = {
                "total_names": 0,
                "total_instances": 0,
                "perfect_matches": 0,
                "accuracy_names": 0.0
            }
            return self.results

        perfect_matches = 0

        for name in common_names:
            # Collect all papers present in predictions (ground-truth is filtered to these)
            all_pred_papers: Set[str] = {
                x for cluster in pred[name] for x in cluster
            }

            # Filter ground-truth to only include papers present in predictions
            raw_truth_clusters = self._prepare_truth(truth[name])
            filtered_truth: List[List[str]] = []
            for t_cluster in raw_truth_clusters:
                new_cluster = [p for p in t_cluster if p in all_pred_papers]
                if new_cluster:
                    filtered_truth.append(new_cluster)

            # Evaluate this name
            stats = self._evaluate_one(pred[name], filtered_truth)
            self._accumulate(global_stats, stats)

            # Count perfect matches (no lumping, no splitting)
            if stats["n"] > 0 and stats["le"] == 0.0 and stats["se"] == 0.0:
                perfect_matches += 1

        total_names = len(common_names)
        self.stats = {
            "total_names": total_names,
            "total_instances": global_stats["n"],
            "perfect_matches": perfect_matches,
            "accuracy_names": perfect_matches / total_names if total_names else 0.0
        }

        self.results = self._finalize(global_stats)
        self.results["composite_score"] = self._composite(self.results)

        if self.verbose:
            self.print_results(detailed=False)

        logger.info("Evaluation completed successfully.")
        return self.results

    def _evaluate_one(
        self,
        pred_clusters: List[List[str]],
        truth_clusters: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Evaluate a single author name.

        Computes clustering metrics (pairwise, B³, structural errors) by comparing
        predicted clusters against ground-truth clusters.

        Args:
            pred_clusters: List of predicted clusters (each cluster is list of paper IDs).
            truth_clusters: List of ground-truth clusters.

        Returns:
            Dictionary with metric components for this name.
        """
        truth_clusters = self._prepare_truth(truth_clusters)

        if not pred_clusters or not truth_clusters:
            return self._empty_one()

        # Build indices: paper -> cluster index
        p_index: Dict[str, int] = {}  # prediction index
        t_index: Dict[str, int] = {}  # truth index

        for i, cluster in enumerate(pred_clusters):
            for paper in cluster:
                p_index[paper] = i

        for i, cluster in enumerate(truth_clusters):
            for paper in cluster:
                t_index[paper] = i

        # ===== Pairwise Metrics =====
        # Count (paper_i, paper_j) pairs; if linked in both, it's TP.
        tp = fp = fn = 0

        for cluster in pred_clusters:
            n = len(cluster)
            tp_local = sum(
                1 for i in range(n) for j in range(i + 1, n)
                if cluster[i] in t_index and cluster[j] in t_index
                and t_index[cluster[i]] == t_index[cluster[j]]
            )
            total = n * (n - 1) // 2
            tp += tp_local
            fp += total - tp_local

        for cluster in truth_clusters:
            n = len(cluster)
            tp_local = sum(
                1 for i in range(n) for j in range(i + 1, n)
                if cluster[i] in p_index and cluster[j] in p_index
                and p_index[cluster[i]] == p_index[cluster[j]]
            )
            total = n * (n - 1) // 2
            fn += total - tp_local

        # ===== B³ Metrics =====
        # Instance-level precision and recall (Bagga & Baldwin, 2009)
        b3p_sum = 0.0
        b3r_sum = 0.0

        all_instances = set(p_index.keys()) | set(t_index.keys())

        for paper in all_instances:
            if paper not in p_index or paper not in t_index:
                continue
            pc = set(pred_clusters[p_index[paper]])
            tc = set(truth_clusters[t_index[paper]])
            intersection = len(pc & tc)
            b3p_sum += intersection / len(pc)
            b3r_sum += intersection / len(tc)

        n_instances = len(all_instances)

        # ===== Structural Errors (Cluster-level) =====
        # Lumping Error: fraction of true authors mixed in each predicted cluster
        le_sum = 0.0
        for cluster in pred_clusters:
            gt_ids = {t_index[x] for x in cluster if x in t_index}
            g = len(gt_ids)
            if g > 0:
                le_sum += max(0, g - 1) / g
        le = le_sum / len(pred_clusters) if pred_clusters else 0.0

        # Splitting Error: fraction of predicted clusters for each true author
        se_sum = 0.0
        for cluster in truth_clusters:
            pred_ids = {p_index[x] for x in cluster if x in p_index}
            p = len(pred_ids)
            if p > 0:
                se_sum += max(0, p - 1) / p
        se = se_sum / len(truth_clusters) if truth_clusters else 0.0

        return {
            "tp": tp, "fp": fp, "fn": fn,
            "b3p": b3p_sum,
            "b3r": b3r_sum,
            "le": le,
            "se": se,
            "n": n_instances,
            "num_names": 1
        }

    def _finalize(self, g: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate global statistics and compute final metrics.

        Args:
            g: Global stats dictionary (accumulated across all names).

        Returns:
            Dictionary with final precision/recall/F1 for each metric family.
        """
        # Pairwise F1
        pw_p = g["tp"] / (g["tp"] + g["fp"]) if (g["tp"] + g["fp"]) > 0 else 0.0
        pw_r = g["tp"] / (g["tp"] + g["fn"]) if (g["tp"] + g["fn"]) > 0 else 0.0
        pw_f1 = 2 * pw_p * pw_r / (pw_p + pw_r) if (pw_p + pw_r) > 0 else 0.0

        # B³ F1
        n_instances = max(g["n"], 1)
        b3_p = g["b3p"] / n_instances
        b3_r = g["b3r"] / n_instances
        b3_f1 = 2 * b3_p * b3_r / (b3_p + b3_r) if (b3_p + b3_r) > 0 else 0.0


        # Structural score
        num_names = max(g["num_names"], 1)
        le = g["le"] / num_names
        se = g["se"] / num_names
        struct_score = 1.0 - (le + se) / 2.0

        return {
            "pairwise": {
                "precision": pw_p, "recall": pw_r, "f1": pw_f1
            },
            "b3": {
                "precision": b3_p, "recall": b3_r, "f1": b3_f1
            },
            "structural": {
                "lumping_error": le,
                "splitting_error": se,
                "score": struct_score
            }
        }

    def _composite(self, results: Dict[str, Any]) -> float:
        """
        Compute composite score as weighted average of metric families.

        Weights: 0.3 Pairwise-F1 + 0.5 B³-F1 + 0.2 Structural Score

        Args:
            results: Finalized metrics dictionary.

        Returns:
            Composite score in [0, 1].
        """
        return (
            0.3 * results["pairwise"]["f1"] +
            0.5 * results["b3"]["f1"] +
            0.2 * results["structural"]["score"]
        )

    def print_results(self, detailed: bool = True) -> None:
        """
        Pretty-print evaluation results.

        Args:
            detailed: If True, include additional analysis sections.
        """
        if not hasattr(self, 'results') or not self.results:
            logger.warning("No results available. Run .evaluate() first.")
            return

        print("\n" + "=" * 70)
        print("📊 AUTHOR NAME DISAMBIGUATION - MULTI-METRIC EVALUATION")
        print("=" * 70)

        if hasattr(self, 'stats') and self.stats:
            print("\n[DATASET STATISTICS]")
            print(f"   Total names:       {self.stats.get('total_names', 'N/A')}")
            print(f"   Total instances:   {self.stats.get('total_instances', 'N/A')}")
            print(f"   Perfect matches:   {self.stats.get('perfect_matches', 'N/A')}")

        pw = self.results.get('pairwise', {})
        print("\n[PAIRWISE-F (Link-based)]")
        print(f"   Precision: {pw.get('precision', 0):.4f}")
        print(f"   Recall:    {pw.get('recall', 0):.4f}")
        print(f"   F1:        {pw.get('f1', 0):.4f}")

        b3 = self.results.get('b3', {})
        print("\n[B³ (Bagga & Baldwin, 2009)]")
        print(f"   Precision: {b3.get('precision', 0):.4f}")
        print(f"   Recall:    {b3.get('recall', 0):.4f}")
        print(f"   F1:        {b3.get('f1', 0):.4f}")

        struc = self.results.get('structural', {})
        print("\n[STRUCTURAL ERRORS]")
        print(f"   Lumping Error:    {struc.get('lumping_error', 0):.4f}")
        print(f"   Splitting Error:  {struc.get('splitting_error', 0):.4f}")
        print(f"   Struct Score:     {struc.get('score', 0):.4f}")

        print("\n[COMPOSITE SCORE (0.3 PW + 0.5 B³ + 0.2 Struct)]")
        print(f"   {self.results.get('composite_score', 0):.4f}")

        print("\n" + "=" * 70)

    def save_to_excel(
        self,
        base_output_path: str | Path = "reports",
        experiment_name: str = "Exp",
        args: Any = None
    ) -> str:
        """
        Export results to Excel report.

        Filename format: {EMBEDDING}_{DATASET}_{EXPERIMENT}_{TIMESTAMP}.xlsx

        Args:
            base_output_path: Directory where Excel file will be saved.
            experiment_name: Experiment identifier for filename.
            args: Optional argparse.Namespace with metadata:
                - emb_type: Embedding type (W2V, SBERT, SPECTER, etc.)
                - save_path: Path used to infer dataset name (WhoIsWho, OC, etc.)
                - db_eps, db_min: DBSCAN parameters
                - use_citations, rel_on: Feature flags

        Returns:
            Path to generated Excel file.

        Raises:
            Exception: If Excel export fails (logged, not re-raised).
        """
        logger.info("Generating Excel report...")
        r = self.results

        # Extract embedding type
        emb_type = "EMB"
        if args and hasattr(args, 'emb_type'):
            emb_type = str(args.emb_type).upper()

        # Infer dataset name
        dataset_name = "Dataset"
        if args and hasattr(args, 'save_path'):
            path_str = str(args.save_path).lower()
            if "whoiswho" in path_str:
                dataset_name = "WhoIsWho"
            elif "oc" in path_str or "opencitations" in path_str:
                dataset_name = "OC"

        # Prepare filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{emb_type}_{dataset_name}_{experiment_name}_{timestamp}.xlsx"
        full_path = Path(base_output_path) / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data row
        pw = r.get("pairwise", {})
        b3 = r.get("b3", {})
        struc = r.get("structural", {})
        stats = getattr(self, "stats", {})

        data = {
            "Dataset": dataset_name,
            "Emb_Type": emb_type,
            "Mode": experiment_name,
            "Composite_Score": r.get("composite_score", 0),
            "Pairwise_F1": pw.get("f1", 0),
            "Pairwise_Prec": pw.get("precision", 0),
            "Pairwise_Rec": pw.get("recall", 0),
            "B3_F1": b3.get("f1", 0),
            "B3_Prec": b3.get("precision", 0),
            "B3_Rec": b3.get("recall", 0),
            "Lumping_Error": struc.get("lumping_error", 0),
            "Splitting_Error": struc.get("splitting_error", 0),
            "Structural_Score": struc.get("score", 0),
            "Perfect_Matches": stats.get("perfect_matches", 0),
            "Accuracy_Names": stats.get("accuracy_names", 0),
            "Total_Instances": stats.get("total_instances", 0),
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add optional DBSCAN parameters if present
        if args:
            if hasattr(args, 'db_eps'):
                data["Db_Eps"] = getattr(args, 'db_eps', 'N/A')
            if hasattr(args, 'db_min'):
                data["Db_Min"] = getattr(args, 'db_min', 'N/A')
            if hasattr(args, 'use_citations'):
                data["Use_Citations"] = getattr(args, 'use_citations', 'N/A')
            if hasattr(args, 'rel_on'):
                data["Rel_On"] = getattr(args, 'rel_on', 'N/A')

        try:
            df = pd.DataFrame([data])
            df.to_excel(full_path, index=False)
            logger.info(f"Excel report saved: {full_path}")
            print(f"📊 Report saved: {full_path}")
        except Exception as e:
            logger.error(f"Failed to generate Excel report: {e}")
            print(f"⚠️ Excel generation failed: {e}")

        return str(full_path)

    # =====================================================
    # Helpers
    # =====================================================

    def _prepare_truth(
        self,
        truth: Dict[str, List[str]] | List[List[str]]
    ) -> List[List[str]]:
        """
        Normalize ground-truth data format.

        Accepts either dict (cluster_id -> paper_ids) or list of lists.
        Filters out empty clusters.

        Args:
            truth: Ground-truth in flexible format.

        Returns:
            List of clusters (each cluster is list of paper IDs).
        """
        if isinstance(truth, dict):
            return [v for v in truth.values() if v and isinstance(v, list)]
        elif isinstance(truth, list):
            return [v for v in truth if isinstance(v, list) and v]
        return []

    def _accumulate(
        self,
        global_stats: Dict[str, float],
        single_stats: Dict[str, float]
    ) -> None:
        """
        Accumulate statistics from one name into global totals.

        Args:
            global_stats: Accumulator dictionary.
            single_stats: Single-name statistics.
        """
        for key in global_stats:
            global_stats[key] += single_stats[key]

    def _empty_one(self) -> Dict[str, float]:
        """
        Return empty stats template for a single name.

        Returns:
            Zero-initialized stats dictionary.
        """
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "b3p": 0.0, "b3r": 0.0,
            "le": 0.0, "se": 0.0,
            "n": 0,
            "num_names": 1
        }

    def _empty(self) -> Dict[str, Any]:
        """
        Return empty results template (all zeros/worst case).

        Returns:
            Zero-initialized results dictionary.
        """
        return {
            "pairwise": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "b3": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "structural": {
                "lumping_error": 1.0,
                "splitting_error": 1.0,
                "score": 0.0
            },
            "composite_score": 0.0
        }


if __name__ == "__main__":
    # Example usage
    evaluator = MultiMetricEvaluator(
        "predictions.json",
        "ground_truth.json",
        verbose=True
    )

    results = evaluator.evaluate()
    save_json(results, "results.json")
    evaluator.save_to_excel(base_output_path="reports")
