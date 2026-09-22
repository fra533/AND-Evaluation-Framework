#!/usr/bin/env python3
"""
Example usage of MultiMetricEvaluator.

This script demonstrates:
1. Basic evaluation
2. Accessing results
3. Exporting to Excel
4. Custom experiment metadata
"""

import argparse
from evaluate_BOND import MultiMetricEvaluator, save_json, load_json


def example_basic():
    """Simple evaluation with default settings."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Evaluation")
    print("=" * 70)

    evaluator = MultiMetricEvaluator(
        predictions_file="example_predictions.json",
        ground_truth_file="example_ground_truth.json",
        verbose=True
    )

    results = evaluator.evaluate()

    # Access specific metrics
    print("\n[Programmatic Access]")
    print(f"B³ F1: {results['b3']['f1']:.4f}")
    print(f"Composite Score: {results['composite_score']:.4f}")

    # Save results
    save_json(results, "example_results.json")
    print("\nResults saved to: example_results.json")


def example_with_metadata():
    """Evaluation with experiment metadata for Excel export."""
    print("\n" + "=" * 70)
    print("Example 2: Evaluation with Experiment Metadata")
    print("=" * 70)

    evaluator = MultiMetricEvaluator(
        predictions_file="example_predictions.json",
        ground_truth_file="example_ground_truth.json",
        verbose=True
    )

    # Run evaluation
    results = evaluator.evaluate()

    # Create experiment metadata
    args = argparse.Namespace(
        emb_type="SBERT",
        save_path="./datasets/OC",
        db_eps=0.5,
        db_min=2,
        use_citations=True,
        rel_on="cosine"
    )

    # Export to Excel with metadata
    report_path = evaluator.save_to_excel(
        base_output_path="reports",
        experiment_name="SBERT_DBSCAN_v1",
        args=args
    )
    print(f"\nExcel report saved to: {report_path}")


def example_batch_evaluation():
    """Evaluate multiple experiments and compare results."""
    print("\n" + "=" * 70)
    print("Example 3: Batch Evaluation (Multiple Experiments)")
    print("=" * 70)

    experiments = [
        {
            "name": "W2V_Baseline",
            "emb_type": "W2V",
            "db_eps": 0.3,
            "db_min": 2
        },
        {
            "name": "SBERT_v1",
            "emb_type": "SBERT",
            "db_eps": 0.5,
            "db_min": 2
        },
        {
            "name": "SPECTER_v1",
            "emb_type": "SPECTER",
            "db_eps": 0.6,
            "db_min": 3
        }
    ]

    results_summary = []

    for exp in experiments:
        print(f"\n→ Evaluating: {exp['name']}")

        evaluator = MultiMetricEvaluator(
            predictions_file="example_predictions.json",
            ground_truth_file="example_ground_truth.json",
            verbose=False  # Silence individual runs
        )

        results = evaluator.evaluate()

        # Prepare metadata
        args = argparse.Namespace(
            emb_type=exp['emb_type'],
            save_path="./datasets/OC",
            db_eps=exp['db_eps'],
            db_min=exp['db_min'],
            use_citations=True
        )

        # Export to Excel
        evaluator.save_to_excel(
            base_output_path="reports",
            experiment_name=exp['name'],
            args=args
        )

        # Collect results for comparison
        results_summary.append({
            "experiment": exp['name'],
            "composite_score": results['composite_score'],
            "b3_f1": results['b3']['f1'],
            "pairwise_f1": results['pairwise']['f1'],
            "lumping_error": results['structural']['lumping_error'],
            "splitting_error": results['structural']['splitting_error']
        })

    # Print comparison table
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Experiment':<20} {'Composite':<12} {'B³ F1':<12} {'PW F1':<12} {'LE':<10} {'SE':<10}")
    print("-" * 100)

    for row in results_summary:
        print(
            f"{row['experiment']:<20} "
            f"{row['composite_score']:<12.4f} "
            f"{row['b3_f1']:<12.4f} "
            f"{row['pairwise_f1']:<12.4f} "
            f"{row['lumping_error']:<10.4f} "
            f"{row['splitting_error']:<10.4f}"
        )

    print("=" * 100)
    print("\nAll Excel reports saved to ./reports/")


def example_detailed_analysis():
    """Evaluate and print detailed results."""
    print("\n" + "=" * 70)
    print("Example 4: Detailed Analysis")
    print("=" * 70)

    evaluator = MultiMetricEvaluator(
        predictions_file="example_predictions.json",
        ground_truth_file="example_ground_truth.json",
        verbose=False
    )

    results = evaluator.evaluate()

    # Print detailed results
    evaluator.print_results(detailed=True)

    # Access nested structures
    print("\n[Direct Access to Results]")
    print(f"Pairwise Precision: {results['pairwise']['precision']:.4f}")
    print(f"B³ Recall: {results['b3']['recall']:.4f}")
    print(f"K-metric: {results['k_metric']['k']:.4f}")
    print(f"Structural Score: {results['structural']['score']:.4f}")

    # Save JSON
    save_json(results, "detailed_results.json")
    print("\nDetailed results saved to: detailed_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example usage of MultiMetricEvaluator"
    )
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Which example to run (default: 1)"
    )

    args = parser.parse_args()

    examples = {
        1: example_basic,
        2: example_with_metadata,
        3: example_batch_evaluation,
        4: example_detailed_analysis
    }

    examples[args.example]()

    print("\n✅ Done!\n")
