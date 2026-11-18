"""Write a measured development-data report from completed PathSnap results."""

from collections import Counter
import json

import pandas as pd

from pathsnap.config import ART, RES, ROOT


MODELS = {
    "logreg": ("Logistic regression", "L2-regularized binary log loss",
               "A scaled linear probability baseline with inexpensive inference.",
               "A linear decision surface does not directly model feature interactions."),
    "knn": ("K-nearest neighbors", "Distance-weighted voting after scaling",
            "Learns local patterns without a parametric decision surface.",
            "Neighbor queries are expensive. Training folds use a 100,000-row cap."),
    "random_forest": ("Random forest", "Gini impurity reduction across bagged trees",
                      "Represents nonlinear splits and averages many trees.",
                      "Each tree uses a 20% bootstrap sample to bound training cost."),
    "lightgbm": ("LightGBM", "Binary log loss optimized by gradient boosting",
                 "Learns interactions between geometric and road-topology features.",
                 "Leaf capacity and boosting rounds affect runtime and overfitting."),
    "mlp": ("Multilayer perceptron", "Binary cross-entropy with L2 regularization",
            "Learns nonlinear combinations of standardized candidate features.",
            "Convergence depends on optimization settings, and training is iterative."),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def percent(score):
    return f"{100 * score:.2f}%"


def main():
    rows = {name: read_json(RES / "models" / f"{name}.json") for name in MODELS}
    search = read_json(RES / "best_params.json")
    manifest = read_json(ART / "final_model_manifest.json")
    tuned = read_json(RES / "models" / f"{search['tuned']}.json")
    inventory = pd.read_csv(RES / "eda_inventory.csv")
    coverage = pd.read_csv(RES / "candidate_stats.csv").set_index("part")
    route = pd.read_csv(RES / "hmm_vs_classifier_per_trip.csv")
    rehearsal = pd.read_csv(RES / "rehearsal.csv")
    trips = Counter(inventory["part"])
    if trips != {"train": 80, "student": 10} or len(route) != 10 or len(rehearsal) != 10:
        raise ValueError("The report requires the complete development splits and rehearsal results.")
    if not rehearsal["status"].eq("pass").all():
        raise ValueError("The local API rehearsal has failing requests.")
    selected = (rows | {tuned["model"]: tuned})[manifest["model"]]
    if (ART / "hmm_model_sha256.txt").read_text().strip() != manifest["sha256"]:
        raise ValueError("Decoder settings and exported classification model do not match.")

    lines = [
        "# PathSnap technical report",
        "",
        "Rishi, Sanjit, Harsha, and Shailaja · UE24CS352A Machine Learning",
        "",
        "## 1. Method and partition policy",
        "",
        "PathSnap scores nearby directed arcs for each noisy GPS observation and uses Viterbi decoding "
        "to select an ordered route. The development archive supplied 80 training and 10 student-test "
        "trips. Model fitting, GroupKFold validation, Bayesian search and decoder tuning used training "
        "trips. Student-test trips measured development performance. The reserved evaluator split "
        "was excluded from this run.",
        "",
        "## 2. Data and exploratory analysis",
        "",
        "| Partition | Trips | GPS observations | Candidate rows | Candidate miss rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for partition in ("train", "student"):
        row = coverage.loc[partition]
        lines.append(f"| {partition} | {int(row.trips)} | {int(row.observations):,} | "
                     f"{int(row.candidate_rows):,} | {percent(row.candidate_miss_rate)} |")
    lines += [
        "",
        f"The median share of GPS samples one second apart was {percent(inventory['pct_dt_1s'].median())}. "
        f"The longest observed interval was {inventory['max_dt'].max():,.0f} seconds. A candidate miss "
        "occurs when the aligned reference arc is absent. Candidate generation uses up to 12 road arcs "
        "within 200 m, with a nearest-arc fallback. Features include distance and relative rank, heading "
        "alignment, position along the arc, speed, timing, and road connectivity. Point labels are "
        "derived by monotonically aligning the supplied reference route to the GPS observations.",
        "",
        "![Development partition inventory](../results/figs/eda_hist.png)",
        "",
        "![Example training GPS trajectory and road network](../results/figs/eda_trip.png)",
        "",
        "## 3. Supervised model comparison",
        "",
        "| Model | Grouped CV Top-1 | CV Macro F1 | Student Top-1 | Student Macro F1 | "
        "Log loss | Student arc adjacency | Inference ms/1k rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(rows.items(), key=lambda item: (-item[1]["cv_top1"], -item[1]["cv_macro_f1"])):
        lines.append(f"| {MODELS[name][0]} | {percent(row['cv_top1'])} | {row['cv_macro_f1']:.4f} | "
                     f"{percent(row['st_top1'])} | {row['st_macro_f1']:.4f} | {row['st_log_loss']:.4f} | "
                     f"{percent(row['st_route_continuity'])} | {row['infer_ms_per_1k_rows']:.2f} |")
    lines += [
        "",
        "Top-1 evaluates the highest-scoring arc for each observation. F1 and log loss use binary "
        "candidate rows. The student continuity column uses direct adjacency of consecutive "
        "distinct chosen arcs. Model selection used mean grouped-CV Top-1 with Macro F1 as a tiebreak.",
        "",
        "![Grouped model comparison](../results/figs/cmp_cv_macro_f1.png)",
        "",
        "![Student comparison](../results/figs/cmp_student_top1_f1.png)",
    ]
    for name, row in rows.items():
        label, objective, strength, limitation = MODELS[name]
        gap = (row["train_top1_20trips"] - row["cv_top1"]) * 100
        lines += [
            "",
            f"### {label}",
            "",
            f"{strength} {limitation} The objective or scoring method is {objective}.",
            "",
            f"The initial configuration was `{row['config']}`.",
            "",
            f"The in-sample diagnostic on the first 20 training trips was {percent(row['train_top1_20trips'])}, "
            f"versus {percent(row['cv_top1'])} in grouped validation and {percent(row['st_top1'])} on "
            f"student test. The training-to-CV difference was {gap:+.2f} points. The diagnostic subset "
            "can have a different trip mix from the folds, so this gap alone does not establish overfitting.",
        ]
    baseline = rows[search["baseline"]]
    lines += [
        "",
        "![LightGBM loss curve](../results/figs/curve_lightgbm.png)",
        "",
        "![MLP loss curve](../results/figs/curve_mlp.png)",
        "",
        "## 4. Bayesian tuning and export",
        "",
        f"Optuna TPE completed {search['completed_trials']} trials on {MODELS[search['baseline']][0]} "
        f"using grouped training CV Top-1. The best search configuration was "
        f"`{json.dumps(search['best_params'], sort_keys=True)}`. Completed search trials took "
        f"{search['completed_trial_seconds'] / 60:.1f} minutes in total.",
        "",
        "The search covered these ranges.",
        "",
    ]
    lines.extend(f"- `{parameter}` · `{distribution}`"
                 for parameter, distribution in search["search_space"].items())
    lines += [
        "",
        "| Configuration | CV Top-1 | CV Macro F1 | Student Top-1 | Student Macro F1 | "
        "Log loss | Inference ms/1k rows | Fit seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (baseline, tuned):
        lines.append(f"| {row['model']} | {percent(row['cv_top1'])} | {row['cv_macro_f1']:.4f} | "
                     f"{percent(row['st_top1'])} | {row['st_macro_f1']:.4f} | {row['st_log_loss']:.4f} | "
                     f"{row['infer_ms_per_1k_rows']:.2f} | {row['train_seconds']:.2f} |")
    lines += [
        "",
        f"The exported classifier is **{manifest['model']}**, with {percent(selected['cv_top1'])} "
        "grouped CV Top-1. The full trial record and paired student diagnostics are under `results/`.",
        "",
        "## 5. Route reconstruction on student test",
        "",
        "At each timestamp, candidate arcs are hidden states and weighted log classifier scores are "
        "emissions. Transition scores measure the discrepancy between directed road distance and "
        "GPS displacement, adjusted for elapsed time. Viterbi maximizes the accumulated score and "
        "then recovers intermediate directed arcs. Unreachable transitions incur a penalty and "
        "appear as breaks. Candidate generation falls back to a nearest arc when a radius search "
        "returns no candidates. Large time gaps affect the distance search bound and transition weight.",
        "",
        "| Method | Mean trip Top-1 | Directed transition continuity | Arc-overlap F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, label in (("classifier", "Classifier only"), ("hmm", "Classifier + Viterbi")):
        lines.append(f"| {label} | {percent(route[f'{name}_top1'].mean())} | "
                     f"{percent(route[f'{name}_continuity'].mean())} | {route[f'{name}_route_f1'].mean():.4f} |")
    differences = route["hmm_top1"] - route["classifier_top1"]
    settings = read_json(ART / "hmm_config.json")
    lines += [
        "",
        f"Viterbi improved Top-1 on {int(differences.gt(0).sum())} of 10 trips and reduced it on "
        f"{int(differences.lt(0).sum())}. The largest decrease was {100 * min(0., differences.min()):.2f} "
        f"points. The training-OOF grid selected `{json.dumps(settings, sort_keys=True)}`. "
        "Arc-overlap F1 counts shared arcs with multiplicity. It does not measure traversal order.",
        "",
        "## 6. Folium visualization",
        "",
        "The map generator renders noisy GPS points, the road network, the reference route, classifier "
        "picks, and Viterbi reconstruction as separate layers. It can regenerate interactive student "
        "maps from the team's development data.",
        "",
        "## 7. API and deployment",
        "",
        f"The local FastAPI rehearsal passed {len(rehearsal)} of 10 requests. The median response "
        f"round-trip time was {rehearsal['round_trip_seconds'].median():.2f} seconds on the test machine. "
        "The model and decoder are bound by SHA-256. The Streamlit client sends a trip ID and renders "
        "the returned GPS points, predicted route, and aggregate scores. The server reads reference "
        "routes after prediction for scoring. The physical two-laptop rehearsal is still pending.",
        "",
        "## Reproducibility and disclosure",
        "",
        "Run the commands in the project README with the organizer-provided development data. "
        "AI assistance was used to integrate and verify the pipeline and assemble this report from "
        "the observed run outputs. The metric values above were generated by the executed pipeline.",
        "",
    ]

    if (RES / "figs" / "student_route_rehearsal.png").is_file():
        heading = lines.index("## 7. API and deployment")
        lines[heading:heading] = [
            "![Student-test route with local coordinates](../results/figs/student_route_rehearsal.png)",
            "",
        ]
    (ROOT / "report" / "official_development_results.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
