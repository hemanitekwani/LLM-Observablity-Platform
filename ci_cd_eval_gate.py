import os
import sys

EXPERIMENT_ID = "530252935482749885"

FAITHFULNESS_THRESHOLD = 0.60
RELEVANCY_THRESHOLD = 0.25
PRECISION_THRESHOLD = 0.40

experiment_dir = os.path.join("mlruns", EXPERIMENT_ID)

runs = []

for item in os.listdir(experiment_dir):
    path = os.path.join(experiment_dir, item)

    if (
        os.path.isdir(path)
        and item != "meta.yaml"
        and not item.startswith(".")
    ):
        runs.append(path)

if not runs:
    print("No MLflow runs found")
    sys.exit(1)

latest_run = max(runs, key=os.path.getmtime)

print(f"Latest Run: {latest_run}")

metrics_dir = os.path.join(latest_run, "metrics")


def read_metric(metric_name):
    metric_file = os.path.join(metrics_dir, metric_name)

    if not os.path.exists(metric_file):
        return None

    with open(metric_file, "r") as f:
        content = f.read().strip().split()

    return float(content[1])


faithfulness = read_metric("faithfulness")
relevancy = read_metric("answer_relevancy")
precision = read_metric("context_precision")
recall = read_metric("context_recall")


print("\nModel Quality Report")
print("--------------------")
print(f"Faithfulness      : {faithfulness}")
print(f"Answer Relevancy  : {relevancy}")
print(f"Context Precision : {precision}")
print(f"Context Recall : {recall}")


if faithfulness is None:
    sys.exit("Faithfulness metric missing")

if relevancy is None:
    sys.exit("Answer Relevancy metric missing")

if precision is None:
    sys.exit("Context Precision metric missing")

if recall is None:
    sys.exit("Context Recalll metric missing")


if faithfulness < FAITHFULNESS_THRESHOLD:
    sys.exit(
        f"FAILED: Faithfulness below threshold "
        f"({faithfulness} < {FAITHFULNESS_THRESHOLD})"
    )

if relevancy < RELEVANCY_THRESHOLD:
    sys.exit(
        f"FAILED: Relevancy below threshold "
        f"({relevancy} < {RELEVANCY_THRESHOLD})"
    )

if precision < PRECISION_THRESHOLD:
    sys.exit(
        f"FAILED: Precision below threshold "
        f"({precision} < {PRECISION_THRESHOLD})"
    )

print("\nPASSED: Model meets deployment criteria")