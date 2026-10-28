import argparse
import csv
import json
from pathlib import Path

from tensorflow import keras

from src.config import DATASET_DIR, MODELS_DIR
from src.dataset import DatasetLoader
from src.thresholds import run_validation_threshold_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation-threshold sweep for confidence selection.")
    parser.add_argument("--model_path", default="models/final_model.keras")
    parser.add_argument("--dataset_dir", default=str(DATASET_DIR))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="models/threshold_sweep.csv")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    loader = DatasetLoader(dataset_dir=args.dataset_dir, batch_size=args.batch_size)
    _, val_ds, _ = loader.get_datasets()
    model = keras.models.load_model(str(model_path))

    summary = run_validation_threshold_sweep(model, val_ds)
    rows = summary["rows"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "accepted_count",
                "review_count",
                "acceptance_rate",
                "precision",
                "recall",
                "f1",
                "accuracy",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(json.dumps({
        "best_threshold": summary["best_threshold"],
        "output_csv": str(output_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
