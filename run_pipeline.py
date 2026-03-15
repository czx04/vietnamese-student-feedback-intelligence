import pandas as pd

from ml.data import export_processed
from ml.train import run_training


def main() -> None:
    paths = export_processed()
    print("Prepared:", ", ".join(paths))
    results = run_training()
    comparison = pd.DataFrame(results)[
        ["task", "model", "accuracy", "macro_f1", "weighted_f1"]
    ]
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
