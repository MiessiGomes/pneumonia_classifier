from evaluate import evaluate
from train import train


def main() -> None:
    """Runs the complete training and evaluation pipeline."""
    print("--- Starting Training ---")
    train()
    print("\n--- Starting Evaluation ---")
    evaluate()
    print("\n--- Pipeline Finished ---")


if __name__ == "__main__":
    main()
