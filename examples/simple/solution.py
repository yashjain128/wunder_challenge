import os
import sys
import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add project root folder to path for importing utils
sys.path.append(f"{CURRENT_DIR}/../..")

from utils import DataPoint, ScorerStepByStep

class PredictionModel:
    """
    Simple model that predicts the next value as a moving average
    of all previous values in the current sequence.
    """

    def __init__(self):
        self.current_seq_ix = None
        self.sequence_history = []

    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        self.sequence_history.append(data_point.state.copy())

        if not data_point.need_prediction:
            return None

        return np.mean(self.sequence_history, axis=0)


if __name__ == "__main__":
    # Check existence of test file
    test_file = f"{CURRENT_DIR}/../../datasets/train.parquet"

    # Create and test our model
    model = PredictionModel()

    # Load data into scorer
    scorer = ScorerStepByStep(test_file)

    print("Testing simple model with moving average...")
    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")

    print(f">>{scorer.dataset}")

    seq_length = 10
    max_time = scorer.dataset.shape[0]
    print(f">>{max_time}<<")
    n = 32
    timesteps = np.arange(0, max_time)

    print(scorer.dataset.columns.to_list())

    plt.figure(figsize=(12, 6))
    for i in range(0, n, 1):
        plt.plot(timesteps[:1000], scorer.dataset[str(i)][:1000], label='Original Data')

    plt.show()
    # Evaluate our solution
    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(min(5, len(scorer.features))):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")

    print(f"\nTotal features: {len(scorer.features)}")

    print("\n" + "=" * 60)
    print("Try submitting an archive with solution.py file")
    print("to test the solution submission mechanism!")
    print("=" * 60)
