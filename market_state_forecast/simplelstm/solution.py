import os
import numpy as np
import torch
import torch.nn as nn

# Set seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

from utils import DataPoint, ScorerStepByStep

from lstm import LSTMModel

class PredictionModel:
    """
    Prediction model using a pre-trained, stateful LSTM.
    """
    def __init__(self, model_path="lstm_model.pth", input_dim=32, hidden_dim=200, layer_dim=2, output_dim=32, dropout=0.2):
        self.current_seq_ix = -1
        self.hidden_state = None
        self.cell_state = None

        self.input_dim = input_dim
        
        # Load the pre-trained model
        self.model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, dropout)
        
        # Construct the full path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_model_path = os.path.join(current_dir, model_path)

        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Model file not found at {full_model_path}. Please run train.py first.")
            
        self.model.load_state_dict(torch.load(full_model_path))
        self.model.eval() # Set model to evaluation mode

    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            # Start of a new sequence, reset state
            self.current_seq_ix = data_point.seq_ix
            self.hidden_state = None
            self.cell_state = None

        # Prepare the input tensor from the current data_point's state
        # The state is a single time step, so we reshape it to (1, 1, input_dim)
        # where 1 is the batch size and 1 is the sequence length
        input_tensor = torch.tensor(data_point.state, dtype=torch.float32).reshape(1, 1, self.input_dim)

        # The hidden state for the LSTM is a tuple (hidden_state, cell_state)
        hidden = (self.hidden_state, self.cell_state) if self.hidden_state is not None else None

        # Pass the input and the hidden state to the model
        with torch.no_grad():
            prediction, (self.hidden_state, self.cell_state) = self.model(input_tensor, hidden)

        if not data_point.need_prediction:
            # We've updated the state, but we don't need to return a prediction
            return None
        
        # The model returns predictions for each step in the input sequence.
        # Since our input sequence length is 1, we take the first and only prediction.
        return prediction.numpy().flatten()

def test_model(model, scorer):
    results = scorer.score(model)
    print("\nResults:")
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(CURRENT_DIR, "..", "datasets", "train.parquet")

    if not os.path.exists(test_file):
        print(f"Error: Test file not found at {test_file}")
    else:
        scorer = ScorerStepByStep(test_file)
        print("Testing LSTM model...")
        print(f"Feature dimensionality: {scorer.dim}")
        
        try:
            model = PredictionModel()
            test_model(model, scorer)
        except FileNotFoundError as e:
            print(e)
            
        print("\n" + "=" * 60)
        print("Submission-ready solution using a pre-trained model.")
        print("=" * 60)
