import torch
import torch.nn as nn

class CNNTextClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        """
        CNN for text classification based on numerical features (TF-IDF, manual, BERT, etc.).
        Architecture:
            - Input: (batch_size, input_dim)
            - Reshape to (batch_size, 1, input_dim) for Conv1D
            - Conv1D -> ReLU -> AdaptiveMaxPool -> Dense (Sigmoid)
        """
        super(CNNTextClassifier, self).__init__()

        # TODO 1: You can increase out_channels to improve model capacity
        raise NotImplementedError()

        # TODO 2: You may adjust pooling strategy or output dim of fc layer
        raise NotImplementedError()

    def forward(self, x):
        """
        TODO 3: Implement the forward pass of this CNN:
            Step 1. Unsqueeze the input to make shape (batch_size, 1, input_dim)
            Step 2. Apply Conv1D followed by ReLU activation
            Step 3. Apply AdaptiveMaxPool1D to reduce sequence to single vector
            Step 4. Flatten and apply the final linear layer
            Step 5. Apply sigmoid for binary classification
        """
        raise NotImplementedError()

        return x
