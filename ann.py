import torch

class ANN:
    def __init__(self, layer_dims, learning_rate, seed=42):
        torch.manual_seed(seed)
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self._build_model()

    def _build_model(self):
        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()

        for i in range(len(self.layer_dims) - 1):
            in_dim = self.layer_dims[i]
            out_dim = self.layer_dims[i + 1]
            W = torch.nn.Parameter(torch.randn(in_dim, out_dim) * torch.sqrt(torch.tensor(2.0 / in_dim)))
            b = torch.nn.Parameter(torch.zeros(1, out_dim))
            self.weights.append(W)
            self.biases.append(b)

        self.optimizer = torch.optim.SGD(list(self.weights) + list(self.biases), lr=self.learning_rate)

    def relu_manual(self, x):
        # TODO 1: Implement ReLU manually using torch operations
        # Formula: ReLU(x) = max(0, x)
        raise NotImplementedError("Write your code here")
        return 

    def sigmoid_manual(self, x):
        # TODO 2: Implement Sigmoid manually using torch operations
        # Formula: Sigmoid(x) = 1 / (1 + exp(-x))
        raise NotImplementedError("Write your code here")
        return 

    def _forward(self, X):
        # TODO 3: Write the feed-forward implementation
        # Steps:
        # - For each layer:
        #   1. Linear transform: Z = A @ W + b
        #   2. Apply activation: ReLU for hidden, Sigmoid for output
        A = X
        for i in range(len(self.weights)):
            raise NotImplementedError("Write your code here")
        return A

    def _compute_loss(self, y_hat, y_true):
        y_hat = torch.clamp(y_hat, 1e-7, 1 - 1e-7)
        # TODO 4: Write loss function to compute binary cross-entropy
        # Formula: -mean(y*log(ŷ) + (1-y)*log(1-ŷ))
        raise NotImplementedError("Write your code here")
        return

    def train_one_epoch(self, X_np, y_np):
        '''
        Do not change this code
        '''
        X = torch.tensor(X_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)

        self.optimizer.zero_grad()
        y_hat = self._forward(X)
        loss = self._compute_loss(y_hat, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict_proba(self, X_np):
        # TODO 5: Convert input to tensor and return predicted probabilities
        raise NotImplementedError("Write your code here")
        return 

    def predict(self, X_np):
        # TODO 6: Use your prefered threshold (e.g 0.5) to return binary class predictions
        raise NotImplementedError("Write your code here")
        return 
