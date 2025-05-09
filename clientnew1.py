import requests
import random
import numpy as np

SERVER_URL = 'http://192.168.29.96:5000'  # Replace with server IP

# -------------------- Data Generation -------------------- #
def load_client_data(num_samples=100, num_features=561):
    """Generate simulated client data."""
    features = np.random.rand(num_samples, num_features)  # Generate random features
    labels = np.random.randint(0, 6, size=num_samples)  # Generate random labels (0 to 5)
    return features, labels

# -------------------- Simple Model Training and Prediction -------------------- #
class SimpleModel:
    """A simple neural network-like model implemented using NumPy."""
    def __init__(self, input_size, num_classes):
        self.weights1 = np.random.rand(input_size, 32) * 0.01  # Input layer to hidden layer 1
        self.weights2 = np.random.rand(32, 16) * 0.01  # Hidden layer 1 to hidden layer 2
        self.weights3 = np.random.rand(16, num_classes) * 0.01  # Hidden layer 2 to output layer
        self.biases1 = np.zeros(32)
        self.biases2 = np.zeros(16)
        self.biases3 = np.zeros(num_classes)

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """Forward pass."""
        self.layer1 = self.relu(np.dot(X, self.weights1) + self.biases1)
        self.layer2 = self.relu(np.dot(self.layer1, self.weights2) + self.biases2)
        self.output = self.softmax(np.dot(self.layer2, self.weights3) + self.biases3)
        return self.output

    def predict(self, X):
        """Predict logits for given input."""
        return self.forward(X)

# -------------------- Performance Score Generation -------------------- #
def generate_performance_score(distribution='beta'):
    """Generate performance scores using various non-normal distributions."""
    if distribution == 'beta':
        score = np.random.beta(a=2, b=5)  # 'a' and 'b' control shape (skewed towards 0)
    elif distribution == 'lognormal':
        score = np.clip(np.random.lognormal(mean=0, sigma=0.5), 0, 1)
    elif distribution == 'exponential':
        score = np.clip(np.random.exponential(scale=0.5), 0, 1)
    elif distribution == 'uniform':
        score = np.random.uniform(0, 1)
    else:
        score = np.clip(np.random.normal(0.8, 0.1), 0, 1)
    return round(score, 4)

# -------------------- Single Client Debugging -------------------- #
if __name__ == '__main__':
    try:
        # Step 1: Load client data
        features, labels = load_client_data()

        # Step 2: Train the local model
        input_size = features.shape[1]
        num_classes = 6
        model = SimpleModel(input_size, num_classes)

        # Debug: Display a sample of features and labels
        print(f"Sample features: {features[:2]}")
        print(f"Sample labels: {labels[:2]}")

        # Step 3: Retrieve public data
        response = requests.get(f"{SERVER_URL}/get_public_data")
        if response.status_code == 200:
            public_data = np.array(response.json())
        else:
            print(f"Failed to retrieve public data. Status code: {response.status_code}")
            public_data = np.random.rand(10, features.shape[1])  # Fallback data for debugging

        # Debug: Display a sample of public data
        print(f"Public data sample: {public_data[:2]}")

        # Step 4: Generate logits and embeddings
        logits = model.predict(public_data)
        embeddings = np.random.rand(len(public_data), 128)  # Generate embeddings

        # Debug: Display sample logits and embeddings
        print(f"Sample logits: {logits[:2]}")
        print(f"Sample embeddings: {embeddings[:2]}")

        # Step 5: Generate performance score
        performance_score = generate_performance_score(distribution='beta')
        print(f"Performance Score (Beta Distribution): {performance_score}")

        # Step 6: Submit logits and embeddings with performance score
        payload = {
            'logits': logits.tolist(),
            'embeddings': embeddings.tolist(),
            'performance_score': performance_score
        }
        response = requests.post(f"{SERVER_URL}/submit_logits", json=payload)
        if response.status_code == 200:
            print(response.json().get('message', 'Submission Successful'))
        else:
            print(f"Submission failed. Status code: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"An error occurred: {e}")