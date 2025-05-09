from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__)

# -------------------- Data Loading Functions -------------------- #
def load_public_data():
    """Load 10% of the UCI HAR dataset for public data."""
    try:
        data_path = 'C:/Users/ADMIN/Downloads/UCI HAR Dataset/UCI HAR Dataset/'
        features = pd.read_csv(data_path + 'features.txt', sep=r'\s+', header=None)
        feature_names = features[1].tolist()
        feature_names = pd.Series(feature_names).astype(str).str.strip()
        feature_names = feature_names + "_" + feature_names.groupby(feature_names).cumcount().astype(str)
        X_train = pd.read_csv(data_path + 'train/X_train.txt', sep=r'\s+', header=None, names=feature_names)
        return X_train.sample(frac=0.1, random_state=42)
    except Exception as e:
        print(f"Error loading public data: {e}")
        return pd.DataFrame()

def load_test_data():
    """Load test data and labels from the UCI HAR dataset."""
    try:
        data_path = 'C:/Users/ADMIN/Downloads/UCI HAR Dataset/UCI HAR Dataset/'
        features = pd.read_csv(data_path + 'features.txt', sep=r'\s+', header=None)
        feature_names = features[1].tolist()
        feature_names = pd.Series(feature_names).astype(str).str.strip()
        feature_names = feature_names + "_" + feature_names.groupby(feature_names).cumcount().astype(str)
        X_test = pd.read_csv(data_path + 'test/X_test.txt', sep=r'\s+', header=None, names=feature_names)
        y_test = pd.read_csv(data_path + 'test/y_test.txt', sep=r'\s+', header=None)[0]
        return X_test, y_test
    except Exception as e:
        print(f"Error loading test data: {e}")
        return pd.DataFrame(), pd.Series()

public_data = load_public_data()
if public_data.empty:
    raise RuntimeError("Failed to load public data. Ensure the dataset path is correct.")

input_shape = (561,)
embedding_dim = 128
num_classes = 6

# -------------------- Global Student Model -------------------- #
global_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
global_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------- Generator -------------------- #
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(561, activation='tanh')
])

# -------------------- Discriminator -------------------- #
discriminator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(561,)),
    tf.keras.layers.Dense(256, activation='leaky_relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='leaky_relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# -------------------- Adversarial Model -------------------- #
class AdversarialModel(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(AdversarialModel, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        
    def compile(self, g_optimizer, d_optimizer):
        super(AdversarialModel, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        
    @tf.function
    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal([batch_size, 100])
        # Train Discriminator
        with tf.GradientTape() as d_tape:
            generated_data = self.generator(noise, training=True)
            real_output = self.discriminator(real_data, training=True)
            fake_output = self.discriminator(generated_data, training=True)
            d_loss_real = self.loss_fn(tf.ones_like(real_output), real_output)
            d_loss_fake = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = d_loss_real + d_loss_fake
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        # Train Generator
        with tf.GradientTape() as g_tape:
            generated_data = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_data, training=True)
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return {"d_loss": d_loss, "g_loss": g_loss}

adversarial_model = AdversarialModel(generator, discriminator)
adversarial_model.compile(
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
)

global_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
generator.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy'
)
discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy'
)

# -------------------- Server State -------------------- #
weights_received = []  # Collects tuples: (logits, embeddings, performance_score)

# -------------------- Performance Metrics Logging -------------------- #
training_losses = []  # Stores training loss per epoch
epoch_times = []      # Stores epoch durations

# -------------------- Evaluation Function -------------------- #
def evaluate_model():
    X_test, y_test = load_test_data()
    if X_test.empty or y_test.empty:
        return {"error": "Test data not available."}
    X_test_np = X_test.values.astype(np.float32)
    y_test_adjusted = y_test - 1  
    y_test_onehot = tf.keras.utils.to_categorical(y_test_adjusted, num_classes=num_classes)
    predictions = global_model.predict(X_test_np)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test_onehot, axis=1)
    accuracy = float(accuracy_score(true_labels, predicted_labels))
    precision = float(precision_score(true_labels, predicted_labels, average='macro', zero_division=0))
    recall = float(recall_score(true_labels, predicted_labels, average='macro', zero_division=0))
    f1 = float(f1_score(true_labels, predicted_labels, average='macro', zero_division=0))
    conf_matrix = confusion_matrix(true_labels, predicted_labels).tolist()
    avg_training_loss = float(np.mean(training_losses)) if training_losses else None
    epoch_times_native = [float(t) for t in epoch_times]
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "average_training_loss": avg_training_loss,
        "total_epochs": len(training_losses),
        "epoch_times": epoch_times_native
    }
    return metrics

# -------------------- Routes -------------------- #
@app.route('/get_public_data', methods=['GET'])
def get_public_data_route():
    if public_data.empty:
        return jsonify({"error": "Public data not available"}), 500
    return jsonify(public_data.values.tolist())

@app.route('/submit_logits', methods=['POST'])
def submit_logits_route():
    try:
        data = request.json
        if 'logits' not in data or 'embeddings' not in data or 'performance_score' not in data:
            return jsonify({"error": "Missing required fields"}), 400
        logits = np.array(data['logits'])
        embeddings = np.array(data['embeddings'])
        performance_score = data.get('performance_score', 0.5)
        weights_received.append((logits, embeddings, performance_score))
        # Process once at least two clients have submitted their data.
        if len(weights_received) >= 2:
            aggregate_and_update()
            return jsonify({"message": "Global model updated with distillation."})
        return jsonify({"message": "Waiting for more clients."})
    except Exception as e:
        print(f"Error in submit_logits: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate', methods=['GET'])
def evaluate_route():
    metrics = evaluate_model()
    return jsonify(metrics)

# -------------------- Graph-Based Aggregation Function -------------------- #
def graph_based_aggregation(client_logits, performance_scores):
    # Build feature matrix: flatten each client's logits and append its performance score
    features = []
    for logits, score in zip(client_logits, performance_scores):
        flattened = np.array(logits.flatten(), dtype=np.float32)
        feature_vector = np.concatenate([flattened, np.array([score], dtype=np.float32)])
        features.append(feature_vector)
    features = np.array(features, dtype=np.float32)
    
    print("Graph aggregation - features shape:", features.shape)
    
    # Normalize features and compute cosine similarity matrix
    norms = np.linalg.norm(features, axis=1, keepdims=True).astype(np.float32)
    norm_features = features / norms
    similarity_matrix = np.dot(norm_features, norm_features.T).astype(np.float32)
    
    # Convert to TensorFlow tensors
    features_tf = tf.convert_to_tensor(features, dtype=tf.float32)
    similarity_matrix_tf = tf.convert_to_tensor(similarity_matrix, dtype=tf.float32)
    
    # Simple GNN-like layer: aggregate neighbor features with a learnable weight
    weight = tf.Variable(tf.random.normal([features.shape[1], 1], dtype=tf.float32), trainable=True)
    aggregated_features = tf.nn.relu(tf.matmul(similarity_matrix_tf, features_tf))
    aggregated_features = tf.matmul(aggregated_features, weight)
    aggregated_features = tf.squeeze(aggregated_features, axis=1)  # shape: (num_clients,)
    
    adaptive_weights = tf.nn.softmax(aggregated_features)
    if isinstance(adaptive_weights, tf.Tensor):
        adaptive_weights_np = adaptive_weights.numpy().astype(np.float32)
    else:
        adaptive_weights_np = adaptive_weights.astype(np.float32)
    
    print("Adaptive weights from graph-based aggregation:", adaptive_weights_np)
    # Use these adaptive weights to combine the client logits
    aggregated_logits = np.sum([w * np.array(l, dtype=np.float32)
                                for w, l in zip(adaptive_weights_np, client_logits)], axis=0)
    return aggregated_logits

# -------------------- Aggregate and Update Function -------------------- #
def aggregate_and_update():
    global generator, discriminator, global_model, adversarial_model
    try:
        def heterogeneity_aware_selection(client_logits, performance_scores, threshold=0.85):
            selected_fl = []
            selected_fd = []
            for logits, score in zip(client_logits, performance_scores):
                if score >= threshold:
                    selected_fl.append(logits)
                else:
                    selected_fd.append(logits)
            return selected_fl, selected_fd

        # Use the received client data (no simulation)
        client_logits = [item[0] for item in weights_received]
        performance_scores = [item[2] for item in weights_received]
        selected_fl, selected_fd = heterogeneity_aware_selection(client_logits, performance_scores)
        
        # Use graph-based aggregation for combining client logits
        aggregated_logits = graph_based_aggregation(client_logits, performance_scores)
        print("Aggregated logits shape:", aggregated_logits.shape)
        
        # For debugging, reduce quantization and noise by commenting these out
        def quantize_data(data):
            # Comment out quantization for debugging:
            return data  # np.round(data * 255) / 255
        quantized_logits = quantize_data(aggregated_logits)
        print("Quantized logits sample:", quantized_logits.flatten()[:10])
        
        def apply_differential_privacy(data, epsilon=10.0):  # Increase epsilon to reduce noise
            noise = np.random.laplace(scale=1/epsilon, size=data.shape)
            return data + noise
        dp_logits = apply_differential_privacy(quantized_logits)
        soft_labels = tf.nn.softmax(dp_logits, axis=1).numpy()
        
        def generate_diverse_public_data(generator, num_samples=735):
            noise = np.random.normal(0, 1, (num_samples, 100))
            generated_data = generator.predict(noise)
            print("Generator output sample:", generated_data.flatten()[:10])
            return generated_data
        diverse_public_data = generate_diverse_public_data(generator)
        
        def simulate_non_iid_data(data, heterogeneity_factor=0.3):
            shuffled_data = data.copy()
            np.random.shuffle(shuffled_data)
            return shuffled_data[:int(len(data) * heterogeneity_factor)]
        non_iid_data = simulate_non_iid_data(diverse_public_data)
        print("Non-iid data shape:", non_iid_data.shape)
        
        def intermediate_layer_distillation(client_features, global_features):
            return tf.reduce_mean(tf.square(client_features - global_features))
        # For debugging, use fixed random values (in real use, supply actual client features)
        client_features = np.random.rand(735, 128)
        reshaped_data = tf.keras.layers.Dense(128)(diverse_public_data)
        global_features = reshaped_data
        intermediate_loss = intermediate_layer_distillation(client_features, global_features)
        print("Intermediate Layer Distillation Loss:", intermediate_loss.numpy())
        
        def train_discriminator(discriminator, real_data, fake_data):
            with tf.GradientTape() as tape:
                real_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(tf.ones_like(discriminator(real_data)), discriminator(real_data))
                )
                fake_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(tf.zeros_like(discriminator(fake_data)), discriminator(fake_data))
                )
                total_loss = real_loss + fake_loss
            gradients = tape.gradient(total_loss, discriminator.trainable_variables)
            discriminator.optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
            return total_loss
        noise = tf.random.normal([len(public_data), 100])
        fake_data = generator.predict(noise)
        real_data = public_data.values.astype(np.float32)
        discriminator_loss = train_discriminator(discriminator, real_data, fake_data)
        print("Discriminator Loss:", discriminator_loss.numpy())
        
        def jensen_shannon_divergence(predictions):
            avg_prediction = tf.reduce_mean(predictions, axis=0)
            kl_divergence = tf.keras.losses.KLDivergence()
            divergence = tf.reduce_mean([kl_divergence(avg_prediction, pred) for pred in predictions])
            return divergence
        
        def train_generator(generator, discriminator, aggregated_logits, public_data):
            noise = tf.random.normal([public_data.shape[0], 100])
            with tf.GradientTape() as tape:
                generated_data = generator(noise, training=True)
                adversarial_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(tf.ones_like(discriminator(generated_data)), discriminator(generated_data))
                )
                distillation_loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(aggregated_logits, global_model(generated_data, training=False))
                )
                client_predictions = [global_model(generated_data, training=False)]
                diversity_loss = jensen_shannon_divergence(client_predictions)
                total_loss = adversarial_loss + distillation_loss - 0.1 * diversity_loss
            gradients = tape.gradient(total_loss, generator.trainable_variables)
            generator.optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
            return total_loss
        generator_loss = train_generator(generator, discriminator, soft_labels, public_data)
        print("Generator Loss:", generator_loss.numpy())
        
        # Build a dataset from the synthetic non-iid data and soft labels
        dataset = tf.data.Dataset.from_tensor_slices((non_iid_data, soft_labels[:len(non_iid_data)])).batch(32)
        
        def multi_objective_loss(accuracy_loss, robustness_loss, efficiency_loss, alpha=0.5, beta=0.3):
            return accuracy_loss + alpha * robustness_loss + beta * efficiency_loss
        
        def train_global_step(x, y):
            x = tf.cast(tf.convert_to_tensor(x), tf.float32)
            y = tf.cast(tf.convert_to_tensor(y), tf.float32)
            with tf.GradientTape(persistent=True) as tape:
                predictions = global_model(x, training=True)
                accuracy_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, predictions))
                noise = tf.random.normal(x.shape, stddev=0.01)
                perturbed_data = x + noise
                perturbed_predictions = global_model(perturbed_data, training=True)
                robustness_loss = tf.reduce_mean(tf.square(predictions - perturbed_predictions))
                efficiency_loss = tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in global_model.trainable_weights])
                total_loss = multi_objective_loss(accuracy_loss, robustness_loss, efficiency_loss)
            gradients = tape.gradient(total_loss, global_model.trainable_variables)
            global_model.optimizer.apply_gradients(zip(gradients, global_model.trainable_variables))
            return total_loss
        
        # For debugging, we use fewer epochs
        start_time = time.time()
        epochs = 50  # Reduced for debugging
        for epoch in range(epochs):
            epoch_start = time.time()
            for batch_x, batch_y in dataset:
                loss = train_global_step(batch_x, batch_y)
                tf.print("Global Model Training Loss:", loss)
            epoch_duration = time.time() - epoch_start
            epoch_times.append(epoch_duration)
            training_losses.append(loss.numpy())
            print(f"Epoch {epoch+1}/{epochs} completed in {epoch_duration:.2f} seconds.")
        total_training_time = time.time() - start_time
        print(f"Total Training Time: {total_training_time:.2f} seconds")
        
        print("Global model updated with FedIOD and multi-objective optimization.")
        weights_received.clear()
    except Exception as e:
        print(f"Error in aggregate_and_update: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
