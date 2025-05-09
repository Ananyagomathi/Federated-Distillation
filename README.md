# Federated Distillation with Heterogeneity-Aware Distillation (HAD) Server & Client

## Overview

This repository implements a simple federated-distillation pipeline on the UCI HAR dataset.  
- **Server** (`iodistill+HAD+evalmet.py`)  
  - Hosts a global “student” model (Dense network) and an adversarial GAN (generator + discriminator).  
  - Exposes REST endpoints for clients to fetch public data, submit logits/embeddings, and trigger model aggregation & evaluation.  
  - Implements heterogeneity-aware client selection (HAD), graph-based adaptive weighting, differential privacy, multi-objective optimization, and distillation steps.  
- **Client** (`clientnew1.py`)  
  - Generates synthetic “local” data (features + labels), trains a tiny NumPy-based network, produces logits & embeddings, scores its own performance, and POSTs them to the server.

## Features

- **Public Data Loading**: Samples 10 % of UCI HAR features for all clients.  
- **Local Client Simulation**: Random feature/label generation and a lightweight NumPy model.  
- **Federated Aggregation**:  
  - Heterogeneity-aware selection (threshold on client performance).  
  - Graph-based adaptive weighting of client logits.  
  - Quantization, differential privacy, and intermediate layer distillation.  
- **Adversarial Data Augmentation**:  
  - GAN generator to synthesize new feature vectors.  
  - Discriminator training for realism.  
- **Multi-Objective Global Training**:  
  - Combines accuracy, robustness (via perturbed inputs), and weight-decay efficiency.  
- **Evaluation Endpoint**: Returns accuracy, precision, recall, F1, confusion matrix, training losses, and timing metrics on held-out test split.

## Requirements

- Python 3.8+  
- **Server** dependencies:  
  - `Flask`  
  - `tensorflow`  
  - `pandas`  
  - `scikit-learn`  
  - `numpy`  
- **Client** dependencies:  
  - `requests`  
  - `numpy`  
