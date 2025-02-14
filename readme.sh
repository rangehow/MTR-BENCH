# MTR: A Multi-turn Dialogue Retrieval Benchmark for Modern Industrial Applications

This repository hosts the **MTR** (Multi-turn Dialogue Retrieval) benchmark, designed to facilitate research in dialogue systems and retrieval models. MTR is a benchmark tailored for modern industrial applications, where efficient, multi-turn dialogue retrieval plays a key role in providing contextually relevant responses in real-world scenarios.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Benchmarking](#benchmarking)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Citations](#citations)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

The MTR benchmark provides a challenging dataset designed to evaluate the performance of dialogue retrieval models in multi-turn settings. It is particularly aimed at addressing the requirements of modern industrial applications, where dialogue systems must efficiently handle real-time conversations with users while considering context and user intent.

Key aspects of the MTR benchmark:
- **Multi-turn dialogues**: Focuses on real-world, multi-turn conversations instead of single-turn interactions.
- **Industrial application relevance**: Includes dialogues from various industry sectors, including customer service, technical support, and e-commerce.
- **Realistic challenges**: Incorporates noisy and incomplete dialogue data typical of industrial applications.

For more details on the methodology and experimental setup, please refer to the paper: [MTR: A Multi-turn Dialogue Retrieval Benchmark for Modern Industrial Applications](#link-to-paper).

## Dataset

The MTR benchmark consists of a large-scale dataset that includes:
- **Multi-turn dialogues**: Conversations spanning several turns between users and service agents.
- **Domain-specific data**: Data collected from various industrial domains to provide real-world context.
- **Annotated responses**: Responses labeled with relevance and quality metrics to facilitate benchmarking.

### Download the Dataset

You can download the MTR dataset from the following link:
- [Dataset Link](#)

Note: Please refer to the LICENSE section for terms of use and citation guidelines.

## Benchmarking

To benchmark your retrieval models on the MTR dataset, you can use the provided evaluation scripts.

### Evaluation Metrics
- **Retrieval accuracy**: Measures how well the model retrieves contextually relevant responses.
- **Relevance scores**: Based on annotated labels for each response in the dataset.

### Benchmarking Script

To evaluate your model, run the following command:
```bash
python evaluate.py --model YOUR_MODEL_PATH --data YOUR_DATASET_PATH
