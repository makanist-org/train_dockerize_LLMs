# Recession Model - LLM Training and Deployment

This repository contains code for training and deploying a custom language model focused on economic recession topics.

## Repository Structure

```
train_LLM_model/
├── distilledgpt2_local_training/
│   ├── recession_model/        # Trained model files (not tracked in Git)
│   ├── historical_recessions.csv
│   ├── query_custom_model.py   # Script to query the model locally
│   ├── train_gemma3.py         # Training script
│   ├── training_data.jsonl     # Training data
│   └── training.log
├── distilledgpt2_train_venv/   # Virtual environment
├── Dockerfile                  # For containerizing the model
├── test_model.py               # Test script for Docker container
├── build_and_run.sh            # Helper script for Docker
└── .gitignore                  # Ignores large model files
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Training Pipeline                    Deployment Pipeline   │
│  ┌───────────────┐                   ┌──────────────────┐   │
│  │ Training Data │                   │                  │   │
│  │  (JSONL)      │───┐           ┌──▶│  Docker Image    │   │
│  └───────────────┘   │           │   │                  │   │
│                      ▼           │   └──────────────────┘   │
│  ┌───────────────┐   ┌───────────┴───┐                      │
│  │ Base Model    │──▶│ Custom Trained │                     │
│  │ (DistilGPT2)  │   │ Model          │                     │
│  └───────────────┘   └───────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Usage Instructions

### Local Development

1. **Set up the environment**:
   ```bash
   # Activate the virtual environment
   source distilledgpt2_train_venv/bin/activate
   ```

2. **Query the model locally**:
   ```bash
   cd distilledgpt2_local_training
   python query_custom_model.py "What are the signs of an economic recession?"
   ```

### Docker Deployment

1. **Build and run the Docker container**:
   ```bash
   # Build the container and run a test query
   ./build_and_run.sh
   ```

2. **Run custom queries**:
   ```bash
   docker run --rm recession-model "How long do recessions typically last?"
   ```

## Model Training

The model was fine-tuned from DistilGPT2 using economic recession data:

1. **Prepare training data** in JSONL format
2. **Run the training script**:
   ```bash
   python train_gemma3.py
   ```

## Notes

- Large model files are excluded from Git tracking using `.gitignore`
- The Docker container provides a lightweight way to deploy and test the model
- For production use, consider implementing a proper API server