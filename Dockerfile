FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir torch transformers

# Copy model files and code
COPY distilledgpt2_local_training/query_custom_model.py .
COPY distilledgpt2_local_training/recession_model/ ./recession_model/

# Create a simple script to test the model
COPY test_model.py .

# Set the entrypoint
ENTRYPOINT ["python", "test_model.py"]