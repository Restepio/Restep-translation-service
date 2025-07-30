# Use the official RunPod Pytorch image
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script to download the model and run it
COPY download_model.py .
RUN python download_model.py

# Copy the handler file
COPY rp_handler.py .

# Set the RUNPOD_HANDLER environment variable
ENV RUNPOD_HANDLER=rp_handler.handler

# Run the handler script when the container launches
CMD ["python", "-m", "runpod.serverless.worker"]
