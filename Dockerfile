FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive


# install Python and venv + build deps
RUN apt-get update && apt-get install -y \
  python3 python3-venv python3-pip build-essential git wget ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy everything into the container
COPY requirements.txt /app/requirements.txt

# create virtualenv, activate it and install requirements into it
RUN python3 -m venv /opt/venv \
  && /opt/venv/bin/pip install --upgrade pip \
  && /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt


ENV MODEL_PATH=jimjunior/event-diffusion-model
EXPOSE 8000

# Command to start your app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
