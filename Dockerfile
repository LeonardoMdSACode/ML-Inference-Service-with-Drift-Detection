# Use official Python slim image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (build-essential needed for some packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt requirements-dev.txt ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the entire repo
COPY . .

# Expose the port HF Spaces expects
EXPOSE 7860

# Environment variable for Uvicorn to bind to all interfaces
ENV HOST=0.0.0.0
ENV PORT=7860

# Ensure necessary directories exist
RUN mkdir -p data/processed data/production logs reports/evidently models/v1 models/v2

# Use uvicorn with a single process (HF Spaces limitation)
# --host 0.0.0.0 ensures it binds to the container's interface
# --port 7860 is default for Spaces
# --reload is optional, remove for production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
