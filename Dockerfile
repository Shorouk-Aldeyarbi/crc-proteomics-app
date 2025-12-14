FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for matplotlib, seaborn)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port (Hugging Face uses 7860)
EXPOSE 7860

# Start Flask app with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
