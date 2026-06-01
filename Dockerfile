FROM python:3.11-slim

# System dependencies required by nautilus_trader (Rust-compiled wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY cyber_trader/ ./cyber_trader/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY pyproject.toml .

# Install the package
RUN pip install --no-cache-dir -e .

# Data catalog mount point
VOLUME ["/app/data"]

# Default: paper trading
CMD ["python", "scripts/run_paper.py", "--config", "config/paper_trading.yaml"]
