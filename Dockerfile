# ========== BUILDER STAGE ==========
FROM python:3.12-slim AS builder

# Install necessary packages for Python compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV VENV_PATH="/opt/.venv"
RUN python -m venv "${VENV_PATH}"

# Set the path of virtual environment into system path
ENV PATH="${VENV_PATH}/bin:$PATH"

# Copy requirement file and install dependences
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Clean up 
RUN find "${VENV_PATH}" -type d -name "tests" -exec rm -rf {} + 2>/dev/null \
    && find "${VENV_PATH}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null \
    && find "${VENV_PATH}" -type d -name "benchmarks" -exec rm -rf {} + 2>/dev/null \
    && find "${VENV_PATH}" -type f -name "*.pyc" -delete \
    && find "${VENV_PATH}" -type f -name "*.pyo" -delete \
    && find "${VENV_PATH}" -type f -name "*.a" -delete

# ========== RUNTIME STAGE ==========
FROM python:3.12-slim AS runtime

# Clean up system
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Set up environment
ENV VENV_PATH="/opt/.venv"
COPY --from=builder "${VENV_PATH}" "${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:$PATH"

WORKDIR /app
COPY . .

EXPOSE 7002
CMD ["python", "./app/index.py"]