# Slim Python base. Multi-stage builds aren't worth the complexity here.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# OpenMP runtime is needed by torch for CPU threading.
# Without libgomp1, torch on slim images can fail with cryptic errors.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so Docker can cache this layer.
# Changing app.py won't reinstall packages; changing requirements will.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and the trained checkpoint.
COPY . .

# Gradio's default port.
EXPOSE 7860

CMD ["python", "app.py"]