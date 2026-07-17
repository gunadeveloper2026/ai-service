FROM python:3.11

WORKDIR /app

# Install minimal system dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=6000
EXPOSE 6000

CMD ["gunicorn", "--bind", "0.0.0.0:6000", "app:app"]
