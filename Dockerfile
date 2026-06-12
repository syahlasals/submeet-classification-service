FROM python:3.11-slim

WORKDIR /app

# Install system dependencies untuk TensorFlow dan NLTK
RUN apt-get update && apt-get install -y \
  gcc \
  g++ \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements dulu (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data yang dibutuhkan preprocessor
RUN python -m nltk.downloader stopwords punkt

# Copy seluruh kode aplikasi
COPY . .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]