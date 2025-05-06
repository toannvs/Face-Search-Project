# Dockerfile
FROM python:3.9-slim

# Cài đặt các công cụ cần thiết, bao gồm CMake
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean

# Sao chép và cài đặt các gói Python
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]