FROM python:3.11-slim
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y graphviz && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torchlens==0.3.1 pydantic graphviz

WORKDIR /app
COPY worker.py /app/worker.py

ENTRYPOINT ["python"]
