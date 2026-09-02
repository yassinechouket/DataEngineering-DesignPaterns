# Airflow base image + the handful of libs include/rag/* imports.
#
# Why an image instead of _PIP_ADDITIONAL_REQUIREMENTS: that env var
# reinstalls on every container start and refuses to run as root, which
# breaks airflow-init (it runs as 0:0). Baking the deps in avoids both.
ARG AIRFLOW_IMAGE_NAME=apache/airflow:3.3.1
FROM ${AIRFLOW_IMAGE_NAME}

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Pre-download the local embedding model so tasks never pay the download
# cost — or fail on it. Must match embeddings.EMBEDDING_MODEL / _CACHE_DIR.
ENV FASTEMBED_CACHE_DIR=/home/airflow/.cache/fastembed
RUN python -c "\
from fastembed import TextEmbedding; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='${FASTEMBED_CACHE_DIR}')"