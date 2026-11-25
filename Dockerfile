FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as user 1000; give it a writable home
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .

EXPOSE 7860
# API stays internal (127.0.0.1:8000); only Streamlit is public on 7860.
# XSRF protection is off because Spaces serves the app inside an iframe, which
# otherwise blocks file uploads.
CMD ["sh", "-c", "uvicorn api.main:app --host 127.0.0.1 --port 8000 & exec streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true --server.enableXsrfProtection false --server.maxUploadSize 20"]
