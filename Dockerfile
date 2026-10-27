FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
# The service imports the shared core (app.main -> core). Without it the image
# cannot even import the app - that is what broke the image (review P1-1).
COPY core ./core
COPY scripts ./scripts
# The fingerprint of what this image carries (scripts/tree_digest.py): stored here, reported by /health as "build", so a
# clone of the published tag can prove it is what runs. The Dockerfile itself is copied in because the deployed
# configuration lives in its ENV lines.
COPY Dockerfile ./Dockerfile
RUN python scripts/tree_digest.py > BUILD_DIGEST
# PORT: the port uvicorn listens on inside the container. Zeabur routes the public domain to 8080; override with the
# PORT environment variable if a host routes elsewhere. Local runs without Docker are unaffected (README: --port 8000).
ENV DB_PATH=/data/memory.db PORT=8080
# The configuration the answer ruler decided on (see OPERATIONS.md, "Submission deployment"): keyword + meaning-based
# recall, DashScope embeddings, no rerank, no stemming, neighbouring turns on. Written here, not left to code defaults
# or to the host's environment page, so a variable missed on the host cannot turn the meaning-based path off.
ENV RECALL_PATHS=keyword,dense EMBED_PROVIDER=dashscope RERANK_ENABLED=0 STEM_ENGLISH=0 NEIGHBOUR_CANDIDATES=1
# Places at the embedding gate (code default 2). A search never waits for a place: with none free it drops its
# meaning-based half and answers from keywords alone. At 2 places, 16 / 32 simultaneous searches lost that half for
# 94% / 93% of the questions (measured with real provider calls). Raised so every simultaneous search gets a place.
ENV EMBED_MAX_CONCURRENCY=64
# No key lives in the image. The host must set API_KEY and DASHSCOPE_API_KEY; with this switch on, the service refuses
# to start without a real API_KEY (and, with EMBED_PROVIDER=dashscope, without DASHSCOPE_API_KEY), naming the variable.
ENV REQUIRE_REAL_API_KEY=1
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT}\" --workers 1"]