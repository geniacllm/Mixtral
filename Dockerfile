FROM asia-northeast1-docker.pkg.dev/pfn-artifactregistry/tmp/eric-mix:v1.0.1

RUN mkdir -p /app/.triton && \
    chmod -R 777 /app/.triton

RUN mkdir -p ~/.triton && \
    chmod -R 777 /app/.triton

RUN mkdir -p /.triton && \
    chmod -R 777 /app/.triton

RUN mkdir -p /.triton/autotune && \
    chmod -R 777 /.triton
    
USER root