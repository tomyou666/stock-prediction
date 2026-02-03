# マルチビルド-ビルドイメージ
FROM debian:bullseye-slim as builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# マルチビルド-実際に使用するイメージ
FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y \
    git

RUN apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN echo 'source "$HOME/.local/bin/env"' >> ~/.bashrc
RUN echo "export UV_LINK_MODE=copy" >> ~/.bashrc
COPY --from=builder /root/.local/ /root/.local/