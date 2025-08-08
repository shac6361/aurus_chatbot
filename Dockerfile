FROM python:3.12-slim

# 🛠 Install Rust (includes cargo)
RUN apt-get update && apt-get install -y curl build-essential pkg-config libssl-dev zlib1g-dev clang \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y

# ✅ Add Rust binaries to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# 🚀 Install uv
RUN cargo install --git https://github.com/astral-sh/uv --tag 0.8.3 uv

# 🔍 Confirm uv works
RUN uv --version && which uv