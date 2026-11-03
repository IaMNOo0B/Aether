FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /workspace
RUN python -m pip install --no-cache-dir pytest
USER 65532:65532
CMD ["python", "-m", "pytest", "-q"]
