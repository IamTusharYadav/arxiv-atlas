"""Run the API locally against live Qdrant and Bedrock, for frontend work on routes the
deployed stack does not serve yet.

    uv run --with uvicorn python scripts/dev_api.py

Point the frontend at it with VITE_API_PROXY=http://localhost:8000 in frontend/.env.local.
Synchronous path (no job store): a run executes inside the request, so expect 30-120s per
call. No cache, no budget guard, no rate limit here; every call spends real Bedrock money.
Needs .env (Qdrant + AWS credentials) and the exported model in models/ (scripts/export_onnx.py).
"""

from pathlib import Path

import uvicorn
from qdrant_client import QdrantClient

from atlas_agents.bedrock import BedrockClient
from atlas_api import create_app
from atlas_core.config import Settings, setup_logging
from atlas_core.embedding import OnnxEmbedder
from atlas_core.vectorstore import QdrantStore


def main() -> None:
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    models = Path(__file__).resolve().parent.parent / "models"
    app = create_app(
        store=QdrantStore(QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)),
        embedder=OnnxEmbedder(models / "model_quantized.onnx", models / "tokenizer.json"),
        client=BedrockClient(),
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
