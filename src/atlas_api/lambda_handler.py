import os
from pathlib import Path

import boto3
from fastapi import FastAPI
from mangum import Mangum
from qdrant_client import QdrantClient

from atlas_agents.bedrock import BedrockClient
from atlas_api.app import create_app
from atlas_api.dynamo import dynamo_backends
from atlas_core.config import Settings, setup_logging
from atlas_core.embedding import OnnxEmbedder
from atlas_core.vectorstore import QdrantStore


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def build_app() -> FastAPI:
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    store = QdrantStore(QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key))
    onnx_dir = Path(_require("ATLAS_ONNX_DIR"))
    embedder = OnnxEmbedder(onnx_dir / "model_quantized.onnx", onnx_dir / "tokenizer.json")
    limiter, cache, budget = dynamo_backends(
        boto3.resource("dynamodb"),
        bucket_table=_require("ATLAS_BUCKET_TABLE"),
        cache_table=_require("ATLAS_CACHE_TABLE"),
        budget_table=_require("ATLAS_BUDGET_TABLE"),
    )
    return create_app(
        store=store,
        embedder=embedder,
        client=BedrockClient(),
        limiter=limiter,
        cache=cache,
        budget=budget,
    )


# Built once per cold start, reused across warm invocations. SAM points at this symbol.
handler = Mangum(build_app())
