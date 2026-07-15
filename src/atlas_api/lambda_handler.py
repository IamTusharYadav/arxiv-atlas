import json
import os
from pathlib import Path
from typing import Any

import boto3
from mangum import Mangum
from qdrant_client import QdrantClient

from atlas_agents.bedrock import BedrockClient
from atlas_api.app import create_app, run_job
from atlas_api.dynamo import DynamoJobStore, dynamo_backends
from atlas_core.config import Settings, setup_logging
from atlas_core.embedding import OnnxEmbedder
from atlas_core.vectorstore import QdrantStore


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


# Built once per cold start, reused across warm invocations.
_settings = Settings.from_env()
setup_logging(_settings.log_level)
_store = QdrantStore(QdrantClient(url=_settings.qdrant_url, api_key=_settings.qdrant_api_key))
_onnx_dir = Path(_require("ATLAS_ONNX_DIR"))
_embedder = OnnxEmbedder(_onnx_dir / "model_quantized.onnx", _onnx_dir / "tokenizer.json")
_client = BedrockClient()
_dynamodb = boto3.resource("dynamodb")
_limiter, _cache, _budget = dynamo_backends(
    _dynamodb,
    bucket_table=_require("ATLAS_BUCKET_TABLE"),
    cache_table=_require("ATLAS_CACHE_TABLE"),
    budget_table=_require("ATLAS_BUDGET_TABLE"),
)
_jobs = DynamoJobStore(_dynamodb.Table(_require("ATLAS_JOBS_TABLE")))


def _dispatch(job_id: str) -> None:
    # Fire-and-forget self-invoke: the worker has the full 15 min Lambda budget, unbound by the
    # 30s API Gateway limit the request path is stuck with.
    name = os.environ["AWS_LAMBDA_FUNCTION_NAME"]
    # Target the alias, not the bare name: bare resolves to $LATEST, so after a rollback the API
    # would serve the old version while workers still ran the rolled-back code. The retry cap
    # (EventInvokeConfig) also hangs off the alias qualifier, not the function. Unset off Lambda
    # (local RIE, tests), where there is no alias to speak of.
    alias = os.environ.get("ATLAS_LAMBDA_ALIAS")
    boto3.client("lambda").invoke(
        FunctionName=f"{name}:{alias}" if alias else name,
        InvocationType="Event",
        Payload=json.dumps({"job": {"id": job_id}}).encode(),
    )


_app = create_app(
    store=_store,
    embedder=_embedder,
    client=_client,
    limiter=_limiter,
    cache=_cache,
    budget=_budget,
    jobs=_jobs,
    dispatch=_dispatch,
)
_mangum = Mangum(_app)


def handler(event: dict[str, Any], context: Any) -> Any:
    job = event.get("job") if isinstance(event, dict) else None
    if job:
        run_job(
            job["id"],
            jobs=_jobs,
            client=_client,
            store=_store,
            embedder=_embedder,
            cache=_cache,
            budget=_budget,
        )
        return {"ok": True}
    return _mangum(event, context)
