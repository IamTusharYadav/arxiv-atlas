"""Export bge-small-en-v1.5 to an INT8 ONNX model for the Lambda query path.

    uv run --with 'optimum[onnxruntime]' python scripts/export_onnx.py [out_dir]

Writes model_quantized.onnx and tokenizer.json into out_dir (default: models/). Run at build
time (the Dockerfile builder stage); the artifact is never committed. Backend parity against the
torch embedder is pinned by tests/model/test_embedding_contract.py.
"""

import sys
from pathlib import Path

from atlas_core.embedding import CONTRACT


def main(out_dir: str = "models") -> None:
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = ORTModelForFeatureExtraction.from_pretrained(CONTRACT.model_id, export=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(CONTRACT.model_id).save_pretrained(out)

    # Dynamic INT8 for x86 Lambda; swap for AutoQuantizationConfig.arm64 if the function moves
    # to Graviton.
    quantizer = ORTQuantizer.from_pretrained(out)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=out, quantization_config=qconfig)
    print(f"wrote INT8 onnx export to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
