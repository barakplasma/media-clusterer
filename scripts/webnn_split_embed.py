"""Make EmbeddingGemma 2's fp16 text model buildable on WebNN.

Chrome's WebNN rejects the text model at graph build ("Buffer size is over
inline limit") because of one 268 MB constant, the token-embedding table; every
other weight is under 16 MB. This rewrites only the small graph file: the
table lookup (Gather) becomes a new float16 input, `inputs_embeds`, that the
page fills from the same table in JS. The weights file on Hugging Face is
reused unchanged, since every other initializer keeps its offset in it.

Usage:
    pip install onnx
    python -I scripts/webnn_split_embed.py
Writes public/models/embeddinggemma2-webnn/model_fp16_noembed.onnx and prints
the table's offset/length in model_fp16.onnx_data (src/npuTest.ts EMBED_*).
"""
import urllib.request
from pathlib import Path

import onnx
from onnx import TensorProto, helper

SRC = "https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/main/onnx/model_fp16.onnx"
DST = Path("public/models/embeddinggemma2-webnn/model_fp16_noembed.onnx")
TABLE = "model.embed_tokens.weight"


def main() -> None:
    with urllib.request.urlopen(SRC) as r:
        model = onnx.load_from_string(r.read())  # external data stays external
    g = model.graph

    gather = next(n for n in g.node if n.op_type == "Gather" and n.input[0] == TABLE)
    table = next(t for t in g.initializer if t.name == TABLE)
    ext = {kv.key: kv.value for kv in table.external_data}
    rows, width = table.dims
    assert table.data_type == TensorProto.FLOAT16, table.data_type

    # The Gather's output name becomes a graph input, so every consumer is untouched.
    out = gather.output[0]
    g.node.remove(gather)
    g.initializer.remove(table)
    for vi in list(g.value_info):
        if vi.name == out:
            g.value_info.remove(vi)
    g.input.append(helper.make_tensor_value_info(out, TensorProto.FLOAT16, ["batch_size", "sequence_length", width]))
    # Friendlier name for callers.
    for n in g.node:
        n.input[:] = ["inputs_embeds" if i == out else i for i in n.input]
    g.input[-1].name = "inputs_embeds"

    assert not any(n.op_type == "Gather" and n.input[0] == TABLE for n in g.node)
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_bytes(model.SerializeToString())
    print(f"wrote {DST} ({DST.stat().st_size} bytes)")
    print(f"table {TABLE} [{rows}, {width}] fp16: file={ext['location']} offset={ext['offset']} length={ext['length']}")


if __name__ == "__main__":
    main()
