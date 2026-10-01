"""Lossless chunked transfer of model files through RunPod's /stream API.

Same wire protocol as the TRELLIS v1 worker (trellis-files-v1), generalized to
an arbitrary set of files: TRELLIS.2 produces a mesh only, no gaussian PLY.
"""
import base64
import gzip
import hashlib

STREAM_PROTOCOL = "trellis-files-v1"
CHUNK_BYTES = 256 * 1024
ALLOWED_KINDS = ("ply", "glb")


def stream_files(files):
    """Yield manifest, chunk and complete events; each stays below RunPod's 1 MB limit."""
    files = {kind: data for kind, data in files.items() if data}
    if not files:
        raise ValueError("TRELLIS.2 no generó ningún archivo.")
    unknown = [kind for kind in files if kind not in ALLOWED_KINDS]
    if unknown:
        raise ValueError(f"Tipo de archivo no admitido por el protocolo: {unknown}")
    manifest = {
        kind: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
               "chunks": (len(data) + CHUNK_BYTES - 1) // CHUNK_BYTES}
        for kind, data in files.items()
    }
    yield {"protocol": STREAM_PROTOCOL, "type": "manifest", "files": manifest,
           "chunk_bytes": CHUNK_BYTES}
    for kind, data in files.items():
        for index, offset in enumerate(range(0, len(data), CHUNK_BYTES)):
            chunk = gzip.compress(data[offset:offset + CHUNK_BYTES], compresslevel=6, mtime=0)
            yield {"protocol": STREAM_PROTOCOL, "type": "chunk", "file": kind,
                   "index": index, "data": base64.b64encode(chunk).decode("ascii")}
    yield {"protocol": STREAM_PROTOCOL, "type": "complete"}
