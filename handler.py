"""
handler.py — Worker Serverless de RunPod para Microsoft TRELLIS.2 (SB-176)

Genera una malla con texturas PBR (.GLB) a partir de UNA imagen. A diferencia
del worker de TRELLIS v1, no produce gaussian splats (.PLY): TRELLIS.2 solo
entrega malla.

Modelos que usa:
  - microsoft/TRELLIS.2-4B            pesos públicos (MIT), horneados en la imagen.
  - facebook/dinov3-vitl16-…          acceso restringido: se descarga al arrancar
                                      con HF_TOKEN (no se hornea: la imagen es
                                      pública y redistribuiría pesos con licencia).
  - ZhengPeng7/BiRefNet               quita-fondo (MIT), en lugar de briaai/RMBG-2.0
                                      (restringido y de licencia no comercial).
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")

import base64
import io
import json
import tempfile
import time
import traceback

import runpod
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from result_transport import stream_files

TRELLIS2_REPO = "microsoft/TRELLIS.2-4B"
DINO_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
REMBG_REPO = "ZhengPeng7/BiRefNet"

BAKED_MODEL_DIR = "/models/TRELLIS.2-4B"
VOLUME_DIR = "/runpod-volume"
PIPELINE_CONFIG = "pipeline.sb.json"

# Solo lo que usa la inferencia: los codificadores (enc) son para entrenar.
WEIGHT_PATTERNS = [
    "pipeline.json",
    "ckpts/ss_flow_img_dit_1_3B_64_bf16.*",
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.*",
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.*",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.*",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.*",
    "ckpts/shape_dec_next_dc_f16c32_fp16.*",
    "ckpts/tex_dec_next_dc_f16c32_fp16.*",
]

PIPELINE_TYPES = ("512", "1024", "1024_cascade", "1536_cascade")
TEXTURE_SIZES = (1024, 2048, 4096)
NVDIFFRAST_MAX_FACES = 16777216

pipeline = None
pipeline_error = None


def _persistent_dir(name):
    """A folder on the network volume if one is attached, else inside the container."""
    base = VOLUME_DIR if os.path.isdir(VOLUME_DIR) else "/models"
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    return path


def ensure_trellis2_weights():
    if os.path.exists(os.path.join(BAKED_MODEL_DIR, "pipeline.json")):
        return BAKED_MODEL_DIR
    # Imagen construida con BAKE_WEIGHTS=0: bajar ~15 GB al volumen (una sola vez).
    target = _persistent_dir("TRELLIS.2-4B")
    print(f"⬇️  Pesos de TRELLIS.2 no horneados; descargando a {target}...", flush=True)
    snapshot_download(TRELLIS2_REPO, local_dir=target, allow_patterns=WEIGHT_PATTERNS)
    return target


def ensure_dino():
    target = _persistent_dir("dinov3-vitl16")
    if os.path.exists(os.path.join(target, "config.json")) and \
            os.path.exists(os.path.join(target, "model.safetensors")):
        return target
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        raise RuntimeError(
            "Falta la variable HF_TOKEN en el endpoint. DINOv3 es de acceso restringido: "
            "se necesita un token de una cuenta de Hugging Face con la licencia de "
            f"{DINO_REPO} aceptada y aprobada."
        )
    print(f"⬇️  Descargando DINOv3 a {target}...", flush=True)
    try:
        snapshot_download(DINO_REPO, local_dir=target,
                          allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"])
    except Exception as e:
        raise RuntimeError(
            f"No se pudo descargar {DINO_REPO}: {type(e).__name__}: {e}. "
            "Si es un 403, la solicitud de acceso a DINOv3 todavía no fue aprobada para esa cuenta."
        ) from e
    return target


def write_pipeline_config(model_dir, dino_dir):
    """Runtime copy of pipeline.json pointing at the local DINOv3 and the ungated background remover."""
    with open(os.path.join(model_dir, "pipeline.json")) as f:
        config = json.load(f)
    config["args"]["image_cond_model"]["args"]["model_name"] = dino_dir
    config["args"]["rembg_model"]["args"]["model_name"] = REMBG_REPO
    config["args"]["low_vram"] = resolve_low_vram()
    with open(os.path.join(model_dir, PIPELINE_CONFIG), "w") as f:
        json.dump(config, f, indent=2)


def resolve_low_vram():
    """low_vram mueve cada modelo a la GPU solo mientras se usa: más lento, cabe en 24 GB."""
    setting = os.environ.get("TRELLIS2_LOW_VRAM", "auto").lower()
    if setting in ("1", "true", "yes"):
        return True
    if setting in ("0", "false", "no"):
        return False
    if not torch.cuda.is_available():
        return True
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return total_gb < 40


def get_pipeline():
    global pipeline, pipeline_error
    if pipeline is not None:
        return pipeline, None
    try:
        started = time.time()
        if not torch.cuda.is_available():
            raise RuntimeError("No hay GPU CUDA disponible en este worker.")
        props = torch.cuda.get_device_properties(0)
        print(f"🚀 Cargando TRELLIS.2 en {torch.cuda.get_device_name(0)} "
              f"({props.total_memory / (1024 ** 3):.1f} GB de VRAM)...", flush=True)

        model_dir = ensure_trellis2_weights()
        dino_dir = ensure_dino()
        write_pipeline_config(model_dir, dino_dir)

        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        pipe = Trellis2ImageTo3DPipeline.from_pretrained(model_dir, PIPELINE_CONFIG)
        pipe.cuda()
        pipeline = pipe
        pipeline_error = None
        print(f"✅ TRELLIS.2 listo en {time.time() - started:.0f} s (low_vram={pipe.low_vram})", flush=True)
    except Exception as e:
        pipeline_error = f"{type(e).__name__}: {e}"
        print(f"❌ Error cargando TRELLIS.2: {pipeline_error}", flush=True)
        traceback.print_exc()
    return pipeline, pipeline_error


def parse_options(job_input):
    pipeline_type = str(job_input.get("pipeline_type", "1024_cascade"))
    if pipeline_type not in PIPELINE_TYPES:
        raise ValueError(f"pipeline_type inválido: {pipeline_type}. Opciones: {', '.join(PIPELINE_TYPES)}")
    texture_size = int(job_input.get("texture_size", 4096))
    if texture_size not in TEXTURE_SIZES:
        raise ValueError(f"texture_size inválido: {texture_size}. Opciones: {TEXTURE_SIZES}")
    decimation_target = int(job_input.get("decimation_target", 1000000))
    if not 10000 <= decimation_target <= 2000000:
        raise ValueError("decimation_target debe estar entre 10000 y 2000000.")
    texture_format = str(job_input.get("texture_format", "webp")).lower()
    if texture_format not in ("webp", "png"):
        raise ValueError("texture_format debe ser 'webp' o 'png'.")
    return {
        "seed": int(job_input.get("seed", 42)),
        "pipeline_type": pipeline_type,
        "texture_size": texture_size,
        "decimation_target": decimation_target,
        "texture_format": texture_format,
        "remesh": bool(job_input.get("remesh", True)),
    }


def decode_image(image_raw):
    if image_raw.startswith("data:image/"):
        image_raw = image_raw.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(image_raw)))
    # Conservar el canal alfa: si la imagen ya viene recortada, TRELLIS.2 usa ese
    # recorte y no corre el quita-fondo.
    return img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")


def handler(job):
    """
    Entrada en job['input']:
    {
        "image": "<base64 o data URL>",       obligatorio
        "output_transport": "trellis-files-v1", obligatorio
        "seed": 42,
        "pipeline_type": "1024_cascade",       512 | 1024 | 1024_cascade | 1536_cascade
        "texture_size": 4096,                  1024 | 2048 | 4096
        "decimation_target": 1000000,          caras de la malla exportada
        "texture_format": "webp",              webp | png
        "remesh": true
    }
    Salida: eventos del protocolo trellis-files-v1 por /stream, con un único archivo "glb".
    """
    job_input = job.get("input", {})
    if "transport_probe_bytes" in job_input:
        size = int(job_input["transport_probe_bytes"])
        if not 1 <= size <= 80 * 1024 * 1024:
            raise ValueError("Tamaño de prueba de transporte inválido")
        yield from stream_files({"glb": os.urandom(size)})
        return
    if job_input.get("output_transport") != "trellis-files-v1":
        yield {"error": "Este worker solo entrega modelos por fragmentos (output_transport: trellis-files-v1)."}
        return

    image_raw = job_input.get("image")
    if not image_raw:
        yield {"error": "No se proporcionó el parámetro 'image' en el input."}
        return

    try:
        options = parse_options(job_input)
        img = decode_image(image_raw)
    except Exception as e:
        yield {"error": str(e)}
        return

    pipe, err = get_pipeline()
    if pipe is None:
        yield {"error": f"El pipeline de TRELLIS.2 no pudo inicializarse: {err}"}
        return

    try:
        import o_voxel

        started = time.time()
        print(f"🎨 Ejecutando TRELLIS.2 {options} sobre imagen {img.size} {img.mode}...", flush=True)
        mesh = pipe.run(img, seed=options["seed"], pipeline_type=options["pipeline_type"])[0]
        mesh.simplify(NVDIFFRAST_MAX_FACES)
        print(f"🧱 Malla generada en {time.time() - started:.0f} s; exportando GLB...", flush=True)

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=options["decimation_target"],
            texture_size=options["texture_size"],
            remesh=options["remesh"],
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
        del mesh
        torch.cuda.empty_cache()

        with tempfile.TemporaryDirectory() as tmp:
            glb_path = os.path.join(tmp, "model.glb")
            glb.export(glb_path, extension_webp=options["texture_format"] == "webp")
            with open(glb_path, "rb") as f:
                glb_bytes = f.read()
        print(f"✨ GLB de {len(glb_bytes)} bytes listo en {time.time() - started:.0f} s en total", flush=True)

        yield from stream_files({"glb": glb_bytes})

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Error durante la inferencia de TRELLIS.2: {e}", flush=True)
        torch.cuda.empty_cache()
        yield {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    # Pre-cargar al arrancar el contenedor: el primer trabajo no paga la carga.
    get_pipeline()
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": False})
