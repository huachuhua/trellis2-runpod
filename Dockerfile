# Worker RunPod Serverless con Microsoft TRELLIS.2 (malla + texturas PBR)
# Sigue el setup.sh oficial (PyTorch 2.6.0 + CUDA 12.4), sin conda ni gradio.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Commit de microsoft/TRELLIS.2 contra el que se escribió handler.py.
ARG TRELLIS2_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd
# 1 = pesos públicos dentro de la imagen ($0 en reposo, imagen de ~30 GB).
# 0 = imagen liviana; el worker los baja al volumen de red en el primer arranque.
ARG BAKE_WEIGHTS=1

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    HF_HOME=/opt/hf \
    ATTN_BACKEND=flash_attn \
    SPARSE_CONV_BACKEND=flex_gemm \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0+PTX"

WORKDIR /app

# 1. Dependencias del sistema (compiladores y librerías gráficas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    ninja-build \
    cmake \
    libjpeg-dev \
    libegl1 \
    libgl1 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Dependencias de Python (--basic del setup.sh, sin gradio/tensorboard/lpips,
#    y con pillow en lugar de pillow-simd). einops lo pide el código de BiRefNet.
RUN pip install --no-cache-dir \
    imageio \
    imageio-ffmpeg \
    tqdm \
    easydict \
    opencv-python-headless \
    ninja \
    trimesh \
    "transformers>=4.56" \
    pandas \
    zstandard \
    kornia \
    timm \
    einops \
    plyfile \
    safetensors \
    pillow \
    runpod \
    huggingface_hub \
    && pip install --no-cache-dir git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# 3. flash-attn: rueda precompilada para torch 2.6 + cu12 + cp311
RUN pip install --no-cache-dir flash-attn==2.7.3

# 4. Extensiones CUDA compiladas (mismo orden y ramas que el setup.sh)
RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/ext/nvdiffrast && \
    pip install --no-cache-dir --no-build-isolation /tmp/ext/nvdiffrast && \
    rm -rf /tmp/ext

RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/ext/nvdiffrec && \
    pip install --no-cache-dir --no-build-isolation /tmp/ext/nvdiffrec && \
    rm -rf /tmp/ext

RUN git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/ext/CuMesh && \
    pip install --no-cache-dir --no-build-isolation /tmp/ext/CuMesh && \
    rm -rf /tmp/ext

RUN git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/ext/FlexGEMM && \
    pip install --no-cache-dir --no-build-isolation /tmp/ext/FlexGEMM && \
    rm -rf /tmp/ext

# 5. Código de TRELLIS.2 en el commit fijado + o-voxel (trae eigen como submódulo)
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git /tmp/trellis2_repo && \
    git -C /tmp/trellis2_repo checkout ${TRELLIS2_COMMIT} && \
    git -C /tmp/trellis2_repo submodule update --init --recursive && \
    cp -r /tmp/trellis2_repo/trellis2 /app/trellis2 && \
    pip install --no-cache-dir --no-build-isolation /tmp/trellis2_repo/o-voxel && \
    rm -rf /tmp/trellis2_repo

# 6. Modelos públicos chicos: quita-fondo BiRefNet (MIT) y el decodificador de
#    estructura que TRELLIS.2 toma de TRELLIS v1.
RUN python -c "from huggingface_hub import snapshot_download, hf_hub_download; \
snapshot_download('ZhengPeng7/BiRefNet'); \
[hf_hub_download('microsoft/TRELLIS-image-large', 'ckpts/ss_dec_conv3d_16l8_fp16' + e) for e in ('.json', '.safetensors')]"

# 7. Pesos de TRELLIS.2 (~14,8 GB) en tres capas: ghcr.io rechaza capas de más de 10 GB.
#    DINOv3 NO se hornea: es de acceso restringido y esta imagen es pública.
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then python -c "from huggingface_hub import snapshot_download; \
snapshot_download('microsoft/TRELLIS.2-4B', local_dir='/models/TRELLIS.2-4B', allow_patterns=[ \
'pipeline.json', 'ckpts/ss_flow_img_dit_1_3B_64_bf16.*', \
'ckpts/shape_dec_next_dc_f16c32_fp16.*', 'ckpts/tex_dec_next_dc_f16c32_fp16.*'])"; fi
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then python -c "from huggingface_hub import snapshot_download; \
snapshot_download('microsoft/TRELLIS.2-4B', local_dir='/models/TRELLIS.2-4B', allow_patterns=[ \
'ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.*', 'ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.*'])"; fi
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then python -c "from huggingface_hub import snapshot_download; \
snapshot_download('microsoft/TRELLIS.2-4B', local_dir='/models/TRELLIS.2-4B', allow_patterns=[ \
'ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.*', 'ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.*'])" \
    && rm -rf /models/TRELLIS.2-4B/.cache; fi

# 8. Comprobar las extensiones compiladas. flex_gemm (y o_voxel, que lo importa) no se
#    pueden importar sin GPU: sus kernels de triton piden un driver activo al cargarse.
#    Para esos dos solo se comprueba que quedaron instalados.
RUN python -c "import importlib.util as u, flash_attn, nvdiffrast.torch, cumesh; \
assert all(u.find_spec(m) for m in ('flex_gemm', 'o_voxel')); print('extensiones OK')"

# 9. Handler de RunPod
COPY handler.py result_transport.py ./

CMD ["python", "-u", "handler.py"]
