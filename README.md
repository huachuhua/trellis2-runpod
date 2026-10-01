# Worker RunPod Serverless: Microsoft TRELLIS.2

Empaqueta **Microsoft TRELLIS.2** (`microsoft/TRELLIS.2-4B`) para **RunPod Serverless**.
Convive con el worker de TRELLIS v1 (`../worker-trellis/`): es otro endpoint, no un reemplazo.

> **Estado: escrito, sin construir ni probar en GPU.** La imagen nunca se compiló y el
> handler nunca corrió. Lo único verificado es el transporte (`test_result_transport.py`)
> y que el código coincide con el repo oficial en el commit fijado en el `Dockerfile`.

## Qué entrega

Un `.glb` con malla y texturas PBR (color, metálico/rugosidad) a partir de **una** imagen.
No entrega gaussian splats (`.ply`): TRELLIS.2 no los genera. Tampoco acepta varias vistas.

## Entrada (`input`)

| Campo | Default | Valores |
|---|---|---|
| `image` | — (obligatorio) | base64 o data URL. Si trae transparencia, se usa ese recorte y no corre el quita-fondo |
| `output_transport` | — (obligatorio) | `trellis-files-v1` |
| `seed` | `42` | entero |
| `pipeline_type` | `1024_cascade` | `512`, `1024`, `1024_cascade`, `1536_cascade` |
| `texture_size` | `4096` | `1024`, `2048`, `4096` |
| `decimation_target` | `1000000` | caras de la malla exportada (10 000 – 2 000 000) |
| `texture_format` | `webp` | `webp` (GLB chico, extensión `EXT_texture_webp`) o `png` |
| `remesh` | `true` | bool |

Salida por `/stream/{jobId}`: el mismo protocolo por fragmentos del worker v1
(`trellis-files-v1`: manifiesto con SHA-256, fragmentos gzip de 256 KiB, `complete`),
con **un solo archivo, `glb`**. `/status` no contiene el modelo.

⚠️ **El backend de SB-176 todavía no sabe usar este worker**: `runpod-stream.mjs`
rechaza manifiestos sin `ply`, y `backend.mjs` no tiene el motor `trellis2` ni su
endpoint en la configuración. Hay que agregarlo antes de generar desde la app.

Diagnóstico sin inferencia: `input.transport_probe_bytes` (1–80 MiB) manda datos
aleatorios por el mismo protocolo.

## Modelos y licencias

| Modelo | Dónde vive | Notas |
|---|---|---|
| `microsoft/TRELLIS.2-4B` (~14,8 GB usados) | Horneado en la imagen | MIT. Solo los pesos de inferencia, sin los codificadores |
| `microsoft/TRELLIS-image-large` (un decodificador, 150 MB) | Horneado | TRELLIS.2 lo reutiliza de v1 |
| `ZhengPeng7/BiRefNet` (440 MB) | Horneado | MIT. Reemplaza a `briaai/RMBG-2.0`, que es de acceso restringido y licencia no comercial |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` (1,2 GB) | **Se descarga al arrancar** | Acceso restringido con aprobación manual de Meta. No se hornea: la imagen es pública |

El handler escribe una copia del `pipeline.json` oficial (`pipeline.sb.json`) con esos
dos cambios: DINOv3 desde la carpeta local y BiRefNet como quita-fondo.

## Despliegue

1. **Repositorio en GitHub** (p.ej. `trellis2-runpod`) con el contenido de esta carpeta.
   GitHub Actions compila y publica `ghcr.io/<usuario>/trellis2-runpod:latest`.
   El paquete de ghcr tiene que ser público, o RunPod necesita credenciales del registro.
2. **Endpoint Serverless en RunPod**:
   - Container Image: `ghcr.io/<usuario>/trellis2-runpod:latest`
   - GPU: **48 GB** (A6000 / A40 / L40S). El mínimo oficial es 24 GB, pero solo está
     probado en A100/H100; con menos de 40 GB el worker activa `low_vram` solo (más lento).
   - Container Disk: 40 GB o más.
   - Workers Min: 0 · Idle Timeout: 60 s.
   - Variable de entorno **`HF_TOKEN`**: token de lectura de una cuenta de Hugging Face
     con el acceso a DINOv3 **aprobado**. Cargarla como secreto de RunPod, no en texto plano.
3. **Second Brain**: pendiente (ver aviso de arriba).

### Variables opcionales

- `TRELLIS2_LOW_VRAM`: `auto` (default: activo si la GPU tiene menos de 40 GB), `1`, `0`.

### Volumen de red (opcional)

Si el endpoint tiene un volumen en `/runpod-volume`, DINOv3 se guarda ahí y no se
vuelve a bajar en cada arranque en frío. Sin volumen se baja cada vez (1,2 GB).
Un volumen cuesta aunque no haya trabajos y ata el endpoint a un centro de datos.

### Si el build de GitHub se queda sin disco

La imagen con pesos ronda los 30 GB y el runner gratuito va justo. Alternativa:
lanzar el workflow a mano con `bake_weights = 0` (imagen liviana) y adjuntar un
volumen de red de 30 GB: el worker baja los pesos de TRELLIS.2 ahí en el primer arranque.

## Puntos sin verificar (revisar primero si algo falla)

- **Versión de `transformers`**: el `setup.sh` oficial no la fija. Hace falta ≥ 4.56
  (DINOv3); una versión muy nueva podría romper el código remoto de BiRefNet.
- **BiRefNet en lugar de RMBG-2.0**: misma arquitectura y misma clase del repo, pero
  no probado. Se evita por completo mandando la imagen ya recortada con transparencia.
- **Compilación de CuMesh / FlexGEMM / o-voxel sin GPU** en el runner de GitHub.
- **Tiempos y VRAM reales** en GPUs de RunPod (los oficiales son de una H100).

## Prueba local sin GPU

```bash
python3 -B -m unittest test_result_transport.py
```
