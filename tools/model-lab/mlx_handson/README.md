# MLX hands-on

Four short exercises that load Qwen3-VL 8B directly with Apple's MLX (via `mlx-vlm`), with no Ollama involved, so you can look inside and change it. Run them from `~/memoria`. Results show up on the dashboard's **MLX hands-on** page (http://127.0.0.1:8765/mlx), next to each script's source.

```bash
.venv/bin/python tools/model-lab/mlx_handson/01_load_and_inspect.py    # what's in the box
.venv/bin/python tools/model-lab/mlx_handson/02_quantize.py            # make 8/4/3/2-bit copies yourself
.venv/bin/python tools/model-lab/mlx_handson/03_weights_up_close.py    # real numbers, and what 4-bit does to them
.venv/bin/python tools/model-lab/mlx_handson/04_change_the_weights.py  # noise, layer knock-outs, a "blind" model
```

The first run needs the full-precision model, `mlx-community/Qwen3-VL-8B-Instruct-bf16` (17.6 GB, Apache 2.0), downloaded with `.venv/bin/hf download mlx-community/Qwen3-VL-8B-Instruct-bf16`. Quantized copies go in `data/model-lab/mlx/models/` (about 24 GB for all four); delete that folder when you're done.
