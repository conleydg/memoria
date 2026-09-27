# Model lab

A test bench for the local model stack (ADR-0003/0009/0010/0021). It samples about 200 assets from the Photos library, runs every model over them into one SQLite file (memoria's schema from `src/memoria/db.py`, plus lab-only comparison tables), and serves a dashboard for looking at the results.

Everything runs on this Mac. Models are served from Ollama on 127.0.0.1 or loaded in-process. The dashboard binds to 127.0.0.1 and loads nothing from the internet. All outputs live under the gitignored `data/model-lab/`.

## Start the dashboard

```bash
tools/model-lab/start.sh     # http://127.0.0.1:8765
```

Pages:
- **Gallery**, filterable by type, year, zero-shot label, sample stratum and aesthetic score.
- **Asset detail**: 30B vs 8B captions, tags and OCR side by side (raw prompt and response included), SigLIP2 zero-shot scores, the timestamped Whisper transcript, Q-Align scores, and per-model timings. Also *How could this be found?* (example queries, with this asset's keyword, vector and hybrid rank and the RRF math) and *More like this*.
- **Search playground**: keyword, vector and hybrid results side by side, with a min-similarity slider (ADR-0026) and switches for the keyword index (30B, 8B or 32B captions) and the vector model (SigLIP2 so400m or giant).
- **Embedding map**: UMAP or PCA of the SigLIP2 vectors, colored by label, year, type or aesthetic score.
- **Compare models**: an automatic retrieval eval (Recall@k and MRR for every keyword index, vector model and hybrid pair) plus caption stats per VLM.
- **Ask**: question answering over the test set (filter by person/year → hybrid retrieval → the 8B or 30B answers from the retrieved items, with citations).
- **MLX hands-on**: four exercises that load Qwen3-VL 8B directly with MLX (see `mlx_handson/`).
- **Model stats**: load time, latency, tok/s, peak memory, and the projected full-library first pass.

## Re-run the pipeline

```bash
tools/model-lab/run_pipeline.sh <ssh-host> "<path to .photoslibrary on that Mac>"
```

Stages: `lab.sample` → `lab.fetch` (ADR-0027 RemoteLibrary) → `lab.prepare` (JPEG derivatives, 4 video keyframes, H.264 preview, 16 kHz audio) → `lab.people` (named people, local time, GPS from Photos) → `lab.vlm` for 30B then 8B (unloaded afterwards with `keep_alive: 0`) → `lab.siglip` → `lab.whisper_run` → `lab/qalign.py` → `lab.index --queries`.

Q-Align runs in its own venv because OneAlign's remote code needs transformers 4.36:

```bash
/opt/homebrew/bin/python3.12 -m venv tools/model-lab/.venv-qalign
tools/model-lab/.venv-qalign/bin/pip install "transformers==4.36.1" torch "accelerate<0.30" sentencepiece protobuf pillow psutil icecream einops
```

## Findings from the first run (2026-09-26)

306 assets processed: a 202-item stratified sample (191 fetched) plus 4 complete events (Photos moments, 115 items) so similarity search has real neighbours.


- 11 of 202 sampled originals weren't on the library Mac. All 11 had `ZCLOUDLOCALSTATE = 2` (iCloud-only), as do 8,437 assets library-wide. ADR-0017/0027's "originals fully local" assumption doesn't quite hold yet.
- Greedy decoding sent 4 VLM outputs (out of 382) into a repetition loop until the token cap. A retry with `repeat_penalty 1.15` and at most 20 tags fixed all of them. The stored prompt marks those rows.
- The 30B-A3B mixture-of-experts writes faster than the dense 8B (about 116 vs 85 tok/s) because only about 3B parameters are active per token.
- Q-Align's peak memory (about 23 GB with fp16 on MPS) is higher than the 30B's (about 19 GB).
- Similarity works when there's something similar to find. 75% of an event photo's top-5 neighbours are from the same event, and the median nearest-neighbour cosine is 0.98 for event items vs 0.78 for random-sample items.
- Names come only from Photos (ADR-0015). On name-only queries, keyword search puts the tagged photo in the top 10 far more often than vector search does, because SigLIP2 has never seen these people.
- Larger models (overnight run, 306 assets). Qwen3-VL 32B dense: 10.9 s per photo vs 2.1 s for the 30B-A3B (24 vs 115 tok/s), with 25% longer captions and 3 more tags on average. A full-library pass would take about 394 h vs 83 h. SigLIP2 giant (1536-d): retrieval equal to so400m on this set (MRR 0.673 vs 0.675), with the same share of same-event neighbours. Keyword recall on the 32B index is lower, but the example queries are written from 30B captions, so that comparison is biased and inconclusive. Caption quality needs human judgement (ADR-0006).
- A first hint for the ADR-0026 threshold: real matches reach about 0.10 SigLIP2 cosine, while a query for something absent tops out around 0.04.
- Video frames (44 test videos, 30B-A3B). Default 4 fixed frames: 5.5K prompt tokens, 6.6 s per video. SigLIP2 content-change sampling (4-12, 6 on average): 6.4K tokens, 7.5 s. Dense 8-16 frames: 9.4K tokens, 11.6 s. Classic scene-cut detection finds almost nothing in home videos (one continuous take), and Ollama's Qwen3-VL gives every image at least ~1,000 tokens, so smaller frames don't save tokens.
- MLX quantization of Qwen3-VL 8B (your own conversions): bf16 17.6 GB / 36 tok/s, 8-bit 9.9 GB / 65, 4-bit 5.8 GB / 110, 3-bit 4.8 GB / 128, 2-bit 3.7 GB / 163. Caption meaning kept vs bf16 (SigLIP2 text similarity): 0.98 / 0.85 / 0.78 / 0.19. 2-bit breaks the model.
- Weight edits: noise up to 5% of each matrix's spread barely changes captions (0.92+); removing layer 0's MLP breaks it (0.42) while a middle or last layer barely matters (0.84-0.88); zeroing the vision projector makes it blind (0.30).
