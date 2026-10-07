# Efficiency benchmark

Script: [`scripts/benchmark_latency.py`](../scripts/benchmark_latency.py). Raw data: [`results/latency/rtx4090.json`](../results/latency/rtx4090.json).

Protocol: one RTX 4090 24GB, fp32 weights with bf16 autocast, CUDA-synchronized timing. 100 warm-up and 1,000 measured steps sampled (seed 0) from test_idd inputs; labels are not used. One decision asks the operation question and the target question over all candidates in a single forward pass. Data loading is excluded; "end-to-end" adds tokenization. Software: laya 0.3.28, transformers 4.57.6, torch 2.14.1+cu130.

## Interactive latency (batch 1)

| Hardware | Precision | Median | p95 | Peak VRAM |
|---|---|---:|---:|---:|
| RTX 4090 | bf16 autocast | 39.7 ms | 43.5 ms | 1.66 GB |

End-to-end median including tokenization: see `e2e_ms` in the raw file. A previous run of the same script gave a 38.3 ms median; expect ±1–2 ms between runs.

## By number of candidates

| Candidates | n | Median | p95 |
|---|---:|---:|---:|
| 1-5 | 102 | 39.7 ms | 42.1 ms |
| 6-10 | 171 | 39.5 ms | 42.1 ms |
| 11-20 | 406 | 39.6 ms | 41.5 ms |
| 21-50 | 295 | 40.3 ms | 43.9 ms |
| >50 | 21 | 49.0 ms | 78.6 ms |

![Accuracy and latency vs number of candidates](../figures/candidate_scaling.png)

## By input length

| Input tokens | n | Median | p95 |
|---|---:|---:|---:|
| <=512 | 649 | 39.6 ms | 41.8 ms |
| 513-1024 | 292 | 39.9 ms | 42.5 ms |
| 1025-1536 | 49 | 42.9 ms | 48.4 ms |
| >1536 | 10 | 53.1 ms | 105.5 ms |

Latency is flat up to ~1,024 tokens: at batch 1 the forward pass is dominated by fixed overhead, not sequence length.

## By batch size

| Batch | Median batch latency | p95 | Throughput | Peak VRAM allocated | Peak VRAM reserved |
|---:|---:|---:|---:|---:|---:|
| 1 | 39.1 ms | 42.3 ms | 25.3 steps/s | 1.66 GB | 1.91 GB |
| 4 | 40.0 ms | 67.4 ms | 90.4 steps/s | 2.30 GB | 3.45 GB |
| 8 | 71.3 ms | 158.1 ms | 98.7 steps/s | 3.16 GB | 6.30 GB |
| 16 | 177.9 ms | 371.3 ms | 80.2 steps/s | 4.89 GB | 11.99 GB |
| 32 | 431.1 ms | 827.8 ms | 58.3 steps/s | 8.34 GB | 23.35 GB |

Throughput peaks around batch 8. Larger batches lose throughput because every step in a batch is padded to the longest one (inputs are not length-sorted here); length bucketing would help offline serving.

External models are not timed on this hardware, so no cross-model latency comparison is made.
