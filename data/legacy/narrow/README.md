# Historical narrow-model benchmark inputs

WP1 expects the three measured aggregate benchmark CSVs below:

```text
A100.csv
RTX5000.csv
RTX_6000_Ada.csv
```

These are small non-clinical benchmark tables derived from the historical MONAI/Experiment-Impact-Tracker runs. Simulated files such as `A100_sim.csv` and multimodal/VLM pilot files are intentionally excluded from WP1.

Expected SHA256 checksums:

```text
A100.csv           30ac79634ffdf914c21fbcd7c37b42c877aea0957e1983227f92c8da58f40bb4
RTX5000.csv        e2fcf2759ef3d0c940c13dea7435ec7bee0c094aeb7c88ef8e3c6861c73f5f56
RTX_6000_Ada.csv   1ec0faa176e1a5b982971b674f0a6b891b78ffa2cf057ad070e9268f620809d1
```

The WP1 runner fails closed if any file is absent or has a different checksum. This protects the publication analysis from silent input drift.

The source copies used to define these checksums came from the project's historical `models_gpus/` summaries.
