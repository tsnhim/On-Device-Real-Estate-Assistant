# Experiment Log

## Retained artifact stage

The current repository keeps the following artifact families as the active pipeline:

- `FLAN-T5` seq2seq checkpoint in `02_Models/PyTorch/flan_t5_zillow_final1/`
- optimized `FLAN-T5` binaries in FP16 and dynamic INT8 formats
- `Whisper` speech-recognition checkpoints in `02_Models/Speech/whisper_model/`
- retrieval assets in `03_Data/Retrieval/`

## Current limitation

The exact notebook or script lineage for the retained FLAN-T5 artifact stage is not fully preserved in the committed repo, so the remaining work is centered on reproducibility and benchmarking rather than historical experiment tracing.
