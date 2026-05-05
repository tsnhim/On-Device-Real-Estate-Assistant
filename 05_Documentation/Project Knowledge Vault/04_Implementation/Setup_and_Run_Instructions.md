# Setup and Run Instructions

## Current state

The repo currently preserves model and retrieval artifacts, but it does not yet include one clean scripted runtime for the full FLAN-T5 plus Whisper pipeline.

## Minimal inspection workflow

1. Inspect `02_Models/PyTorch/flan_t5_zillow_final1/` for the main text model.
2. Inspect `02_Models/Optimized/` for lighter FLAN-T5 deployment artifacts.
3. Inspect `02_Models/Speech/whisper_model/` for the retained speech-recognition baseline.
4. Inspect `03_Data/Retrieval/` for the retrieval corpus and FAISS index.

## Current gaps

- no top-level inference script
- no committed FLAN-T5 conversion script
- no committed end-to-end speech-to-answer demo
- no benchmark script for local inference
