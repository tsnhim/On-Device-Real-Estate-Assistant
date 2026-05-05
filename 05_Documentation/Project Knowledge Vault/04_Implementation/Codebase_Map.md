# Codebase Map

## Top-level view

- `02_Models/PyTorch/flan_t5_zillow_final1/`
  - main FLAN-T5 seq2seq checkpoint
- `02_Models/Optimized/flan_t5_fp16.bin`
  - optimized half-precision FLAN-T5 artifact
- `02_Models/Optimized/flan_t5_dynamic_int8.bin`
  - optimized dynamic INT8 FLAN-T5 artifact
- `02_Models/Speech/whisper_model/`
  - quantized Whisper ASR models
- `03_Data/Retrieval/zillow_records.json`
  - structured real-estate QA records for retrieval
- `03_Data/Retrieval/zillow_faiss.index`
  - FAISS index built over the retrieval corpus
- `05_Documentation/Project Knowledge Vault/`
  - focused documentation for the retained pipeline

## Current implementation status

The repository keeps the main deployment artifacts, but the exact training and conversion workflow for the FLAN-T5 optimized binaries is not yet scripted in the repo.
