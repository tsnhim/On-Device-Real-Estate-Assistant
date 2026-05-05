# OnDeviceDeepLearning

This repository is now focused on a single lightweight assistant stack:

- `FLAN-T5` as the main seq2seq text-generation pipeline
- `Whisper` as the baseline speech-recognition model
- retrieval assets to support grounded real-estate question answering

## Repository Layout

- `02_Models/`
  - `PyTorch/flan_t5_zillow_final1/`: main FLAN-T5 checkpoint
  - `Optimized/`: optimized FLAN-T5 binaries
  - `Speech/`: Whisper speech-model assets
- `03_Data/`
  - `Retrieval/`: retrieval corpus and FAISS index
- `04_Experiments/`
  - reserved for future reproducible experiment tracking
- `05_Documentation/`
  - project documentation centered on the FLAN-T5 + Whisper pipeline

## Current Main Assets

- `02_Models/PyTorch/flan_t5_zillow_final1/`
- `02_Models/Optimized/flan_t5_fp16.bin`
- `02_Models/Optimized/flan_t5_dynamic_int8.bin`
- `02_Models/Speech/whisper_model/whisper-tiny-q4_1.gguf`
- `02_Models/Speech/whisper_model/whisper-tiny-q8_0.gguf`
- `03_Data/Retrieval/zillow_records.json`
- `03_Data/Retrieval/zillow_faiss.index`

## Start Here

- Project overview: `05_Documentation/Project Knowledge Vault/README.md`
- System summary: `05_Documentation/Project Knowledge Vault/00_Project_Overview/Project_Summary.md`
- Codebase map: `05_Documentation/Project Knowledge Vault/04_Implementation/Codebase_Map.md`
