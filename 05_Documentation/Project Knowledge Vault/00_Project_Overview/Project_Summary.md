# Project Summary

## One-paragraph summary

This project is now organized around an on-device real-estate assistant pipeline that uses a FLAN-T5 seq2seq model for answer generation, Whisper for speech recognition, and retrieval assets for grounding and search support.

## What the project currently contains

- a main FLAN-T5 checkpoint in `02_Models/PyTorch/flan_t5_zillow_final1/`
- optimized FLAN-T5 binaries in FP16 and dynamic INT8 formats
- quantized Whisper speech-recognition models in `02_Models/Speech/whisper_model/`
- a retrieval corpus and FAISS index in `03_Data/Retrieval/`

## Current interpretation of the project

The repo is best understood as an artifact-centered prototype for local inference. The main remaining work is to make the FLAN-T5 and Whisper pipeline fully reproducible, benchmarked, and integrated end to end.
