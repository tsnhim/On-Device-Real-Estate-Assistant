# Error Analysis

## Current uncertainty

Because the repo now keeps mainly retained artifacts rather than the full earlier experiment history, the main unresolved questions are:

- how answer quality changes across base FLAN-T5, FP16, and dynamic INT8 variants
- whether retrieval reduces unsupported or incomplete answers
- how Whisper transcription quality affects downstream answer quality

## Recommended analysis

- compare FLAN-T5 artifacts on the same held-out prompt set
- score factuality, helpfulness, and repetition manually
- measure CPU latency, RAM use, and startup time
- separate ASR errors from generation errors
