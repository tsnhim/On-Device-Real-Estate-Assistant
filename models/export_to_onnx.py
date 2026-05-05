from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

model_path = "models/flan_t5_zillow_final1"
output_path = "app/android/app/src/main/assets/onnx_model"

model = ORTModelForSeq2SeqLM.from_pretrained(
    model_path,
    export=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
