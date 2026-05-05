from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

model_path = "flan_t5_zillow_final1"

model = ORTModelForSeq2SeqLM.from_pretrained(
    model_path,
    export=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model.save_pretrained("onnx_model")
tokenizer.save_pretrained("onnx_model")