import warnings
warnings.filterwarnings('ignore')

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Assuming you've installed transformers and torch
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")

# Example translation task
input_text = "translate English to Spanish: Hello"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))

# Save the model and tokenizer
model.save_pretrained('./saved_models')
tokenizer.save_pretrained('./saved_models')
