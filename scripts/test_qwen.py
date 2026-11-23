"""Load Qwen and make it answer a simple question."""

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

messages = [
    {"role": "user", "content": "What is the capital of France?"},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
print("PROMPT :")
print(prompt)

inputs = tokenizer(prompt, return_tensors="pt")
print("NUMÉROS :", inputs["input_ids"][0].tolist())

outputs = model.generate(**inputs, max_new_tokens=50)
prompt_length = inputs["input_ids"].shape[1]
answer_ids = outputs[0][prompt_length:]
answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
print("RÉPONSE :", answer)