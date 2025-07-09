# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)

# inputs = tokenizer("Who are you?", return_tensors="pt")
# outputs = model.generate(**inputs, max_new_tokens=50)

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# downgrade the tansformers version to 4.31.0 