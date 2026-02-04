from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./ruciq-ai")
tokenizer = AutoTokenizer.from_pretrained("./ruciq-ai")

while True:
    prompt = input("You: ")
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=80)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
