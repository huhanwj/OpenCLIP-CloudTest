import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()
image_path = input("Image path (or press Enter to skip): ")
if image_path:
    image = Image.open(image_path).convert('RGB')
else:
    image = None
while True:

    question = input("User: ")
    if question.lower() == 'exit':
        break

    msgs = [{'role': 'user', 'content': question}]

    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7,
        stream=True
    )

    print("Assistant: ", end='')
    generated_text = ""
    for new_text in res:
        generated_text += new_text
        print(new_text, flush=True, end='')
    print()