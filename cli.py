import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import cv2

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

# Open the video stream from the remote side
video_stream_url = 'rtsp://example.com/video_stream'  # Replace with the actual URL of the remote video stream
video_capture = cv2.VideoCapture(video_stream_url)

while True:
    question = input("User: ")
    if question.lower() == 'exit':
        break

    # Capture the current frame from the video stream
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame from the video stream.")
        break

    # Convert the frame from BGR to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to PIL Image format
    image = Image.fromarray(frame_rgb)

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

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()