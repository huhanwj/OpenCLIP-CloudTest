import datetime
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers import BlipProcessor, BlipForQuestionAnswering
import cv2
import asyncio
import threading
import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
import queue
model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
model.eval()
peers = {}

message_queue = queue.Queue()

def display_frame(message_queue):
    while True:
        if not message_queue.empty():
            img = message_queue.get()
            cv2.imshow("Live Video", img)
            cv2.waitKey(1)

def handle_input(message_queue):
    while True:
        question = input("User: ")
        msgs = [{'role': 'user', 'content': question}]
        frame = message_queue.get()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frameText = "Time: " + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # cv2.putText(img, frameText, (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        # cv2.imwrite("1.png",frame)
        img.save("1.png")
        res = model.chat(
            image=img,
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

process_thread = threading.Thread(target=display_frame, args=(message_queue,))
input_thread = threading.Thread(target=handle_input, args=(message_queue,))

process_thread.start()
input_thread.start()

async def process_track(track):
    print("Recording!")
    frame_index = 1
    try:
        while True:
            global message_queue
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            message_queue.put(img)
            frame_index += 1

    except Exception as e:
        print("An error occurred: ", e)
    finally:
        cv2.destroyAllWindows()

async def index(request):
    content = open('index.html', 'r').read()
    return web.Response(content_type='text/html', text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    #连接建立与管理
    pc = RTCPeerConnection()
    pc_id = "peer_connection_{}".format(len(peers))
    peers[pc_id] = pc

    @pc.on("track")
    async def on_track(track):
        print("Track %s received" % track.kind)

        if track.kind == "video":
            original_track = track
            # 创建一个任务以处理接收到的视频轨
            asyncio.create_task(process_track(original_track))

        @track.on("ended")
        async def on_ended():
            print("Track %s ended", track.kind)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


#启动web服务器
app = web.Application()
app.router.add_get('/', index)
#offer响应视频轨道请求
app.router.add_post('/offer', offer)

web.run_app(app, port=8080)

