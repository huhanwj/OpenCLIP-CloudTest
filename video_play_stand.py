# server.py
import asyncio
import threading
import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
import time
import pyvirtualcam
from pyvirtualcam import PixelFormat
import logging

relay = MediaRelay()

# Dictionary to hold information about peer connections
peers = {}

"""
用于小车采集视频并发包,基于aiortc。基于aiohttp发起web请求。
"""
async def process_track(track):
    print("Streaming!")
    frame_index = 1

    # Initialize virtual camera. Adjust the preferred_fps as needed.
    async def main():
        with pyvirtualcam.Camera(width=1280, height=720, fps=30, fmt=PixelFormat.BGR) as cam:
            print(f'Using virtual camera: {cam.device}')
            frame_index = 0
            try:
                while True:
                    frame = await track.recv()  # Assuming track is defined and recv() is properly awaited
                    img = frame.to_ndarray(format="bgr24")

                    # Utilize CUDA operations if possible
                    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        gpu_frame = cv2.cuda_GpuMat()
                        gpu_frame.upload(img)
                        # Perform any GPU-based operations here
                        processed_img = gpu_frame.download()
                    else:
                        processed_img = img  # Fallback to CPU processing if CUDA is not available

                    print(f"Received a frame: {frame_index} at {time.time()}")

                    # Send the frame to the virtual camera
                    cam.send(processed_img)
                    cam.sleep_until_next_frame()

                    frame_index += 1
            except Exception as e:
                print("An error occurred: ", e)
            finally:
                print("Streaming stopped.")

        # Close the OpenCV window
        # cv2.destroyAllWindows()


async def index(request):
    content = open("index.html", "r").read()
    return web.Response(content_type="text/html", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # 连接建立与管理
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

    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


# 启动web服务器
app = web.Application()
app.router.add_get("/", index)
# offer响应视频轨道请求
app.router.add_post("/offer", offer)

if __name__ == "__main__":
    web.run_app(app, port=8080)
