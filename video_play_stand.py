# server.py
import asyncio

import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
import time
import datetime

import logging

# Define the current time in a format suitable for a filename
current = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
try:
    current = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    logging.basicConfig(
        level=logging.DEBUG,
        filename=f"Log/video_log_{current}.log",
        filemode='a',
        format='%(asctime)s - [%(levelname)s] - %(message)s'
    )
except Exception as e:
    print(f"Logging setup failed: {e}")


relay = MediaRelay()

# Dictionary to hold information about peer connections
peers = {}

"""
用于小车采集视频并发包,基于aiortc。基于aiohttp发起web请求。
"""

async def process_track(track):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("record_ori.avi", fourcc, 60, (1920, 1080))

    print("Recording!")
    frame_index = 1
    try:
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="rgb24")
            print("Received a frame:" + str(frame_index) + " at "+ str(time.time()))
            now_frame_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f') #- first_frame
            logging.debug("Received a frame: %s at %s", frame_index, now_frame_timestamp)
            # rec_interval = (now_frame_timestamp - last_frame_timestamp) * 1000
            frame_index += 1
            # frameText = "Frame: " + str(frame_index) + " Time: " + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            # cv2.putText(img, frameText, (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)

            # Write the frame to the output video file
            out.write(img)
    except Exception as e:
        print("An error occurred: ", e)
    finally:
        # Release the VideoWriter when done
        out.release()
        print("Recording stopped.")


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

if __name__ == '__main__':
    web.run_app(app, port=8080)
