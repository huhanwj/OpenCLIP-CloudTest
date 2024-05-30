# client.py
import asyncio
import platform
import aiohttp
import av
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay
import time

relay = []
webcam = []

"""
用于小车采集视频并发包,基于aiortc。基于aiohttp发起web请求。
"""

class VideoFrameTrack(VideoStreamTrack):
    """
    A video stream track that relays frames from OpenCV's VideoCapture.
    """

    def __init__(self, source):
        super().__init__()  # don't forget to initialize base class
        self.source = source

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        # Read frame from OpenCV
        ret, frame = self.source.read()
        if not ret:
            raise Exception("Could not read frame from OpenCV VideoCapture")

        # Convert the frame from BGR to RGB
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a raw pyav.VideoFrame from the RGB frame
        av_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        av_frame.pts = pts
        av_frame.time_base = time_base

        return av_frame


# # TODO 接收netlink消息，获取需要重传的序列号
# async def loss_detection_and_retransmit(video_sender):
#     while True:
#         await asyncio.sleep(1)
#         await video_sender.fast_retransmit(0)


async def main():
    pc = RTCPeerConnection()

    # 获取视频源
    # OpenCV's video capture object
    capture = cv2.VideoCapture(2)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Relay is used to reuse the same video source across multiple consumers
    relay = MediaRelay()

    # Create the video track
    video_track = VideoFrameTrack(capture)
    # 获取rtcrtpsender类的实例
    video_sender = pc.addTrack(video_track)

    # Create an offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # Send the offer to the server
    async with aiohttp.ClientSession() as session:
        # 请修改IP地址以匹配自己的服务器
        async with session.post('http://127.0.0.1:8080/offer', json={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }) as resp:
            answer = await resp.json()
            print("Received answer")

            # Set the remote description
            await pc.setRemoteDescription(RTCSessionDescription(
                sdp=answer["sdp"],
                type=answer["type"]
            ))

    print("Begin Video capture AND rtp transmission")

    await asyncio.sleep(4)
    # 测试 重传当前发包历史中索引为0的包
    # await video_sender.fast_retransmit(0)

    # # TODO tx-status接收与重传调用
    # await loss_detection_and_retransmit(video_sender)

    # 传输5秒
    await asyncio.sleep(120)

    # Close the track and connection
    video_track.stop()
    await pc.close()


if __name__ == '__main__':
    asyncio.run(main())
