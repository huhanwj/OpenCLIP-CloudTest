# client.py
import asyncio
import aiohttp
import av
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay
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
    # Keep the video transmission running indefinitely
    while True:
        await asyncio.sleep(1)


    # 传输5秒
    await asyncio.sleep(120)

    # Close the track and connection
    video_track.stop()
    await pc.close()


if __name__ == '__main__':
    asyncio.run(main())
