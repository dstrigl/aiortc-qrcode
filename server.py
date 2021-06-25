import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
import numpy as np
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

from detect import QRCodeDetector
from replace import QRCodeReplacer
from utils import polygon_points_inside_image, polygon_scale

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()


class VideoTransformTrack(MediaStreamTrack):
    """A video stream track that transforms frames from an another track."""

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self._track = track
        self._transform = transform
        self._detector = QRCodeDetector()
        self._replacer = QRCodeReplacer()
        self._replace_image_cv = cv2.imread("smiley.jpg")

    async def recv(self):
        frame = await self._track.recv()

        if self._transform in ("detect", "replace"):

            # try to localize and decode the QR code
            image_pil = frame.to_image()
            data, points = self._detector.detect(image_pil)

            if data and len(points):

                # enlarge the polygon of the recognized QR code by 5%
                points = polygon_scale(points, 1.05)

                # check whether the enlarged polygon is still inside the image
                if polygon_points_inside_image(points, image_pil):

                    if self._transform == "detect":
                        # visualize the recognized QR code in the image
                        image_pil = self._detector.visualize(image_pil, data, points)
                        # and rebuild a video frame from the new image
                        new_frame = VideoFrame.from_image(image_pil)

                    elif self._transform == "replace":
                        # replace the recognized QR code by another image
                        image_cv = cv2.cvtColor(
                            np.asarray(image_pil), cv2.COLOR_RGB2BGR
                        )
                        image_cv = self._replacer.replace(
                            image_cv, points, self._replace_image_cv
                        )
                        # and rebuild a video frame from the new image
                        new_frame = VideoFrame.from_ndarray(image_cv, format="bgr24")

                    else:
                        assert False

                    # preserving timing information of the new video frame
                    new_frame.pts = frame.pts
                    new_frame.time_base = frame.time_base
                    return new_frame

        return frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = f"PeerConnection({uuid.uuid4()})"
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"]
                )
            )

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def main():
    parser = argparse.ArgumentParser(description="WebRTC QR code detection")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8081, help="Port for HTTP server (default: 8081)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)

    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )


if __name__ == "__main__":
    main()
