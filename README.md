# WebRTC video and data channel server sample

This example illustrates how to use [Web Real-Time Communication (WebRTC)](https://webrtc.org/) and [Object Real-Time Communication (ORTC)](https://ortc.org/) in Python to establishing a video and data channel with a browser, which performs some image processing on the video frames.

## Running

First install the required packages:

```bash
  $ pip install -r requirements.tx
```

When you start the example, it will create an HTTP server which you can connect to from your browser:

```bash
  $ python server.py
```

Once you click `Start` the browser will send the video from its webcam to the server.

The server will send the received video back to the browser and optionally applying one of the following two transformations to it:

- the visualisation of a detected and decoded QR code or
- the replacement of a detected QR code by another image

In parallel to the video stream, the browser sends a "ping" message over the data channel, and the server replies with "pong".

## Note

It should be noted that this example only works when using HTTPS.

## Credits

- [aiortc ](https://github.com/aiortc/aiortc): a library for Web Real-Time Communication (WebRTC) and Object Real-Time Communication (ORTC) in Python
