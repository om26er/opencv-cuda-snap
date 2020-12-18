#!/usr/bin/env python3

import os

from autobahn.twisted import wamp
import cv2
import numpy


class ClientSession(wamp.ApplicationSession):
    def detect_faces(self, data, uuid):
        image = numpy.frombuffer(data, dtype=numpy.uint8)
        gray = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        gpu_frame = cv2.cuda_GpuMat(gray)

        # Detect faces in the image
        faces = FACE_FASCADE.detectMultiScale(gpu_frame)
        coords = faces.download()
        if coords is not None:
            coords = coords.data.tolist()
            self.publish("io.crossbar.demo.faces", coords[0], uuid)

    async def onJoin(self, details):
        self.log.info("Connected:  {details}", details=details)
        self._transport.MAX_LENGTH = 1000000
        await self.subscribe(self.detect_faces, "io.crossbar.demo.frames")

    def onLeave(self, details):
        self.log.info("Router session closed ({details})", details=details)
        self.disconnect()


if __name__ == '__main__':
    FACE_FASCADE = cv2.cuda.CascadeClassifier_create(os.path.join(os.path.expandvars("$SNAP"), "haarcascade_frontalface_default.xml"))
    session = ClientSession(wamp.ComponentConfig('realm1', {}))
    runner = wamp.ApplicationRunner('rs://localhost:8080', 'realm1')
    runner.run(session)

