import sys
from abc import ABC, abstractmethod
from typing import List, Tuple

import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
from PIL import Image, ImageDraw, ImageFont

from utils import Point, polygon_area  # , polygon_centroid


class Detector(ABC):
    def order_points(self, points: List[Point]) -> List[Point]:
        # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
        pts = np.array(points)

        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = [None, None, None, None]

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def visualize(
        self,
        image_pil: Image,
        data: str,
        points: List[Point],
        color: Tuple[int, int, int] = (0, 200, 0),
        opacity: int = 96,
    ) -> Image:
        img = image_pil.convert("RGBA")
        mask = Image.new("RGBA", img.size, color + (0,))
        draw = ImageDraw.Draw(mask)

        draw.polygon(points, outline=color + (255,), fill=color + (opacity,))
        # centroid = polygon_centroid(points)
        # draw.ellipse(
        #    [centroid.x - 3, centroid.y - 3, centroid.x + 3, centroid.y + 3],
        #    outline=color + (255,),
        #    fill=color + (opacity,),
        # )

        if data:
            font = ImageFont.truetype(
                "Montserrat-Regular.ttf", 10
            )  # TODO load font in ctr
            text_width, text_height = font.getsize(data)
            draw.rectangle(((0, 0), (text_width, text_height)), fill=color + (opacity,))
            draw.text((0, 0), text=data, fill="white", font=font)

        img = Image.alpha_composite(img, mask)
        return img.convert("RGB")

    @abstractmethod
    def detect(image_pil: Image) -> Tuple[str, List[Point]]:
        pass


class PyZBarDetector(Detector):
    def detect(self, image_pil: Image) -> Tuple[str, List[Point]]:
        codes = pyzbar.decode(image_pil, symbols=[pyzbar.ZBarSymbol.QRCODE])

        # loop over all detected QR codes and take the largest
        selected_data = ""
        selected_pts = []
        max_area = 0
        for code in codes:
            points = [Point(*map(float, pnt)) for pnt in code.polygon]
            area = polygon_area(points)
            if area >= max_area:
                max_area = area
                selected_data = code.data.decode("UTF-8")
                selected_pts = points
        return (
            selected_data,
            self.order_points(selected_pts) if len(selected_pts) == 4 else [],
        )


class Cv2Detector(Detector):
    def detect(self, image_pil: Image) -> Tuple[str, List[Point]]:
        image_cv = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()
        try:
            data, points, _ = detector.detectAndDecode(image_cv)
        except cv2.error:
            data, points = "", None
        points = [] if points is None else points
        points = [Point(*map(float, pnt)) for pnt in np.squeeze(points)]
        return (data, self.order_points(points) if len(points) == 4 else [])


class QRCodeDetector(Detector):

    _detectors = (PyZBarDetector(), Cv2Detector())

    def detect(self, image_pil: Image) -> Tuple[str, List[Point]]:
        for detector in self._detectors:
            data, points = detector.detect(image_pil)
            if len(points):
                return (data, points)
        else:
            return ("", [])


def main():
    image_pil = Image.open(sys.argv[1]).convert("RGB")
    detectors = (PyZBarDetector(), Cv2Detector(), QRCodeDetector())
    for detector in detectors:
        print(f"=== {detector} ===")
        data, points = detector.detect(image_pil)
        print(f"data: {data!r}")
        print(f"points: {points!r}")


if __name__ == "__main__":
    main()
