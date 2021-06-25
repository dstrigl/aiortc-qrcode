import sys
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from detect import QRCodeDetector
from utils import (
    Point,
    polygon_mean_side_len,
    polygon_points_inside_image,
    polygon_scale,
)


class QRCodeReplacer:
    def extract(self, image_cv: np.ndarray, points: List[Point]) -> np.ndarray:
        # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
        (tl, tr, br, bl) = points

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        width_a = np.sqrt(((br.x - bl.x) ** 2) + ((br.y - bl.y) ** 2))
        width_b = np.sqrt(((tr.x - tl.x) ** 2) + ((tr.y - tl.y) ** 2))
        max_width = max(int(width_a), int(width_b))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        height_a = np.sqrt(((tr.x - br.x) ** 2) + ((tr.y - br.y) ** 2))
        height_b = np.sqrt(((tl.x - bl.x) ** 2) + ((tl.y - bl.y) ** 2))
        max_height = max(int(height_a), int(height_b))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(np.array(points, dtype="float32"), dst)
        warped = cv2.warpPerspective(image_cv, M, (max_width, max_height))

        # return the warped image
        return warped

    def replace(
        self, img_cv_dst: np.ndarray, points_dst: List[Point], img_cv_src: np.ndarray
    ) -> np.ndarray:
        points_dst = np.array(points_dst, dtype=np.float32)
        rows, cols, _ = img_cv_src.shape
        points_src = np.array(
            [[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]],
            dtype=np.float32,
        )

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(points_src, points_dst)
        warped_image_src = cv2.warpPerspective(
            img_cv_src, M, (img_cv_dst.shape[1], img_cv_dst.shape[0])
        )

        # create the corresponding mask and combine the source and destination image
        mask = np.full(img_cv_dst.shape[0:2], 255, dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, points_dst.astype(int), (0, 0, 0))
        masked_image = cv2.bitwise_and(img_cv_dst, img_cv_dst, mask=mask)
        return cv2.add(masked_image, warped_image_src)


def main():
    image_pil = Image.open(sys.argv[1]).convert("RGB")
    scale = float(sys.argv[2])
    detector = QRCodeDetector()
    print(f"=== {detector} ===")
    data, points = detector.detect(image_pil)
    print(f"data: {data!r}")
    print(f"points: {points!r}")
    if data and len(points):
        mean_side_len = polygon_mean_side_len(points)
        print(f"mean side length: {mean_side_len}")
        points = polygon_scale(points, scale)
        print(f"rescaled points: {points!r}")
        if polygon_points_inside_image(points, image_pil):
            image_cv = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)
            qrcode_cv = QRCodeReplacer().extract(image_cv, points)
            print(repr(qrcode_cv))
            # qrcode_pil = Image.fromarray(cv2.cvtColor(qrcode_cv, cv2.COLOR_BGR2RGB))
            # qrcode_pil.save("extracted-qrcode.png")
            cv2.imwrite("extracted-qrcode.png", qrcode_cv)


if __name__ == "__main__":
    main()
