from collections import namedtuple
from math import hypot
from operator import itemgetter
from typing import List

import numpy as np
from PIL import Image

Point = namedtuple("Point", ["x", "y"])


def polygon_area(points: List[Point]) -> float:
    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    x = np.array(list(map(itemgetter(0), points)))
    y = np.array(list(map(itemgetter(1), points)))
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_centroid(points: List[Point]) -> Point:
    x = np.array(list(map(itemgetter(0), points)))
    y = np.array(list(map(itemgetter(1), points)))
    return Point(sum(x) / len(points), sum(y) / len(points))


def polygon_mean_side_len(points: List[Point]) -> float:
    ptdiff = lambda p1, p2: hypot(p1.x - p2.x, p1.y - p2.y)
    diffs = [
        ptdiff(p1, p2)
        for p1, p2 in zip(
            points,
            points[1:]
            + [
                points[0],
            ],
        )
    ]
    return sum(diffs) / len(diffs)


def polygon_scale(points: List[Point], scale: float) -> List[Point]:
    centroid = polygon_centroid(points)
    return list(
        map(
            lambda pnt: Point(
                scale * (pnt.x - centroid.x) + centroid.x,
                scale * (pnt.y - centroid.y) + centroid.y,
            ),
            points,
        )
    )


def polygon_points_inside_image(points: List[Point], image_pil: Image) -> bool:
    return all(
        map(
            lambda pnt: 0 <= pnt.x < image_pil.width and 0 <= pnt.y < image_pil.height,
            points,
        )
    )
