import shapely.geometry
import shapely.affinity

class RotatedRect:
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())


r1 = RotatedRect(10, 15, 10, 5, 0)
r2 = RotatedRect(15, 15, 10, 5, 0)

print(r1.intersection(r2).area)

from matplotlib import pyplot
from descartes import PolygonPatch

fig = pyplot.figure(1, figsize=(10, 4))
ax = fig.add_subplot(121)
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)

ax.add_patch(PolygonPatch(r1.get_contour(), fc='#990000', alpha=0.7))
ax.add_patch(PolygonPatch(r2.get_contour(), fc='#000099', alpha=0.7))
# ax.add_patch(PolygonPatch(r1.intersection(r2), fc='#009900', alpha=1))

pyplot.show()
