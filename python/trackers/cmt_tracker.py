import cv2
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "thirdparty" / "CMT"))

from trackers.base_tracker import BaseTracker
from thirdparty.CMT.CMT import CMT

class CMTTracker(BaseTracker):
    def __init__(self):
        self.tracker = CMT()

    def init(self, image, box):
        x, y, w, h = map(int, box)
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.tracker.initialise(im_gray, (x, y), (x + w, y + h))

    def update(self, image):
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.tracker.process_frame(im_gray)

        if self.tracker.has_result:
            x, y = self.tracker.tl
            w = self.tracker.tr[0] - self.tracker.tl[0]
            h = self.tracker.bl[1] - self.tracker.tl[1]
            return True, [x, y, w, h]
        else:
            return False, [0, 0, 0, 0]