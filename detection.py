import cv2
import logging
import argparse
import sys
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('detection')
YOLO_SIZE = (640, 640)


class Detection:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument('--path',
                            help='Path to the video',
                            default='./data/video/1.mp4',
                            type=str,
                            )
        parser.add_argument('--model',
                            help='Path to the model weight',
                            default='./configs/weight/best.pt',
                            type=str,
                            )

        self.cli_args = parser.parse_args(sys_argv)
        self.model = YOLO(self.cli_args.model)  # Load model with weight

    def detect_video(self) -> None:
        cap = cv2.VideoCapture(self.cli_args.path)

        while cap.isOpened():
            success, img = cap.read()
            res = self.model.predict(img)

            for r in res:  # Create bounding boxes

                annotator = Annotator(img)

                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]
                    c = box.cls
                    annotator.box_label(b, self.model.names[int(c)])

            img = annotator.result()

            cv2.imshow('Face detection', img)

            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Detection().detect_video()
