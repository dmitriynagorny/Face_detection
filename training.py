from ultralytics import YOLO
import logging
import argparse
import sys


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Train')


class Trainer:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument('--path',
                            help='Path to the YAML config file',
                            default='./configs/config.yaml',
                            type=str,
                            )
        parser.add_argument('--device',
                            help='CUDA or CPU',
                            default='cuda',
                            type=str,
                            )
        parser.add_argument('--epochs',
                            help='Count of epochs',
                            default=10,
                            type=int,
                            )
        parser.add_argument('--batch',
                            help='Batch size',
                            default=16,
                            type=int,
                            )
        parser.add_argument('--img_size',
                            help='IMG size',
                            default=640,
                            type=int,
                            )

        self.cli_args = parser.parse_args(sys_argv)
        self.device = self.cli_args.device
        logger.info(self.device)

    def train(self) -> None:
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')
        logger.info('Created YOLOv8n model')

        results = model.train(data='./configs/config.yaml', epochs=self.cli_args.epochs, resume=True, iou=0.5, conf=0.001,
                              batch=self.cli_args.batch, device=self.device, imgsz=self.cli_args.img_size)
        logger.info('Successfully training')


if __name__ == '__main__':
    Trainer().train()