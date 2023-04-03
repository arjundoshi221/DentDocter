"""
File to predict on a pretrained YOLO model
"""

from ultralytics import YOLO


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="what_is_this_bs_42069",
        description="Predict using YOLOv8 model",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Pretrained YOLO model (in PyTorch format)",
        default="../weights/cell_detection_YOLO.pt",
        required=True,
    )

    parser.add_argument(
        "--image_path", type=str, help="Filepath of the image", required=True
    )

    args = parser.parse_args()
    model = YOLO(args.model)
    print(model(args.image_path))


if __name__ == "__main__":
    main()
