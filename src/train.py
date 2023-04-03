"""
File to train a YOLO model
"""

from ultralytics import YOLO


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="what_is_this_bs_42069",
        description="Train your model using YOLOv8",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Pretrained YOLO model (in PyTorch format)",
        default="../weights/cell_detection_YOLO.pt",
        required=True,
    )

    parser.add_argument(
        "--data", type=str, help="Filepath of the `data.yaml`", required=True
    )

    parser.add_argument(
        "--epochs", type=str, help="Number of epochs", required=False, default=300
    )

    parser.add_argument(
        "--export",
        type=str,
        help="File format of the model to export",
        required=False,
        default="onnx",
    )
    # Parse the arguments
    args = parser.parse_args()

    # Create a model instance
    model = YOLO(args.model)

    # Train the model
    results = model.train(data=args.data, epochs=int(args.epochs))

    # Export the model
    success = model.export(format=args.export)


if __name__ == "__main__":
    main()
