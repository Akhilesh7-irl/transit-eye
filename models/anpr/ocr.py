import copy
import os
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
AWIROS_DIR = SCRIPT_DIR / "awiros"

CTC_NUM_CLASSES = 64
NRTR_NUM_CLASSES = 67

MODEL_CONFIG = {
    "Architecture": {
        "model_type": "rec",
        "algorithm": "SVTR_HGNet",
        "Transform": None,
        "Backbone": {
            "name": "PPHGNetV2_B4",
            "text_rec": True
        },
        "Head": {
            "name": "MultiHead",
            "out_channels_list": {
                "CTCLabelDecode": CTC_NUM_CLASSES,
                "NRTRLabelDecode": NRTR_NUM_CLASSES,
            },
            "head_list": [
                {
                    "CTCHead": {
                        "Neck": {
                            "name": "svtr",
                            "dims": 120,
                            "depth": 2,
                            "hidden_dims": 120,
                            "kernel_size": [1, 3],
                            "use_guide": True,
                        },
                        "Head": {
                            "fc_decay": 1e-05
                        },
                    }
                },
                {
                    "NRTRHead": {
                        "nrtr_dim": 384,
                        "max_text_length": 25
                    }
                },
            ],
        },
    },
}

IMAGE_SHAPE = [3, 48, 320]


class PlateOCR:
    def __init__(self, weights=None, device="cpu"):
        self.device = device

        if weights is None:
            weights = AWIROS_DIR / "model.safetensors"

        self.weights = Path(weights)

        if not self.weights.exists():
            raise FileNotFoundError(
                f"Awiros weights not found: {self.weights}"
            )

        # Make PaddleOCR importable
        paddleocr_dir = AWIROS_DIR / "PaddleOCR"

        if str(paddleocr_dir) not in sys.path:
            sys.path.insert(0, str(paddleocr_dir))

        import paddle

        from ppocr.modeling.architectures import build_model
        from ppocr.postprocess import build_post_process
        from safetensors.numpy import load_file

        self.paddle = paddle

        # Select device
        if device == "gpu" and not paddle.is_compiled_with_cuda():
            print("CUDA not available. Falling back to CPU.")
            self.device = "cpu"

        paddle.set_device(self.device)

        # Dictionary used by Awiros
        dict_path = AWIROS_DIR / "en_dict.txt"

        # CTC decoder
        self.post_process = build_post_process({
            "name": "CTCLabelDecode",
            "character_dict_path": str(dict_path),
            "use_space_char": True,
        })

        # Build model
        config = copy.deepcopy(MODEL_CONFIG)

        self.model = build_model(
            config["Architecture"]
        )

        self.model.eval()

        # Load safetensors weights
        np_state = load_file(str(self.weights))

        state_dict = {
            key: paddle.to_tensor(value)
            for key, value in np_state.items()
        }

        self.model.set_state_dict(state_dict)

        print(f"Loaded Awiros OCR weights from {self.weights}")
        print(f"OCR device: {self.device}")

    def resize_for_rec(self, image):
        _, target_height, target_width = IMAGE_SHAPE

        image_height, image_width = image.shape[:2]

        ratio = target_height / image_height

        new_width = min(
            int(image_width * ratio),
            target_width
        )

        resized = cv2.resize(
            image,
            (new_width, target_height)
        )

        if new_width < target_width:
            padded = np.zeros(
                (target_height, target_width, 3),
                dtype=np.uint8
            )

            padded[:, :new_width, :] = resized
            resized = padded

        return resized

    def preprocess(self, image):
        image = self.resize_for_rec(image)

        image = image.astype(np.float32) / 255.0

        image = (image - 0.5) / 0.5

        image = image.transpose((2, 0, 1))

        return image

    def read_plate(self, image):
        """
        Run OCR on a license plate.

        image can be:
            - OpenCV BGR image
            - path to an image
        """

        try:
            # If a path was supplied
            if isinstance(image, (str, Path)):
                image = cv2.imread(str(image))

                if image is None:
                    raise ValueError(
                        f"Could not read image: {image}"
                    )

            # Preprocess
            processed = self.preprocess(image)

            tensor = self.paddle.to_tensor(
                np.expand_dims(processed, axis=0)
            )

            # Inference
            with self.paddle.no_grad():
                predictions = self.model(tensor)

            # Get CTC prediction
            if isinstance(predictions, dict):
                prediction = predictions.get(
                    "ctc",
                    next(iter(predictions.values()))
                )

            elif isinstance(predictions, (list, tuple)):
                prediction = predictions[0]

            else:
                prediction = predictions

            # Decode
            result = self.post_process(
                prediction.numpy()
            )

            if (
                isinstance(result, (list, tuple))
                and len(result) > 0
            ):
                text, confidence = result[0]
            else:
                text = ""
                confidence = 0.0

            text = text.strip().upper()
            confidence = float(confidence)

            return {
                "plate_text": text,
                "confidence": round(confidence, 4),
                "accepted": confidence >= 0.80,
                "source": "awiros"
            }

        except Exception as e:

            return {
                "plate_text": "",
                "confidence": 0.0,
                "accepted": False,
                "source": "awiros",
                "error": str(e)
            }


# ---------------------------------------------------------
# Simple command-line test
# ---------------------------------------------------------

if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Awiros Indian License Plate OCR"
    )

    parser.add_argument(
        "image",
        help="Path to license plate crop"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu"]
    )

    parser.add_argument(
        "--weights",
        default=None
    )

    args = parser.parse_args()

    ocr = PlateOCR(
        weights=args.weights,
        device=args.device
    )

    result = ocr.read_plate(args.image)

    print(
        json.dumps(
            result,
            indent=2
        )
    )