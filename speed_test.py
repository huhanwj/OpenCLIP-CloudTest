"""
This script runs to compare the inference speed between Pytorch (CPU/GPU), ONNX (CPU/GPU) and TensorRT models.
"""

import argparse
import torch
from PIL import Image
import open_clip as clip
# from open_clip import tokenizer
# from cn_clip.clip.utils import create_model, _MODEL_INFO, image_transform
# from cn_clip.training.main import convert_models_to_fp32, convert_weights
from utils.benchmark_utils import track_infer_time, print_timings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-arch", 
        required=True, 
        choices=["ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        help="Specify the architecture (model scale) of Chinese-CLIP model for speed comparison."
    )
    parser.add_argument('--pretrained', type=str, default=None, 
                        help='The pretrained model of Open CLIP, if not set, the program will not run Pytorch model.')

    parser.add_argument('--batch-size', default=1, type=int, help='The batch-size of the inference input. Default to 1.')
    parser.add_argument('--n', default=100, type=int, help='The iteration number for inference speed test. Default to 100.')
    parser.add_argument('--warmup', default=10, type=int, help='Warmup iterations. Default to 10.')

    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="CPU or GPU speed test. Default to cuda",)
    parser.add_argument(
        "--context-length", type=int, default=52, help="The padded length of input text (include [CLS] & [SEP] tokens). Default to 52."
    )

    args = parser.parse_args()

    return args


def prepare_pytorch_model(args):
    pt_model,_,preprocess = clip.create_model_and_transforms(args.model_arch,pretrained=args.pretrained)
    tokenizer = clip.get_tokenizer(args.model_arch)
    if args.device == "cuda":
        pt_model.cuda()
    return pt_model,preprocess,tokenizer


onnx_execution_provider_map = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


if __name__ == '__main__':
    args = parse_args()
    
    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")
    print(f"Prepare the Pytorch model from {args.model_arch} {args.pretrained}")
    pt_model,preprocess,tokenizer = prepare_pytorch_model(args)
    image = preprocess(Image.open("CLIP.jpg")).unsqueeze(0)
    text = tokenizer(["a diagram", "a dog", "a cat"])
    if args.device == "cuda":
        image = image.cuda()
        text = text.cuda()

    # test the image feature extraction
    print("Begin the image feature extraction speed test...")

    for i in range(args.warmup):
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = pt_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
    print("Forward the Pytorch image model...")
    time_buffer = list()
    for i in range(args.n):
        with track_infer_time(time_buffer):
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = pt_model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
    print_timings(name=f"Pytorch image inference speed (batch-size: {args.batch_size}):", timings=time_buffer)

    # test the image feature extraction
    print("Begin the text feature extraction speed test...")
    
    for i in range(args.warmup):
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = pt_model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
    print("Forward the Pytorch text model...")
    time_buffer = list()
    for i in range(args.n):
        with track_infer_time(time_buffer):
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = pt_model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
    print_timings(name=f"Pytorch text inference speed (batch-size: {args.batch_size}):", timings=time_buffer)
    del pt_model
