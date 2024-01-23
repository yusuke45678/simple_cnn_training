import argparse


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.MetavarTypeHelpFormatter
):
    """show default values of argparse.
    see
    https://stackoverflow.com/questions/18462610/argumentparser-epilog-and-description-formatting-in-conjunction-with-argumentdef
    for details.
    """


def get_args() -> argparse.Namespace:
    """generate argparse object

    Returns:
        args (argparse.Namespace): object of command line arguments
    """
    parser = argparse.ArgumentParser(
        description="simple image/video classification",
        formatter_class=CustomFormatter
    )

    # dataset
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        default="./downloaded_data",
        help="root of dataset.",
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        default="CIFAR10",
        choices=["CIFAR10", "ImageFolder", "VideoFolder", "ZeroImages"],
        help="name of dataset.",
    )
    parser.add_argument(
        "-td",
        "--train_dir",
        type=str,
        default="train",
        help="subdier name of training dataset.",
    )
    parser.add_argument(
        "-vd",
        "--val_dir",
        type=str,
        default="val",
        help="subdier name of validation dataset.",
    )

    # model
    parser.add_argument(
        "--torch_home",
        type=str,
        default="./pretrained_models",
        help="TORCH_HOME environment variable "
        "where pre-trained model weights are stored.",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50", "x3d"],
        help="name of the model",
    )
    parser.add_argument(
        "--use_pretrained",
        dest="use_pretrained",
        action="store_true",
        help="use pretrained model weights (default)",
    )
    parser.add_argument(
        "--scratch",
        dest="use_pretrained",
        action="store_false",
        help="do not use pretrained model weights, "
        "instead train from scratch (not default)",
    )
    parser.set_defaults(use_pretrained=True)

    # video
    parser.add_argument(
        "--frames_per_clip", type=int, default=16, help="frames per clip."
    )
    parser.add_argument(
        "--clip_duration",
        type=float,
        default=80 / 30,
        help="duration of a clip (in second).",
    )
    parser.add_argument(
        "--clips_per_video",
        type=int,
        default=1,
        help="sampling clips per video for validation",
    )

    # training
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size.")
    parser.add_argument(
        "-w", "--num_workers", type=int, default=2, help="number of workers."
    )
    parser.add_argument(
        "-e", "--num_epochs", type=int, default=25, help="number of epochs."
    )
    parser.add_argument(
        "-vi",
        "--val_interval_epochs",
        type=int,
        default=1,
        help="validation interval in epochs.",
    )
    parser.add_argument(
        "-li",
        "--log_interval_steps",
        type=int,
        default=1,
        help="logging interval in steps.",
    )

    # optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam"],
        help="optimizer.",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="steps to accumlate gradients.",
    )
    parser.add_argument("-lr", type=float, default=1e-4, help="learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum of SGD.")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="weight decay."
    )
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[0.9, 0.999],
        help="betas of Adam.",
    )
    parser.add_argument(
        "--use_scheduler",
        dest="use_scheduler",
        action="store_true",
        help="use scheduler (not default)",
    )
    parser.add_argument(
        "--no_scheduler",
        dest="use_scheduler",
        action="store_false",
        help="do not use scheduler (default)",
    )
    parser.set_defaults(use_scheduler=False)

    parser.add_argument(
        "--use_dp",
        dest="gpu_strategy",
        action="store_const",
        const="dp",
        help="use multi GPUs with data parallel",
    )
    parser.add_argument(
        "--single_gpu",
        dest="gpu_strategy",
        action="store_const",
        const="None",
        help="use single GPU",
    )
    parser.add_argument(
        "--gpu_strategy",
        type=str,
        default="dp",
        choices=["None", "dp", "ddp"],
        help="GPU training strategy. "
        "None: single GPU. "
        "dp: Data Parallel (default). "
        "ddp: Distributed Data Parallel. ",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=2,
        help="how many GPUs are used for dp and ddp.",
    )

    # log dirs
    parser.add_argument(
        "--comet_log_dir",
        type=str,
        default="./comet_logs/",
        help="dir to comet log files.",
    )
    parser.add_argument(
        "--tf_log_dir",
        type=str,
        default="./tf_logs/",
        help="dir to TensorBoard log files.",
    )

    # checkpoint files
    parser.add_argument(
        "--save_checkpoint_dir",
        type=str,
        default="./log",
        help="dir to save checkpoint files.",
    )
    parser.add_argument(
        "--checkpoint_to_resume",
        type=str,
        default=None,
        help="path to the checkpoint file to resume from.",
    )

    # disabling comet for debugging
    parser.add_argument(
        "--disable_comet",
        "--no_comet",
        dest="disable_comet",
        action="store_true",
        help="do not use comet.ml (default: use comet)",
    )
    parser.set_defaults(disable_comet=False)

    args = parser.parse_args()

    if args.dataset_name in ["CIFAR10", "ImageFolder"]:
        args.data_type = "image"
    elif args.dataset_name in ["VideoFolder"]:
        args.data_type = "video"

    print(args)

    return args
