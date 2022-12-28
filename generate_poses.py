import pose
import fist_pose
import argparse
import os


MODELS_DIR = "models"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-path",
        type=str,
        default="../input",
        help="Path to the input directory"
    )

    parser.add_argument(
        "--pose-estimation-model-path",
        type=str,
        default=f"./{MODELS_DIR}/hrn_w48_384x288.onnx",
        help="Pose Estimation model",
    )

    parser.add_argument(
        "--contact-model-path",
        type=str,
        default=f"./{MODELS_DIR}/contact_hrn_w32_256x192.onnx",
        help="Contact model",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Torch device",
    )

    parser.add_argument(
        "--spin-model-path",
        type=str,
        default=f"./{MODELS_DIR}/spin_model_smplx_eft_18.pt",
        help="SPIN model path",
    )

    parser.add_argument(
        "--smpl-type",
        type=str,
        default="smplx",
        choices=["smplx"],
        help="SMPL model type",
    )
    parser.add_argument(
        "--smpl-model-dir",
        type=str,
        default=f"./{MODELS_DIR}/models/smplx",
        help="SMPL model dir",
    )
    parser.add_argument(
        "--smpl-mean-params-path",
        type=str,
        default=f"./{MODELS_DIR}/data/smpl_mean_params.npz",
        help="SMPL mean params",
    )
    parser.add_argument(
        "--essentials-dir",
        type=str,
        default=f"./{MODELS_DIR}/smplify-xmc-essentials",
        help="SMPL Essentials folder for contacts",
    )

    parser.add_argument(
        "--parametrization-path",
        type=str,
        default=f"./{MODELS_DIR}/smplx_parametrization/parametrization.npy",
        help="Parametrization path",
    )
    parser.add_argument(
        "--bone-parametrization-path",
        type=str,
        default=f"./{MODELS_DIR}/smplx_parametrization/bone_to_param2.npy",
        help="Bone parametrization path",
    )
    parser.add_argument(
        "--foot-inds-path",
        type=str,
        default=f"./{MODELS_DIR}/smplx_parametrization/foot_inds.npy",
        help="Foot indinces",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save the results",
    )

    parser.add_argument(
        "--img-path",
        type=str,
        help="Path to img to test",
    )

    parser.add_argument(
        "--use-contacts",
        action="store_true",
        help="Use contact model",
    )
    parser.add_argument(
        "--use-msc",
        action="store_true",
        help="Use MSC loss",
    )
    parser.add_argument(
        "--use-natural",
        action="store_true",
        help="Use regularity",
    )
    parser.add_argument(
        "--use-cos",
        action="store_true",
        help="Use cos model",
    )
    parser.add_argument(
        "--use-angle-transf",
        action="store_true",
        help="Use cube foreshortening transformation",
    )

    parser.add_argument(
        "--c-mse",
        type=float,
        default=0,
        help="MSE weight",
    )
    parser.add_argument(
        "--c-par",
        type=float,
        default=10,
        help="Parallel weight",
    )

    parser.add_argument(
        "--c-f",
        type=float,
        default=1000,
        help="Cos coef",
    )
    parser.add_argument(
        "--c-parallel",
        type=float,
        default=100,
        help="Parallel weight",
    )
    parser.add_argument(
        "--c-reg",
        type=float,
        default=1000,
        help="Regularity weight",
    )
    parser.add_argument(
        "--c-cont2d",
        type=float,
        default=1,
        help="Contact 2D weight",
    )
    parser.add_argument(
        "--c-msc",
        type=float,
        default=17_500,
        help="MSC weight",
    )

    parser.add_argument(
        "--fist",
        nargs="+",
        type=str,
        choices=list(fist_pose.INT_TO_FIST),
    )

    args = parser.parse_args()

    return args


def main(args):
    path = args.input_path
    args.use_contacts = True
    args.use_natural = True
    args.use_cos = True
    args.use_angle_transf = True
    for dirname in os.listdir(path):
        if not os.path.isdir(os.path.join(path, dirname)):
            continue
        print(f'Processing {dirname}')
        args.img_path = os.path.join(path, dirname)
        args.save_path = os.path.join(path, dirname, 'pose_output')
        pose.main(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
