import argparse
import os
from json import load
import torch
from numpy import pi


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-path',
        type=str,
        default=f'./input/',
        help='Path to the input directory',
    )

    parser.add_argument(
        '--output-path',
        type=str,
        default=f'./output/',
        help='Path to the output directory',
    )

    args = parser.parse_args()

    return args


def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    
    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)
    
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]
    
    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape) * 180 / pi


def main(args):
    input_path = os.path.join(args.input_path, 'bvh_input', 'pose_json')
    output_path = args.output_path


    for dirname in os.listdir(input_path):
        dirpath = os.path.join(input_path, dirname)
        if not os.path.isdir(dirpath):
            continue
        filepath = os.path.join(output_path, f'{dirname}.bvh')
        with open(filepath, 'w') as result_file:
            with open('./output_header.txt', 'r') as header_file:
                result_file.write(header_file.read())
            for filename in os.listdir(dirpath):
                if filename[0] == '0':
                    with open(os.path.join(dirpath, filename),
                              'r') as json_file:
                        info = load(json_file)
                        result_file.write(' '.join(map(str, info['root_pos'])))
                        result_file.write(' ')
                        for quat in info['local_quat']:
                            result_file.write(
                                ' '.join(map(str, qeuler(
                                    torch.Tensor(quat), 'zyx').tolist()[::-1])))
                            result_file.write(' ')
                        result_file.write('\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)