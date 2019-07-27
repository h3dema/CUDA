import argparse
import os
import logging
import subprocess
import numpy as np

import nni

LOG = logging.getLogger('RNN')
TENSORBOARD_DIR = os.environ['NNI_OUTPUT_DIR']


def default_params():
    return {'svm_type': 'epsilon-SVR',
            'kernel_type': 'radial basis',
            'degree': 3,
            'gamma': 0.2,
            'coef0': 0,
            'cost': 1,
            'nu': 0.5,
            'svr_epsilon': 0.1,
            'epsilon': 0.001,
            }


def conv_numeric(_x, _default, f=float):
    try:
        v = f(_x)
    except ValueError:
        v = _default
    return v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create and train model on specified dataset')
    parser.add_argument('--dataset', type=str, default='train_dataset.dat', help='datasets path and filename')
    parser.add_argument('--output', type=str, default='output', help='output directory (results)')

    parser.add_argument('--executable', type=str, default='/home/winet/CUDA/src/linux/bin/svm-train-gnu', help='the executable svm-train-gnu full path')
    parser.add_argument('--predict', type=str, default='/home/winet/CUDA/binaries/linux/svm-predict', help='the executable svm-predict full path')

    parser.add_argument('--epsilon', type=float, default=0.001, help='set tolerance of termination criterion')

    parser.add_argument('--verbose', type=int, default=0, help='train verbosity level (0 is less verbosity)')
    parser.add_argument('--debug', action="store_true", help="info debug information")
    parser.add_argument('--log-to-file', type=bool, default=False, help="Save log to file")
    args = vars(parser.parse_args())  # return as dictionary

    args['id'] = nni.get_sequence_id()
    log_level = logging.DEBUG if args['debug'] else logging.INFO
    if args['log_to_file']:
        log_file = os.path.join(args['output'], 'execution-{}.log'.format(args['id']))
        logging.basicConfig(filename=log_file, level=log_level)
    else:
        logging.basicConfig(level=log_level)

    RECEIVED_PARAMS = nni.get_next_parameter()
    has_nni = True
    if RECEIVED_PARAMS is None:
        """this only occurs if you call this python module from the command line with using nnictl
        """
        RECEIVED_PARAMS = default_params()
        has_nni = False
        LOG.info("System is not using NNI")
    args.update(RECEIVED_PARAMS)
    LOG.info(args)

    # define the parameters
    svm_type = 4 if args['svm_type'] == 'nu-SVR' else 3
    kernel_types = ["linear", "polynomial", "radial basis", "sigmoid"]
    kernel_type = 2 if args['kernel_type'] not in kernel_types else kernel_types.index(args['kernel_type'])
    degree = conv_numeric(args['degree'], 3, int)
    gamma = conv_numeric(args['gamma'], 0.2, float)
    coef0 = conv_numeric(args['coef0'], 0, float)
    cost = conv_numeric(args['cost'], 1, float)
    nu = conv_numeric(args['nu'], 0.5, float)
    svr_epsilon = conv_numeric(args['svr_epsilon'], 0.1, float)
    epsilon = conv_numeric(args['epsilon'], 0.001, float)

    if not os.path.exists(args['output']):
        os.makedirs(args['output'], exist_ok=True)

    output = os.path.join(args['output'], 'model-{}.txt'.format(args['id']))

    cmd = "{} -s {} -t {} -d {} -g {} -r {} -c {} -p {} -e {} -h 0 {} {}".format(args['executable'],
                                                                                 svm_type,
                                                                                 kernel_type,
                                                                                 degree,
                                                                                 gamma,
                                                                                 coef0,
                                                                                 cost,
                                                                                 svr_epsilon,
                                                                                 epsilon,
                                                                                 args['dataset'],
                                                                                 output,
                                                                                 )
    LOG.info(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    LOG.info(stdout)

    lines = stdout.decode('utf-8')
    if 'optimization finished' in lines:
        # sucessfully terminated
        predictions = os.path.join(args['output'], 'predict-{}.txt'.format(args['id']))
        cmd = "{} {} {} {}".format(args['predict'], args['dataset'], output, predictions)

        # run predictions
        p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        LOG.info(stdout)

        # find the line with the result
        lines = stdout.decode('utf-8').split('\n')
        r = [_l for _l in lines if 'Mean squared error' in _l]
        if len(r) > 0:
            try:
                loss = float(r[0].split('=')[1].strip().split()[0])
            except (ValueError, IndexError):
                loss = np.Inf  # error
            if has_nni:
                nni.report_final_result(loss)
