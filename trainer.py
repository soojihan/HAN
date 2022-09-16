import os

from depression_classifier import model_training
from data_loader import load_abs_path

if __name__ == '__main__':

    import optparse

    parser = optparse.OptionParser()

    parser.add_option('-t', '--trainset',
                      dest="trainset",
                      help="train set path", default=None)

    parser.add_option('-d', '--heldout',
                      dest="heldout",
                      help="heldout dataset csv file", default=None)

    parser.add_option('-e', '--evaluationset',
                      dest="evalset",
                      help="evaluation/test dataset csv file", default=None)

    parser.add_option('--post_dir',
                      dest="postdir",
                      help="directory where posts are saved", default=None)

    parser.add_option('-p', '--model_file_prefix',
                      dest="model_file_prefix",
                      help="model file prefix for model weight output file", default=None)

    parser.add_option('-g', '--n_gpu',
                      dest="n_gpu",
                      help="gpu device(s) to use (-1: no gpu, 0: 1 gpu). only support int value for device no.",
                      default=-1)


    parser.add_option('--epochs', dest="num_epochs", help="set num_epochs for training", default=2)


    parser.add_option("--max_post_size", dest="max_post_size_option", help="maximum post size (default 200)",
                      default=200)


    options, args = parser.parse_args()

    train_set_path = options.trainset
    heldout_set_path = options.heldout
    evaluation_data_path = options.evalset
    post_data_dir = options.postdir
    model_file_prefix = options.model_file_prefix
    no_gpu = int(options.n_gpu)

    num_epochs = int(options.num_epochs)
    max_post_size_option = int(options.max_post_size_option)

    print("================= model settings ========================")
    print("trainset file path: ", train_set_path)
    print("heldout file path: ", heldout_set_path)
    print("evaluation set path: ", evaluation_data_path)
    print("post data dir: ", post_data_dir)
    print("model file prefix: ", model_file_prefix)
    print("gpu device: ", no_gpu)

    print("num_epochs: ", num_epochs)
    print("max_post_size_option: ", max_post_size_option)
    print("============================================================")

    if no_gpu != -1:
        # check GPU device usage and help to set suitable GPU device
        print("================================ check current GPU usage =============================")
        import subprocess

        gpu_usage_result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        print(gpu_usage_result.stdout.decode('utf-8'))
        print("======================================================================================")

    # see https://pytorch.org/docs/stable/notes/cuda.html
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(no_gpu)

    if not os.path.isfile(train_set_path):
        raise FileNotFoundError("training dataset csv file [%s] not found!", train_set_path)

    if not os.path.isfile(heldout_set_path):
        raise FileNotFoundError("heldout dataset csv file [%s] not found!", heldout_set_path)

    if not os.path.isfile(evaluation_data_path):
        raise FileNotFoundError("test set csv file [%s] not found!", evaluation_data_path)


    print("training HAN on development dataset [%s] and [%s] with gpu [%s]" %
          (train_set_path, heldout_set_path, no_gpu))

    import depression_classifier
    import data_loader
    from depression_classifier import config_gpu_use

    config_gpu_use(no_gpu)


    data_loader.post_data_dir = post_data_dir

    print("post data directory is set to [%s]" % data_loader.post_data_dir)


    train_batch_size = 128


    print("model training in batches [size: %s]" % train_batch_size)
    model_training(train_set_path, heldout_set_path, evaluation_data_path, no_gpu, train_batch_size,
                   model_file_prefix,
                   num_epochs=num_epochs,
                   max_post_size_option=max_post_size_option)
