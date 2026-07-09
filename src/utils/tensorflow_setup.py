import os
import random
import numpy as np
import tensorflow as tf

from config.config import SEED

# Hide TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.get_logger().setLevel("ERROR")


def setup_tensorflow():

    random.seed(SEED)

    np.random.seed(SEED)

    tf.random.set_seed(SEED)