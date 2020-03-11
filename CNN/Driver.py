from functions import _2_2_model, _1_1_model
from Tools import plot, plot_custom


def different_dropout():
    """
    不同的keep_prob值的对照试验
    采用2层卷积 + 2层全连接的结构
    第一个是keep_prob = 0.7的情况
    第二个是keep_prob = 1.0的情况

    """
    conv1_shape = [5, 5, 1, 20]
    conv2_shape = [5, 5, 20, 50]
    fc1_shape = [7 * 7 * 50, 500]
    fc2_shape = [500, 10]
    learning_rate = 0.001
    keep_prob = 0.7  # -------关键参数--0.7
    max_iteration = 20
    result1 = _2_2_model(conv1_shape, conv2_shape, fc1_shape, fc2_shape,
                         learning_rate=learning_rate, keep=keep_prob,
                         max_iteration=max_iteration, save_result=True)
    conv1_shape = [5, 5, 1, 20]
    conv2_shape = [5, 5, 20, 50]
    fc1_shape = [7 * 7 * 50, 500]
    fc2_shape = [500, 10]
    learning_rate = 0.001
    keep_prob = 1.0  # -------关键参数--1.0
    max_iteration = 20
    result2 = _2_2_model(conv1_shape, conv2_shape, fc1_shape, fc2_shape,
                         learning_rate=learning_rate, keep=keep_prob,
                         max_iteration=max_iteration, save_result=True)
    plot(result1, result2, "dropout_0.7", "dropout_1.0", "different_dropout")


def different_layer_num():
    """
    不同的网络层结构的对照试验
    第一个是采用1层卷积 + 1层全连接的结构的情况
    第二个是采用2层卷积 + 2层全连接的结构的情况
    """
    conv1_shape = [3, 3, 1, 20]     # -------关键参数--卷积层1
    fc1_shape = [14 * 14 * 20, 10]  # -------关键参数--全连接层1
    learning_rate = 0.001
    keep_prob = 0.7
    max_iteration = 20
    result1 = _1_1_model(conv1_shape, fc1_shape,
                         learning_rate=learning_rate, keep=keep_prob,
                         max_iteration=max_iteration, save_result=True)
    conv1_shape = [5, 5, 1, 20]     # -------关键参数--卷积层1
    conv2_shape = [3, 3, 20, 50]    # -------关键参数--卷积层2
    fc1_shape = [7 * 7 * 50, 500]   # -------关键参数--全连接层1
    fc2_shape = [500, 10]           # -------关键参数--全连接层2
    learning_rate = 0.001
    keep_prob = 0.7
    max_iteration = 20
    result2 = _2_2_model(conv1_shape, conv2_shape, fc1_shape, fc2_shape,
                         learning_rate=learning_rate, keep=keep_prob,
                         max_iteration=max_iteration, save_result=True)
    plot(result1, result2, "1+1", "2+2", "different_layer_num")


def different_filter_size():
    """
    不同的卷积核大小的对照试验
    第一个是采用 3 * 3 的卷积核的情况
    第二个是采用 5 * 5 的卷积核的情况
    """
    conv1_shape = [3, 3, 1, 20]     # -------关键参数--卷积核，3 * 3
    conv2_shape = [3, 3, 20, 50]    # -------关键参数--卷积核，3 * 3
    fc1_shape = [7 * 7 * 50, 500]
    fc2_shape = [500, 10]
    learning_rate = 0.001
    keep_prob = 0.7
    max_iteration = 20
    result1 = _2_2_model(conv1_shape, conv2_shape, fc1_shape, fc2_shape,
                         learning_rate=learning_rate, keep=keep_prob,
                         max_iteration=max_iteration, save_result=True)
    conv1_shape = [5, 5, 1, 20]     # -------关键参数--卷积核，5 * 5
    conv2_shape = [5, 5, 20, 50]    # -------关键参数--卷积核，5 * 5
    fc1_shape = [7 * 7 * 50, 500]
    fc2_shape = [500, 10]
    learning_rate = 0.001
    keep_prob = 0.7
    max_iteration = 20
    result2 = _2_2_model(conv1_shape, conv2_shape, fc1_shape, fc2_shape,
                         learning_rate=learning_rate, keep=keep_prob,
                         max_iteration=max_iteration, save_result=True)
    plot(result1, result2, "3*3 filter", "5*5 filter", "different_filter_size")


def different_learning_rate():
    """
    不同的学习速率大小的对照试验
    第一个是采用 0.001 的速率的情况
    第二个是采用 0.0005 的速率的情况
    """
    conv1_shape = [5, 5, 1, 20]
    conv2_shape = [5, 5, 20, 50]
    fc1_shape = [7 * 7 * 50, 500]
    fc2_shape = [500, 10]
    learning_rate = 0.001   # -------关键参数--学习率，0.001
    keep_prob = 0.7
    max_iteration = 20
    result1 = _2_2_model(conv1_shape, conv2_shape, fc1_shape, fc2_shape,
                         learning_rate=learning_rate, keep=keep_prob,
                         max_iteration=max_iteration, save_result=True)
    conv1_shape = [5, 5, 1, 20]
    conv2_shape = [5, 5, 20, 50]
    fc1_shape = [7 * 7 * 50, 500]
    fc2_shape = [500, 10]
    learning_rate = 0.0005  # -------关键参数--学习率，0.0005
    keep_prob = 0.7
    max_iteration = 20
    result2 = _2_2_model(conv1_shape, conv2_shape, fc1_shape, fc2_shape,
                         learning_rate=learning_rate, keep=keep_prob,
                         max_iteration=max_iteration, save_result=True)
    plot(result1, result2, "rate0.001", "rate0.0005", "different_learning_rate")


def custom():
    """
    这里自定义参数模型
    """
    conv1_shape = [5, 5, 1, 20]
    conv2_shape = [5, 5, 20, 50]
    fc1_shape = [7 * 7 * 50, 500]
    fc2_shape = [500, 10]
    learning_rate = 0.001
    keep_prob = 0.7
    max_iteration = 2
    result = _2_2_model(conv1_shape, conv2_shape, fc1_shape, fc2_shape,
                         learning_rate=learning_rate, keep=keep_prob,
                         max_iteration=max_iteration, save_result=True)
    plot_custom(result, model_name="test", title="test")


if __name__ == '__main__':
    # different_filter_size()
    # different_layer_num()
    # different_dropout()
    # different_learning_rate()
    custom()
