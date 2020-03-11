import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data


def init_layer_variable(shape):
	"""
	初始化权值 W 和偏差值 b
	:param shape: 如果是卷积核则格式为
		[height, weight, in_channel, out_channel]，否则为 [in, out]
	:return 标准差为0.1的初始化权值W和初始值为0.1的偏差值b
	"""
	weight = tf.truncated_normal(shape, stddev=0.1)
	bias = tf.constant(0.1, shape=[shape[len(shape) - 1]])
	return tf.Variable(weight), tf.Variable(bias)


def conv2d(input, filter, strides=None, padding="SAME"):
	"""
	进行卷积计算
	:param input: 输入的张量
	:param filter: 卷积核(过滤器)
	:param strides: 步长 [batch, height, weight, channel]
	:param padding: 填充方式，默认为"SAME"
	:return: 卷积结果
	"""
	if strides is None:
		strides = [1, 1, 1, 1]
	return tf.nn.conv2d(input, filter, strides=strides, padding=padding)


def max_pool(value, kernel_size=None, strides=None, padding="SAME"):
	"""
	池化，默认将输入的宽高缩小到一半
	:param value: 输入
	:param kernel_size: 池化卷积核形状 [batch=1, height=2, weight=2, channel=1]
	:param strides: 步长 [batch=1, height=2, weight=2, channel=1]
	:param padding: 填充方式，默认为"SAME"
	:return: 输出
	"""
	if strides is None:
		strides = [1, 2, 2, 1]
	if kernel_size is None:
		kernel_size = [1, 2, 2, 1]
	return tf.nn.max_pool(value, ksize=kernel_size, strides=strides, padding=padding)


def _1_1_model(conv1_shape, fc1_shape,
				learning_rate=0.001, keep=0.7,
				max_iteration=20, save_result=False):
	# tensorflow内置了解析数据源的函数，因此这里直接调用，省去了数据源的解析
	# 默认训练集为4w，交叉验证集为1w，测试集1w
	dataset = input_data.read_data_sets('dataset', validation_size=10000, one_hot=True)
	# 设置batch值为400，每轮共学习100次
	batch_size = 400
	n_batch = dataset.train.num_examples // batch_size

	# 先声明占位符
	# MNIST数据的输入层为28*28=784的图片矩阵
	# 输出层为1*10的矩阵(使用了one_hot来分类)
	x = tf.placeholder(tf.float32, [None, 784])
	y = tf.placeholder(tf.float32, [None, 10])
	keep_prob = tf.placeholder(tf.float32)
	# 改变x的格式转为 NHWC 标准格式 [batch, height, width, channels]
	x_image = tf.reshape(x, [-1, 28, 28, 1])

	# 初始化第一层卷积核 W 和第一层偏差值 b
	W_conv1, b_conv1 = init_layer_variable(conv1_shape)
	# 把输入的图片矩阵x和权值向量进行卷积，再加上偏置值得到Z
	Z_conv1 = conv2d(x_image, W_conv1) + b_conv1
	# 通过relu激活
	A_conv1 = tf.nn.relu(Z_conv1)
	# 进行max_pooling 池化层
	A_pool1 = max_pool(A_conv1)

	# -----------------全连接层--------------------------

	# 初始化第一个全连接层的权值并将池化层扁平化
	W_fc1, b_fc1 = init_layer_variable(fc1_shape)
	A_pool2_flat = tf.reshape(A_pool1, [-1, fc1_shape[0]])
	# 求第一个全连接层的输出
	Z_fc1 = tf.matmul(A_pool2_flat, W_fc1) + b_fc1
	prediction = tf.nn.softmax(Z_fc1)

	# 计算交叉熵代价函数
	cross_entropy = -tf.reduce_sum(y * tf.log(prediction))
	# 使用AdamOptimizer进行优化
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
	# 结果存放在一个布尔列表中(argmax函数返回一维张量中最大的值所在的位置)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
	# 求准确率，输出tf.cast将布尔值转换为float型)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		result = [[], [], []]
		start_time = time.process_time()
		sess.run(tf.global_variables_initializer())
		acc = 0.
		for epoch in range(1, max_iteration + 1):
			for batch in range(n_batch):
				x_batch, y_batch = dataset.train.next_batch(batch_size)
				sess.run(train_step, feed_dict={x: x_batch, y: y_batch, keep_prob: keep})  # 进行迭代训练
			# 测试数据计算出准确率
			loss = sess.run(cross_entropy, feed_dict={x: dataset.test.images, y: dataset.test.labels, keep_prob: 1.0})
			acc = sess.run(accuracy, feed_dict={x: dataset.test.images, y: dataset.test.labels, keep_prob: 1.0})
			end_time = time.process_time()
			result[0].append(loss)
			result[1].append(acc)
			result[2].append(end_time-start_time)
			print('Iter:%d, %s s, Loss=%f,Test Accuracy=%f' % (epoch, end_time - start_time, loss, acc))
		if save_result:
			tf.train.Saver().save(sess, save_path="1_1models/saver" + str(acc) + "/model")
	return result


def _2_2_model(conv1_shape, conv2_shape, fc1_shape, fc2_shape,
				learning_rate=0.001, keep=0.7,
				max_iteration=25, save_result=False):
	# tensorflow内置了解析数据源的函数，因此这里直接调用，省去了数据源的解析
	# 默认训练集为4w，交叉验证集为1w，测试集1w
	dataset = input_data.read_data_sets('dataset', validation_size=10000, one_hot=True)
	# 设置batch值为400，每轮共学习100次
	batch_size = 400
	n_batch = dataset.train.num_examples // batch_size

	# 先声明占位符
	# MNIST数据的输入层为28*28=784的图片矩阵
	# 输出层为1*10的矩阵(使用了one_hot来分类)
	x = tf.placeholder(tf.float32, [None, 784])
	y = tf.placeholder(tf.float32, [None, 10])
	keep_prob = tf.placeholder(tf.float32)
	# 改变x的格式转为 NHWC 标准格式 [batch, height, width, channels]
	x_image = tf.reshape(x, [-1, 28, 28, 1])

	# -----------------卷积层--------------------------

	# 初始化第一层卷积核 W 和第一层偏差值 b
	W_conv1, b_conv1 = init_layer_variable(conv1_shape)
	# 把输入的图片矩阵x和权值向量进行卷积，再加上偏置值得到Z
	Z_conv1 = conv2d(x_image, W_conv1) + b_conv1
	# 通过relu激活
	A_conv1 = tf.nn.relu(Z_conv1)
	# 进行max_pooling 池化层
	A_pool1 = max_pool(A_conv1)

	# 初始化第二层卷积核 W 和第二层偏差值 b
	W_conv2, b_conv2 = init_layer_variable(conv2_shape)
	# 把第一个池化层结果和权值向量进行卷积，再加上偏置值
	Z_conv2 = conv2d(A_pool1, W_conv2) + b_conv2
	# 通过relu激活
	A_conv2 = tf.nn.relu(Z_conv2)
	# 进行max_pooling 池化层
	A_pool2 = max_pool(A_conv2)

	# -----------------全连接层--------------------------

	# 初始化第一个全连接层的权值并将池化层扁平化
	W_fc1, b_fc1 = init_layer_variable(fc1_shape)
	A_pool2_flat = tf.reshape(A_pool2, [-1, fc1_shape[0]])
	# 求第一个全连接层的输出
	Z_fc1 = tf.matmul(A_pool2_flat, W_fc1) + b_fc1
	A_fc1 = tf.nn.relu(Z_fc1)
	A_fc1_drop = tf.nn.dropout(A_fc1, keep_prob)

	# 初始化第二个全连接层
	W_fc2, b_fc2 = init_layer_variable(fc2_shape)

	# 用softmax计算输出
	prediction = tf.nn.softmax(tf.matmul(A_fc1_drop, W_fc2) + b_fc2)

	# 计算交叉熵代价函数
	cross_entropy = -tf.reduce_sum(y * tf.log(prediction))
	# 用AdamOptimizer进行优化
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
	# 结果存放在一个bool列表中(argmax函数返回一维张量中最大的值所在的位置)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
	# 求准确率，输出tf.cast将bool转换为float
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# 训练并输出
	with tf.Session() as sess:
		result = [[], [], []]
		start_time = time.process_time()
		sess.run(tf.global_variables_initializer())
		acc = 0.
		for epoch in range(1, max_iteration + 1):
			for batch in range(n_batch):
				x_batch, y_batch = dataset.train.next_batch(batch_size)
				# 进行迭代训练
				sess.run(train_step,
						feed_dict={x: x_batch, y: y_batch,
								keep_prob: keep})
			# 测试数据计算出准确率
			loss = sess.run(cross_entropy,
						feed_dict={x: dataset.test.images,
								y: dataset.test.labels,
								keep_prob: 1.0})
			acc = sess.run(accuracy,
						feed_dict={x: dataset.test.images,
								y: dataset.test.labels,
								keep_prob: 1.0})
			end_time = time.process_time()
			result[0].append(loss)
			result[1].append(acc)
			result[2].append(end_time - start_time)
			print('Iter:%d, %s s, Loss=%f,Test Accuracy=%f'
					% (epoch, end_time - start_time, loss, acc))
		if save_result:
			train_writer = tf.summary.FileWriter('./log/tarin', sess.graph)
			train_writer.close()
			tf.train.Saver().save(sess,
						save_path="2_2models/saver" + str(acc) + "/model")
	return result



