import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def gup_test():
	with tf.device('/cpu:0'):
		a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
		b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
	with tf.device('/gpu:1'):
		c = a + b

	# 注意：allow_soft_placement=True表明：计算设备可自行选择，如果没有这个参数，会报错。
	# 因为不是所有的操作都可以被放在GPU上，如果强行将无法放在GPU上的操作指定到GPU上，将会报错。
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
	sess.run(tf.global_variables_initializer())
	print(sess.run(c))


def plot_custom(result, model_name="custom", title="custom"):
	r1_loss = np.array(result[0])
	r1_acc = np.array(result[1])
	r1_time = np.array(result[2])
	x_lebel = np.array(range(1, r1_acc.shape[0] + 1))
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	loss_1 = ax1.plot(x_lebel, r1_loss, color="green", label=model_name + " Model Loss")

	acc_1 = ax2.plot(x_lebel, r1_acc, color="blue", label=model_name + " Model Accuracy")
	for i in x_lebel:
		ax2.text(i, r1_acc[i-1], str(round(r1_time[i-1], 1))+"s",
				family='serif', style='italic', ha='right', fontsize=6, wrap=True)
	ax1.set_xlabel("iteration")
	ax1.set_ylabel("Loss")
	ax2.set_ylabel("Accuracy")
	curves = loss_1 + acc_1
	labels = [l.get_label() for l in curves]
	# 注意这里不能用可变参数来传curves和labels
	plt.legend(curves, labels, loc="center right")
	plt.savefig(title + "_iteration_" + str(r1_acc.shape[0]))
	plt.show()


def plot(result1, result2, model_name_1="first", model_name_2="second", title=""):
	r1_loss = np.array(result1[0])
	r1_acc = np.array(result1[1])
	r1_time = np.array(result1[2])

	r2_loss = np.array(result2[0])
	r2_acc = np.array(result2[1])
	r2_time = np.array(result2[2])

	x_lebel = np.array(range(1, r1_acc.shape[0] + 1))
	# 绘制曲线
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	loss_1 = ax1.plot(x_lebel, r1_loss, color="green", label=model_name_1+" Model Loss")
	loss_2 = ax1.plot(x_lebel, r2_loss, color="red", label=model_name_2+" Model Loss")

	acc_1 = ax2.plot(x_lebel, r1_acc, color="blue", label=model_name_1+" Model Accuracy")
	acc_2 = ax2.plot(x_lebel, r2_acc, color="skyblue", label=model_name_2+" Model Accuracy")
	for i in x_lebel:
		ax2.text(i, r1_acc[i-1], str(round(r1_time[i-1], 1))+"s",
				family='serif', style='italic', ha='right', fontsize=6, wrap=True)
	for i in x_lebel:
		ax2.text(i, r2_acc[i-1], str(round(r2_time[i-1], 1))+"s",
				family='serif', style='italic', ha='right', fontsize=6, wrap=True)

	ax1.set_xlabel("iteration")
	ax1.set_ylabel("Loss")
	ax2.set_ylabel("Accuracy")
	# 合并图例
	curves = loss_1 + loss_2 + acc_1 + acc_2
	labels = [l.get_label() for l in curves]
	# 注意这里不能用可变参数来传curves和labels
	plt.legend(curves, labels, loc="center right")
	plt.savefig(title+"_iteration_" + str(r1_acc.shape[0]))
	plt.show()