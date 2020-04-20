import matplotlib.pyplot as plt
import numpy as np


class KMeans:
    def __init__(self, pt_num=100, kn_num=5, max_itr=100):
        """
        :param pt_num: 总点数
        :param kn_num: 核数，不要超过定义好的颜色数(20)
        :param max_itr: 最大迭代次数
        """
        self.pt_num = pt_num
        self.kn_num = kn_num
        self.max_itr = max_itr
        self.points = np.random.randn(pt_num, 2)
        self.classes = np.zeros(pt_num)
        self.kernels = np.random.randn(kn_num, 2)
        self.last_kernels = None
        self.colors = ["red", "purple", "green", "blue", "olive",
                       "orange", "darkblue", 'yellowgreen', "pink", "yellow",
                       "brown", "black", "peru", "orchid", "palegreen",
                       "navajowhite", "navy", "oldlace", "rosybrown", "saddlebrown"]

    def is_convergent(self):
        if self.last_kernels is None:
            return False
        else:
            return (self.kernels == self.last_kernels).all()

    def classify(self):
        """
        采用欧氏距离
        """
        for i in range(self.pt_num):
            self.classes[i] = np.argmin(np.sum((self.points[i] - self.kernels) ** 2, axis=1))

    def cal_cost(self):
        """
        按各点和所属簇中心计算欧氏距离作为cost
        """
        cost = 0
        for i in range(self.kn_num):
            cost += np.sum((self.points[self.classes == i] - self.kernels[i]) ** 2)
        return cost

    def cal_new_kernel(self):
        """
        记录上一次迭代的中心坐标，并更新类内平均坐标作为新中心点
        """
        self.last_kernels = self.kernels.copy()
        for i in range(self.kn_num):
            self.kernels[i] = np.average(self.points[self.classes == i], axis=0)

    def run(self):
        self.log(0)
        for i in range(self.max_itr):
            if not self.is_convergent():
                self.classify()
                self.cal_new_kernel()
                self.log(i + 1)
            else:
                self.plot(i)
                print("finish at iteration : " + str(i))
                break

    def plot(self, itr_num):
        plt.figure()
        for i in range(self.kn_num):
            plt.scatter(self.points[self.classes == i, 0], self.points[self.classes == i, 1], label=i, s=20,
                        c=self.colors[i])
            plt.scatter(self.kernels[i, 0], self.kernels[i, 1], s=100,
                        c=self.colors[i])
        plt.title("Iterations : " + str(itr_num))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    def log(self, itr):
        print("iteration : " + str(itr))
        self.plot(itr)
        print("cost: " + str(self.cal_cost()))
        print("kernels: ")
        print(self.kernels)
