from math import log

import numpy as np

from plotter import create_plot


def load_file(filename):
    """
    加载数据集
    :param filename: 文件名
    :return: 数据集矩阵
    """
    f = open(filename)
    dataset = np.array([a.strip("\n").split("\t") for a in f.readlines()])
    print(dataset)
    return dataset


class ID3Tree:
    def __init__(self, dataset, headers):
        """
        :param dataset: 数据集
        :param headers: 属性列名
        """
        self.dataset = dataset
        self.headers = headers

    def cal_entropy(self, dataset):
        """
        计算指定数据集的熵
        :param dataset: 数据集，最后一列是类别标签
        :return: 熵
        """
        # 数据集总项数
        data_len = float(len(dataset))
        # 标签计数对象初始化
        label_counts = {}
        for data in dataset:
            # 最后一列是分类标签
            label = data[-1]
            # 为每一类数据计数
            if label not in label_counts.keys():
                label_counts[label] = 0
            label_counts[label] += 1
        entropy = 0.0
        for key in label_counts.keys():
            prop = label_counts[key] / data_len
            entropy -= prop * log(prop, 2)
        return entropy

    def max_count_label(self):
        """
        计算当前数据集下的最多的标签
        :return: 最多的标签名
        """
        label_counts = {}
        for label in self.dataset[:, -1]:
            if label_counts[label] is None:
                label_counts[label] = 0
            label_counts += 1
        max_k = label_counts.items()
        max_v = None
        for item in label_counts.items():
            if max_v is None or max_v < item[1]:
                max_k = item[0]
                max_v = item[1]
        return max_k

    def filter_horizontal(self, col, feature):
        """
        水平过滤数据集
        :param col: 需要用来选择行的属性列
        :param feature: 需要选择的属性
        :return: 包含指定属性的行
        """
        a = []
        for i in self.dataset:
            if i[col] == feature:
                a.append(i)
        return np.array(a)

    def delete_col(self, col):
        """
        删除指定列
        :param col: 列号
        :return: 删除指定列后的数据集
        """
        a = []
        for i in range(len(self.dataset)):
            if i != col:
                a.append(i)
        return self.dataset[:, a]

    def cal_best_header_col(self):
        """
        选择最佳属性列
        :return: 最佳属性列号
        """
        col_len = len(self.headers) - 1
        # 计算根节点的信息熵
        root_env = self.cal_entropy(self.dataset)
        max_info_gain = 0.0
        best_header = None
        # 对每一列计算信息熵
        for i in range(col_len):
            features = set(self.dataset[:, i])
            entropy = 0.0
            for feature in features:
                sub_data = self.filter_horizontal(i, feature)
                probability = len(sub_data) / float(len(self.dataset))
                entropy += probability * self.cal_entropy(sub_data)
            info_gain = root_env - entropy
            # 选择信息增益最大的列
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_header = i
        return best_header

    def calculate_tree(self):
        """
        计算ID3决策树
        :return: map形式的决策树
        """
        env = self.cal_entropy(self.dataset)
        # 熵等于0意味着所有样本都是同一类型的样本
        if env == 0:
            return self.dataset[0][-1]
        # 没有属性列可分时返回最多的类型标签
        if len(self.headers) == 0:
            return self.max_count_label()
        # 获取最大熵增益列
        best_header_col = self.cal_best_header_col()
        best_header = self.headers[best_header_col]
        # 教程里的树结构是这样的嵌套字典
        tree = {best_header: {}}
        features = set(self.dataset[:, best_header_col])
        for feature in features:
            # 创建子数据集
            sub_dataset = self.filter_horizontal(best_header_col, feature)
            sub_dataset = np.delete(sub_dataset, best_header_col, axis=1)
            sub_headers = self.headers.copy()
            del sub_headers[best_header_col]
            subID3 = ID3Tree(sub_dataset, sub_headers)
            # 递归构建决策树
            tree[best_header][feature] = subID3.calculate_tree()
        return tree


if __name__ == '__main__':
    filename = "lenses.txt"
    headers = ["age", "prescript", "astigmatic", "tearRate", "label"]  # 可选属性
    dataset = load_file(filename)
    id3 = ID3Tree(dataset, headers)
    tree = id3.calculate_tree()
    print(tree)
    create_plot(tree)
