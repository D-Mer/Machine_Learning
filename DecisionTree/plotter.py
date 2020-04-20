import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def get_num_leafs(tree):
    """
    计算决策树的叶子数
    :param tree: 决策树
    :return: 叶子数
    """
    # 叶子数
    numLeafs = 0
    # 节点信息
    sides = list(tree.keys())
    firstStr = sides[0]
    # 分支信息
    secondDict = tree[firstStr]

    for key in secondDict.keys():  # 遍历所有分支
        # 子树分支则递归计算
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += get_num_leafs(secondDict[key])
        # 叶子分支则叶子数+1
        else:
            numLeafs += 1

    return numLeafs


def get_tree_depth(tree):
    """
    计算决策树的深度
    :param tree: 决策树
    :return: 决策树深度
    """
    # 最大深度
    maxDepth = 0
    # 节点信息
    sides = list(tree.keys())
    firstStr = sides[0]
    # 分支信息
    secondDict = tree[firstStr]

    for key in secondDict.keys():  # 遍历所有分支
        # 子树分支则递归计算
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + get_tree_depth(secondDict[key])
        # 叶子分支则叶子数+1
        else:
            thisDepth = 1

        # 更新最大深度
        if thisDepth > maxDepth: maxDepth = thisDepth

    return maxDepth


def plot_node(node_txt, center_pt, parent_pt, node_type):
    """
    在图形界面中显示输入参数指定样式的线段(终端带节点)
    :param node_txt: 终端节点显示内容
    :param center_pt: 终端节点坐标
    :param parent_pt: 起始节点坐标
    :param node_type: 终端节点样式
    """
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt, textcoords='axes fraction',
                             va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


def plot_mid_text(center_pt, parent_pt, txt_string):
    """
    在图形界面指定位置(cntrPt和parentPt中间)显示文本内容(txtString)
    :param center_pt: 终端节点坐标
    :param parent_pt: 起始节点坐标
    :param txt_string: 待显示文本内容
    """
    xMid = (parent_pt[0] - center_pt[0]) / 2.0 + center_pt[0]
    yMid = (parent_pt[1] - center_pt[1]) / 2.0 + center_pt[1]
    create_plot.ax1.text(xMid, yMid, txt_string, va="center", ha="center", rotation=30)


def plot_tree(tree, parent_pt, node_txt):
    """
    在图形界面绘制决策树
    :param tree: 决策树
    :param parent_pt: 根节点坐标
    :param node_txt: 根节点描述文本
    """
    # 当前树的叶子数
    numLeafs = get_num_leafs(tree)
    # 当前树的节点信息
    sides = list(tree.keys())
    firstStr = sides[0]

    # 定位第一棵子树的位置(这是蛋疼的一部分)
    cntrPt = (plot_tree.xOff + (1.0 + float(numLeafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)

    # 绘制当前节点到子树节点(含子树节点)的信息
    plot_mid_text(cntrPt, parent_pt, node_txt)
    plot_node(firstStr, cntrPt, parent_pt, decisionNode)

    # 获取子树信息
    secondDict = tree[firstStr]
    # 开始绘制子树，纵坐标-1。
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD

    for key in secondDict.keys():  # 遍历所有分支
        # 子树分支则递归
        if type(secondDict[key]).__name__ == 'dict':
            plot_tree(secondDict[key], cntrPt, str(key))
        # 叶子分支则直接绘制
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(secondDict[key], (plot_tree.xOff, plot_tree.yOff), cntrPt, leafNode)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntrPt, str(key))

    # 子树绘制完毕，纵坐标+1。
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


def create_plot(tree):
    """
    以树形图显示决策树
    :param tree: 构建完的决策树
    """
    # 创建新的图像并清空 - 无横纵坐标
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 树的总宽度 高度
    plot_tree.totalW = float(get_num_leafs(tree))
    plot_tree.totalD = float(get_tree_depth(tree))
    # 当前绘制节点的坐标
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    # 绘制决策树
    plot_tree(tree, (0.5, 1.0), '')
    plt.show()
