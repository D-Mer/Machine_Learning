from KMeans import *


def custom():
    a = np.random.randn(100, 2)
    colors = ["red", "blue", "green"]
    cs = []
    for i in range(len(a)):
        cs.append(colors[int(3 * np.random.random())])
    plt.figure()
    cs = np.array(cs)
    for i in colors:
        plt.scatter(a[cs == i, 0], a[cs == i, 1], label=i, c=i)
    plt.title('Scatter')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    print(a)


def test():
    for i in range(10):
        print(int(np.random.random() * 5))


if __name__ == "__main__":
    # custom()
    # test()
    k_num = int(np.random.random() * 10) + 3
    print("kernel num : " + str(k_num))
    km = KMeans(kn_num=k_num)
    km.run()
