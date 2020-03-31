from RL_game import *


def custom():
    a = [0] * (6 - 1) + [1]
    print(a)


if __name__ == "__main__":
    # custom()

    GAME_LENGTH = 6  # 长度
    MAX_EPISODES = 15  # 学习轮数

    game = RLGame(GAME_LENGTH)
    for i in range(MAX_EPISODES):
        print("Episode %d :" % i)
        game.reset()
        q_table = game.learn(to_print=True)
        print(q_table)





