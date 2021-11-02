import timeit

import torch

from arcface import ArcFace

label = torch.randint(0, 999, (50,)).to("cuda")
embed = torch.randn((50, 512)).to("cuda") - 2
model = ArcFace(512, 10000, 30, 0.2, True).to("cuda")


def speed_test_unit():
    model.forward(embed, label)

def speed_test():
    time_cost = timeit.repeat(stmt="speed_test_unit()", setup="from __main__ import speed_test_unit", number=100,
                              repeat=1000)
    print("time_cost = {} ms".format(min(time_cost) * 1000))


def equal_test():
    loss1 = model.forward1(embed, label)
    loss2 = model.forward2(embed, label)
    loss3 = model.forward3(embed, label)
    print("loss1: {}".format(loss1))
    print("loss2: {}".format(loss2))
    print("loss3: {}".format(loss3))


if __name__ == '__main__':
    speed_test()
    equal_test()
