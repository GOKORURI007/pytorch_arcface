import timeit

import torch

from arcface import ArcFace

device = "cuda" if torch.cuda.is_available() else "cpu"
label = torch.randint(0, 9999, (50,)).to(device)
embed = torch.randn((50, 512)).to(device)
model = ArcFace(512, 10000, 30, 0.2, False).to(device)


def speed_test_unit1():
    model.forward(embed, label)


def speed_test_unit2():
    model.forward2(embed, label)


def speed_test_unit3():
    model.forward3(embed, label)


def speed_test(repeat=100):
    time_cost1 = timeit.repeat(stmt="speed_test_unit1()", setup="from __main__ import speed_test_unit1", number=100,
                               repeat=repeat)
    time_cost2 = timeit.repeat(stmt="speed_test_unit2()", setup="from __main__ import speed_test_unit2", number=100,
                               repeat=repeat)
    time_cost3 = timeit.repeat(stmt="speed_test_unit3()", setup="from __main__ import speed_test_unit3", number=100,
                               repeat=repeat)
    print("time_cost1 = {} ms\n"
          "time_cost2 = {} ms\n"
          "time_cost3 = {} ms".format(min(time_cost1) * 1000,
                                      min(time_cost2) * 1000,
                                      min(time_cost3) * 1000))


def equal_test():
    with torch.set_grad_enabled(False):
        loss1 = model.forward(embed, label)
        loss2 = model.forward2(embed, label)
        loss3 = model.forward3(embed, label)
    print("loss1: {}".format(loss1))
    print("loss2: {}".format(loss2))
    print("loss3: {}".format(loss3))


if __name__ == '__main__':
    speed_test(100)
    equal_test()
