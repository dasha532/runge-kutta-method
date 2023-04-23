import numpy as np
import math
import matplotlib.pyplot as plt

n = 10
a = 0
b = 2
h = (b - a) / n
rez3 = [0] * (n+1)
rez5 = [0] * (n+1)
x1 = [0] * (n+1)
y1 = [0] * (n+1)
y2 = [0] * (n+1)
rez2 = [0] * (n+1)
rez4 = [0] * (n+1)


def f_1(x_1, x_2, t):
    return 3 * x_1 + 2 * x_2 + 3 * math.e ** (2 * t)


def f_2(x_1, x_2, t):
    return x_1 + 2 * x_2 + math.e ** (2 * t)


x_1 = 0
x_2 = -2
rez2[0] = x_1
rez4[0] = x_2
for i in range(0, n+1):
    K11 = round(h * f_1(x_1, x_2, a + i * h), 4)
    K12 = round(h * f_2(x_1, x_2, a + i * h), 4)
    K21 = round(h * f_1(x_1 + K11 / 2, x_2 + K12 / 2, a + i * h + h / 2), 4)
    K22 = round(h * f_2(x_1 + K11 / 2, x_2 + K12 / 2, a + i * h + h / 2), 4)
    K31 = round(h * f_1(x_1 + K21 / 2, x_2 + K22 / 2, a + i * h + h / 2), 4)
    K32 = round(h * f_2(x_1 + K21 / 2, x_2 + K22 / 2, a + i * h + h / 2), 4)
    K41 = round(h * f_1(x_1 + K31, x_2 + K32, a + i * h + h), 4)
    K42 = round(h * f_2(x_1 + K31, x_2 + K32, a + i * h + h), 4)
    x_1 = round((x_1 + (K11 + 2 * K21 + 2 * K31 + K41) / 6.0), 4)
    x_2 = round((x_2 + (K12 + 2 * K22 + 2 * K32 + K42) / 6.0), 4)
    rez3[i] = x_1
    rez5[i] = x_2
    x1[i] = round(a + i * h, 2)
for i in range(0, n):
    rez2[i+1] = rez3[i]
    rez4[i+1] = rez5[i]

for i in range(0, n+1):
    y1[i] = round(math.e ** (x1[i]) - math.e ** (2 * (x1[i])), 4)
    y2[i] = round(-math.e ** (x1[i]) - math.e ** (2 * (x1[i])), 4)
rez = [x1, y1, rez2, y2, rez4]
for j in range(0, n+1):
    print(f'{rez[0][j]},  {rez[1][j]}, {rez[2][j]}, {rez[3][j]}, {rez[4][j]}')

plt.figure(1)
plt.plot(x1, rez2, label='runge_kutta')
plt.plot(x1, y1, label='analitic')
plt.xlabel("x")
plt.ylabel("y")
plt.title("X1")
plt.grid()
plt.figure(2)
plt.plot(x1, rez4, label='runge_kutta')
plt.plot(x1, y2, label='analytic')
plt.xlabel("x")
plt.ylabel("y")
plt.title("X2")
plt.grid()
plt.show()
error1 = abs(np.array(y1)) - abs(np.array(rez2))
print("max error1 {:.5f}".format(max(abs(error1))))
error2 = abs(np.array(y2)) - abs(np.array(rez4))
print("max error1 {:.5f}".format(max(abs(error2))))
