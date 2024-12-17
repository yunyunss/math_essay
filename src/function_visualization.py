import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# 로젠브록 함수 정의
def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def complicated_func(x, y):
    return math.sin(x)*y*math.log(x+1)*x

# x, y 범위 설정
x = np.linspace(0, 200, 400)
y = np.linspace(-100, 100, 400)
x, y = np.meshgrid(x, y)

# z 값 계산
f = np.vectorize(complicated_func)
z = f(x, y)

# 3D 플롯 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3D 표면 그래프 그리기
ax.plot_surface(x, y, z, cmap='inferno', alpha=0.7)

# 레이블 설정
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 그래프 출력
plt.legend()
plt.show()
