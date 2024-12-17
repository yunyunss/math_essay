import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 로젠브록 함수 정의
def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

# 로젠브록 함수의 기울기(gradient) 정의
def rosenbrock_gradient(x, y, a=1, b=100):
    df_dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    df_dy = 2 * b * (y - x**2)
    return np.array([df_dx, df_dy])

# 경사하강법 구현
def gradient_descent(learning_rate=0.0001, num_iterations=100000, start_point=np.array([0, 0])):
    points = [start_point]  # 점들의 경로를 저장할 리스트
    point = start_point

    for _ in range(num_iterations):
        gradient = rosenbrock_gradient(point[0], point[1])
        point = point - learning_rate * gradient  # 점 업데이트
        points.append(point)

    return np.array(points)

# x, y 범위 설정
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
x, y = np.meshgrid(x, y)

# z 값 계산
z = rosenbrock(x, y)

# 경사하강법으로 점의 경로 구하기
start_point = np.array([2, -1])  # 초기 점
points = gradient_descent(start_point=start_point)

# 3D 플롯 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3D 표면 그래프 그리기
ax.plot_surface(x, y, z, cmap='inferno', alpha=0.7)

# 경사하강법 경로 그리기
ax.plot(points[:, 0], points[:, 1], rosenbrock(points[:, 0], points[:, 1]), color='r', marker='o', markersize=3, linestyle='-', label='Gradient Descent Path')

# 레이블 설정
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 타이틀 설정
ax.set_title('Gradient Descent on Rosenbrock Function')

# 그래프 출력
plt.legend()
plt.show()
