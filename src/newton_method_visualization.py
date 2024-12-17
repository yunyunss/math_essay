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

# 로젠브록 함수의 이계도함수(Hessian) 정의
def rosenbrock_hessian(x, y, a=1, b=100):
    d2f_dx2 = 2 - 4 * b * (y - 3 * x**2)
    d2f_dy2 = 2 * b
    d2f_dxdy = -4 * b * x
    
    H = np.array([
        [d2f_dx2, d2f_dxdy],
        [d2f_dxdy, d2f_dy2]
    ])
    return H

# 뉴턴법 구현
def newton_method(num_iterations=10, start_point=np.array([0, 0]), a=1, b=100):
    points = [start_point]
    point = start_point
    
    for _ in range(num_iterations):
        gradient = rosenbrock_gradient(point[0], point[1], a, b)
        hessian = rosenbrock_hessian(point[0], point[1], a, b)
        
        # 헤시안 행렬의 역행렬 계산 및 점 업데이트
        try:
            hessian_inv = np.linalg.inv(hessian)
            point = point - hessian_inv @ gradient
        except np.linalg.LinAlgError:
            print("Hessian is singular at iteration", _)
            break
        
        points.append(point)
    
    return np.array(points)

# x, y 범위 설정
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
x, y = np.meshgrid(x, y)

# z 값 계산
z = rosenbrock(x, y)

# 뉴턴법으로 점의 경로 구하기
start_point = np.array([1.5, -1.5])  # 초기 점
points = newton_method(start_point=start_point)

# 3D 플롯 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3D 표면 그래프 그리기
ax.plot_surface(x, y, z, cmap='inferno', alpha=0.7)

# 뉴턴법 경로 그리기
ax.plot(points[:, 0], points[:, 1], rosenbrock(points[:, 0], points[:, 1]),
        color='r', marker='o', markersize=3, linestyle='-', label="Newton's Method Path")

# 레이블 설정
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 타이틀 설정
ax.set_title("Newton's Method on Rosenbrock Function")

# 그래프 출력
plt.legend()
plt.show()
