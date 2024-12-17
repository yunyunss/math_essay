import torch, random, time, psutil
from memory_profiler import memory_usage


def logging_time_memory_cpu(original_fn):
    def wrapper_fn(*args, **kwargs):
        # 메모리 사용량 측정을 위한 함수
        def wrapped_fn():
            return original_fn(*args, **kwargs)

        # 시간 측정
        start_time = time.time()
        
        # CPU 사용량 측정 (실행 직전 상태)
        cpu_start = psutil.cpu_percent(interval=None)

        # 메모리 사용량 측정
        mem_usage = memory_usage(wrapped_fn)  # 메모리 사용량 추적

        # 실행
        result = wrapped_fn()

        # 시간 측정 종료
        end_time = time.time()

        # CPU 사용량 측정 (실행 후 상태)
        cpu_end = psutil.cpu_percent(interval=None)

        # 결과 출력
        print(f"WorkingTime[{original_fn.__name__}]: {end_time - start_time:.6f} sec")
        print(f"MemoryUsage[{original_fn.__name__}]: {max(mem_usage) - min(mem_usage):.2f} MB")
        if cpu_end - cpu_start >= 0:
            print(f"CPUUsage[{original_fn.__name__}]: {cpu_end - cpu_start:.2f}%")
        else:
            print(f"CPUUsage[{original_fn.__name__}]: 0%")

        return result

    return wrapper_fn


def rosenbrock(x, y):
    """
    PyTorch를 사용한 로젠브록 함수 구현.
    x, y는 torch.Tensor여야 함.
    """
    a = 1
    b = 100
    return (a - x)**2 + b * (y - x**2)**2

def complicated_func(x, y):
    return x ** 2 * y

@logging_time_memory_cpu
def newton_method(x0, y0, func, tol=0.0001, max_iter=100000):
    """
    PyTorch 자동 미분을 사용한 뉴턴법 최적화 (2차 미분 직접 계산).

    Parameters:
        x0, y0 (float): 초기 위치
        tol (float): 허용 오차
        max_iter (int): 최대 반복 횟수

    Returns:
        (float, float): 최적화된 x, y 좌표
    """
    # x, y를 torch.Tensor로 초기화
    x = torch.tensor(x0, requires_grad=True, dtype=torch.float64)
    y = torch.tensor(y0, requires_grad=True, dtype=torch.float64)

    for i in range(max_iter):
        # 로젠브록 함수 값 계산
        f = func(x, y)

        # Gradient 계산 (1차 도함수)
        grad = torch.autograd.grad(f, (x, y), create_graph=True)
        grad_vector = torch.stack(grad)

        # 종료 조건: Gradient 크기 확인
        if torch.linalg.norm(grad_vector) < tol:
            return x.item(), y.item(), i

        # 직접 2차 미분 계산 (Hessian 행렬 구성)
        h11 = torch.autograd.grad(grad[0], x, retain_graph=True)[0]
        h12 = torch.autograd.grad(grad[0], y, retain_graph=True)[0]
        h21 = torch.autograd.grad(grad[1], x, retain_graph=True)[0]
        h22 = torch.autograd.grad(grad[1], y, retain_graph=True)[0]

        hess_matrix = torch.tensor([[h11, h12], [h21, h22]])
        hess_matrix += torch.eye(hess_matrix.size(0)) * 1e-6  # 작은 정규화 추가


        # 뉴턴 업데이트
        step = torch.linalg.solve(hess_matrix, -grad_vector)

        # x, y 업데이트
        x = x + step[0]
        y = y + step[1]

    print("Reached maximum iterations without convergence.")
    return x.item(), y.item()

@logging_time_memory_cpu
def gradient_descent(x0, y0, func, tol=0.0001, max_iter=100000, learning_late=0.002):
    """
    PyTorch 자동 미분을 사용한 경사하강법 최적화.

    Parameters:
        x0, y0 (float): 초기 위치
        tol (float): 허용 오차
        max_iter (int): 최대 반복 횟수
        learning_late (float) : 학습률

    Returns:
        (float, float): 최적화된 x, y 좌표
    """
    # x, y를 torch.Tensor로 초기화
    x = torch.tensor(x0, requires_grad=True, dtype=torch.float64)
    y = torch.tensor(y0, requires_grad=True, dtype=torch.float64)
    
    for i in range(max_iter):
        f = func(x, y)
        
        grad = torch.autograd.grad(f, (x, y), create_graph=True)
        grad_vector = torch.stack(grad)
        
        if torch.linalg.norm(grad_vector) < tol:
            return x.item(), y.item(), i
        
        x = x - grad[0] * learning_late
        y = y - grad[1] * learning_late
    print("Reached maximum iterations without convergence.")
    return x.item(), y.item()
    
# 초기 위치 설정
x0, y0 = random.random(), random.random()

# 뉴턴법 실행
newton_optimized_x, newton_optimized_y, i = newton_method(x0, y0, complicated_func)
if i != 1000: print(f"Converged after {i} iterations.")
descent_optimized_x, descent_optimized_y, i = gradient_descent(x0, y0, complicated_func)
if i != 1000: print(f"Converged after {i} iterations.")
print(f"Newton optimized position: x = {newton_optimized_x}, y = {newton_optimized_y}")
print(f"Descent optimized position: x = {descent_optimized_x}, y = {descent_optimized_y}")
