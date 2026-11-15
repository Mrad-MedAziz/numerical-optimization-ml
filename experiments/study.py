import numpy as np
import time
from optimization.logistic_model import LogisticRegression

def generate_data(seed: int, n_samples: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic binary classification dataset with overlapping clusters.
    """
    np.random.seed(seed)
    half = n_samples // 2 
    X_0 = np.random.randn(half, 2) + np.array([-1.0, -1.0])
    X_1 = np.random.randn(half, 2) + np.array([1.0, 1.0])
    X = np.vstack([X_0, X_1])
    y = np.array([0.0] * half + [1.0] * half)
    return X, y

def sweep_learning_rates():
    """
    Test how fast Gradient Ascent converges for different learning rates across 20 seeds.
    """
    learning_rates = [0.05, 0.1, 0.5, 1.0, 2.0, 4.0]
    seeds = range(20)
    
    # Dictionary to store a list of iteration counts for each learning rate
    results = {lr: [] for lr in learning_rates}
    
    for seed in seeds:
        X, y = generate_data(seed)
        
        for lr in learning_rates:
            model = LogisticRegression()
            model.fit_gradient_ascent(X, y, lr=lr, epochs=2000)
            
            actual_epochs = len(model.log_likelihood_history) - 1 #We ignore the starting value of Theta (0)
            results[lr].append(actual_epochs)
            
    print("Median Iterations by Learning Rate (over 20 seeds):")
    for lr in learning_rates:
        median_iters = int(np.median(results[lr]))
        print(f"  lr = {lr:<4} : {median_iters} iterations")

def compare_wall_clock_time():
    """
    Measure the actual milliseconds taken by Newton-Raphson vs Gradient Ascent.
    """
    # Using 1000 samples to make the timing more noticeable
    X, y = generate_data(42, n_samples=1000)
    
    # 1. Warm-up: Run a quick dummy model so NumPy loads into memory
    dummy = LogisticRegression()
    dummy.fit_gradient_ascent(X, y, epochs=5)
    
    print("\n--- Wall-Clock Time Comparison ---")
    
    # 2. Time Newton-Raphson
    model_newton = LogisticRegression()
    
    start_newton = time.perf_counter()
    model_newton.fit_newton(X, y, max_iter=20)
    end_newton = time.perf_counter()
    
    newton_total_time = (end_newton - start_newton) * 1000 # Convert to milliseconds
    
    # 3. Time Gradient Ascent at its absolute best learning rate (lr=4.0)
    model_ga = LogisticRegression()
    
    start_ga = time.perf_counter()
    # Gradient Ascent uses the best learning rate from our sweep
    model_ga.fit_gradient_ascent(X, y, lr=4.0, epochs=2000)
    end_ga = time.perf_counter()
    
    ga_total_time = (end_ga - start_ga) * 1000 
    
    newton_iters = len(model_newton.log_likelihood_history) - 1
    ga_iters = len(model_ga.log_likelihood_history) - 1
    
    print(f"Newton-Raphson:")
    print(f"  Iterations: {newton_iters}")
    print(f"  Total Time: {newton_total_time:.2f} ms")
    
    print(f"\nGradient Ascent (lr=4.0):")
    print(f"  Iterations: {ga_iters}")
    print(f"  Total Time: {ga_total_time:.2f} ms")
    
    if newton_total_time > 0 and ga_total_time > 0:
        print(f"\nTime per Iteration:")
        print(f"  Newton: {newton_total_time / newton_iters:.3f} ms")
        print(f"  Gradient Ascent: {ga_total_time / ga_iters:.3f} ms")

def test_separable_data():
    """
    Test how Tikhonov damping (eps_reg) acts as a regularizer on perfectly separable data.
    """
    print("\n--- Separable Data Damping Test ---")
    
    # 1. Generate perfectly separable data (centers pulled far apart to -3 and +3)
    np.random.seed(42)
    X_0 = np.random.randn(100, 2) + np.array([-3.0, -3.0])
    X_1 = np.random.randn(100, 2) + np.array([3.0, 3.0])
    X = np.vstack([X_0, X_1])
    y = np.array([0.0] * 100 + [1.0] * 100)
    
    damping_values = [0.0, 1e-8, 1e-4]
    
    for eps in damping_values:
        model = LogisticRegression()
        model.fit_newton(X, y, eps_reg=eps)
        theta_norm = np.linalg.norm(model.theta)
        
        print(f"Damping eps = {eps:<5} -> Parameter Euclidean Norm: {theta_norm:.2f}")
    
if __name__ == "__main__":
    sweep_learning_rates()
    compare_wall_clock_time()
    test_separable_data()