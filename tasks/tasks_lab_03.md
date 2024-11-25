# Lab 3: Tasks

## Task 1

**Objective**: Implement Fixed Point Iteration to find the root of $f(x) = cos(x) - x$.

**Instructions**:
1. **Define** the function $ g(x) = \cos(x) $.
2. **Set** an initial guess $ x_0 = 0 $.
3. **Implement** Fixed Point Iteration for a maximum of 100 iterations or until convergence ($ |x_{n+1} - x_n| < 10^{-7} $). Store each approximation.
4. **Plot** the function $ g(x) $ and mark the iterations with red scatter points.
5. **Plot** $f(x)$ and $g(x)$ from -1 to 2, marking iterations with red scatter points. 
6. **Output** the found root and display the plot.

## Task 2

**Objective**: Implement Newton's Method to find the root of $ f(x) = e^x - 2 $.

**Instructions**:
1. **Define** the function $ f(x) $ and its derivative $ f'(x) $.
2. **Set** an initial guess $ x_0 = 0.5 $.
3. **Implement** Newton's Method for a maximum of 100 iterations or until convergence ($ |x_{n+1} - x_n| < 10^{-7} $). Store each approximation.
4. **Plot** $ f(x) $ from -1 to 2, marking iterations with red scatter points.
5. **Output** the found root and display the plot.