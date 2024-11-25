## Lab 5

### Task 1: find polynomial approximation for x = -5 to x = 2.1 using newton divided difference method. **Use centered differences.**

$$f(x) = a_0 + a_1(x - x_0) + a_2(x - x_0)(x - x_1) + \cdots + a_n(x - x_0)(x - x_1) \cdots (x - x_{n-1})$$

which can be re-written as

$$f(x) = \sum_{i=0}^{n} a_i n_i(x)$$

where 

$$n_i(x) = \prod_{j=0}^{i-1} (x - x_j)$$

| **x** | **y** |
|-------|-------|
|  -5   |  -2   |
|  -1   |   6   |
|   0   |   1   |
|   2   |   3   |
