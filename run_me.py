# ====================================== Q1 ======================================
print("Q1 RUNNING:\n")

# reference - https://docs.jax.dev/en/latest/_autosummary/jax.scipy.special.logsumexp.html

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

# g(x) = <f(x), u>
def g(x, u):
    return u * logsumexp(x)

def backward_logsumexp(x, u):
    logsumexp_grad = jax.grad(lambda x: g(x, u))
    return logsumexp_grad(x)

def backward_logsumexp_closedform(x, u):
    sm = jnp.exp(x) / jnp.sum(jnp.exp(x))
    return u * sm

# g(A, x) = <solve(A, x), u>
def g_solve(A, x, u):
    return jnp.vdot(jnp.linalg.solve(A, x), u)  # scalar

def backward_solve(A, x, u):
    grad_A, grad_x = jax.grad(lambda A, x: g_solve(A, x, u), argnums=(0, 1))(A, x)
    return grad_A, grad_x

def backward_solve_closedform(A, x, u):
    b = jnp.linalg.solve(A, x)
    AinvT = jnp.linalg.inv(A).T
    grad_x = AinvT @ u
    grad_A = -AinvT @ jnp.outer(u, b)
    return grad_A, grad_x

# g(A) = u * logdet(A)
def g_logdet(A, u):
    sign, logdet = jnp.linalg.slogdet(A)
    return u * logdet

def backward_logdet(A, u):
    grad_A = jax.grad(lambda A: g_logdet(A, u))(A)
    return grad_A

def backward_logdet_closedform(A, u):
    return u * jnp.linalg.inv(A).T


# Run tests to check answers

# logsumexp
x = jnp.array([1.0, 2.0, 3.0])
u = 2.0
grad_auto = backward_logsumexp(x, u)
grad_manual = backward_logsumexp_closedform(x, u)
print("=== logsumexp ===")
print("autodiff:", grad_auto)
print("closed-form:", grad_manual)
print("ans match:", jnp.allclose(grad_auto, grad_manual))
print()

# solve
A = jnp.array([[2.0, 0.5, 0.0],
               [0.0, 3.0, 1.0],
               [0.0, 0.0, 4.0]])
x = jnp.array([1.0, 2.0, 3.0])
u = jnp.array([0.5, -1.0, 2.0])
gradA_auto, gradx_auto = backward_solve(A, x, u)
gradA_manual, gradx_manual = backward_solve_closedform(A, x, u)
print("=== solve ===")
print("grad_x match:", jnp.allclose(gradx_auto, gradx_manual))
print("grad_A match:", jnp.allclose(gradA_auto, gradA_manual))
print()

# logdet
A = jnp.array([[3.0, 1.0],
               [0.0, 2.0]])
u = 2.0
gradA_auto = backward_logdet(A, u)
gradA_manual = backward_logdet_closedform(A, u)
print("=== logdet ===")
print("autodiff:\n", gradA_auto)
print("closed-form:\n", gradA_manual)
print("ans match:", jnp.allclose(gradA_auto, gradA_manual))

print("Q1 DONE RUNNING\n")
# ================================================================================

# ====================================== Q2 ======================================
print("Q2 RUNNING:\n")

# Q2 - Matrix Auto-Diffrenciation
# Using class notes and full code provided
# numpy reference: https://www.w3schools.com/python/numpy/default.asp
# also refered to: https://www.geeksforgeeks.org/python/numpy-tutorial/

import numpy as np


# Op class and opertaions
class Op:
    def __init__(self, name, num_inputs, forward_fn, backward_fn):
        self.name = name
        self.num_inputs = num_inputs
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn

    def __call__(self, *inputs):
        # check all inputs are Var instances
        if not all(isinstance(i, Var) for i in inputs):
            raise TypeError("All inputs must be instances of Var.")
        
        # input, out and return as Var object
        input_val = [i.value for i in inputs]
        out_value = self.forward_fn(*input_val)

        return Var(out_value, op=self, parents=inputs)
    


# Var class
class Var:
    def __init__(self, value, op=None, parents=()):
        self.value = value
        self.op = op
        self.parents = parents
        self.grad = None



def get_grad_size(grad, shape):
    """
    make sure gradients match original variable's shape
    I use this because was getting random errors due to shape mismatch
    """
    # get grad and required shape
    grad = np.asarray(grad)
    req_size = tuple(shape)

    # base condition
    if grad.shape == req_size:
        return grad

    # scalar case
    if grad.size == 1:
        return np.broadcast_to(grad, req_size)

    # extra dim
    while grad.ndim > len(req_size):
        grad = grad.sum(axis=0)

    # grad has more dims but out is scalar maybe
    for i in range(len(req_size)):
        if req_size[i] == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)

    if grad.shape == req_size:
        return grad



# defining the operations needed

# addition
add = Op(
    name="add",
    num_inputs=2,
    forward_fn=lambda x, y: x + y,
    backward_fn=lambda grad_out, x, y: (
        get_grad_size(grad_out, np.shape(x)),
        get_grad_size(grad_out, np.shape(y)),
    ),
)


# substraction
sub = Op(
    name="sub",
    num_inputs=2,
    forward_fn=lambda x, y: x - y,
    backward_fn=lambda grad_out, x, y: (
        get_grad_size(grad_out, np.shape(x)),
        get_grad_size(-grad_out, np.shape(y)),
    ),
)


# multiplication elementwise
mul = Op(
    name="mul",
    num_inputs=2,
    forward_fn=lambda x, y: x * y,
    backward_fn=lambda grad_out, x, y: (
        get_grad_size(grad_out * y, np.shape(x)),
        get_grad_size(grad_out * x, np.shape(y)),
    ),
)


# matmul, defining separate backward as was getting issues for later questions
def backward_fn_matmul(grad_output, X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    grad_output = np.asarray(grad_output)

    # vector @ vector = scalar
    if X.ndim == 1 and Y.ndim == 1:
        dX = grad_output * Y
        dY = grad_output * X

    # matrix @ vector = vector
    elif X.ndim == 2 and Y.ndim == 1:
        dX = np.outer(grad_output, Y)
        dY = X.T @ grad_output

    # vector @ matrix = vector
    elif X.ndim == 1 and Y.ndim == 2:
        dX = grad_output @ Y.T
        dY = np.outer(X, grad_output)

    # matrix @ matrix = matrix
    elif X.ndim == 2 and Y.ndim == 2:
        dX = grad_output @ Y.T
        dY = X.T @ grad_output

    else:
        print("Incorrect matmul shapes !")
    dX = get_grad_size(dX, np.shape(X))
    dY = get_grad_size(dY, np.shape(Y))

    return dX, dY


matmul = Op(
    name="matmul",
    num_inputs=2,
    forward_fn=lambda X, Y: X @ Y,
    backward_fn=backward_fn_matmul,
)


# inner
inner = Op(
    name="inner",
    num_inputs=2,
    forward_fn=lambda x, y: np.sum(x * y),
    backward_fn=lambda grad_out, x, y: (
        get_grad_size(grad_out * y, np.shape(x)),
        get_grad_size(grad_out * x, np.shape(y)),
    ),
)


# sum
sum = Op(
    name="sum",
    num_inputs=1,
    forward_fn=lambda x: np.sum(x),
    backward_fn=lambda grad_out, x: (np.broadcast_to(grad_out, np.shape(x)),),
)


# solve, again here defining a backward func as ran into issues in later questions
def backward_fn_solve(grad_output, A, b):
    A = np.asarray(A)
    b = np.asarray(b)

    # forward sol, grad output and 2 dim
    x = np.linalg.solve(A, b)
    grad_output = np.asarray(grad_output)
    if grad_output.ndim == 0:
        grad_output = np.broadcast_to(grad_output, np.shape(x))
    grad_out_2d = grad_output if grad_output.ndim > 1 else grad_output[:, None]

    # grad wrt A, b
    grad_b_2d = np.linalg.solve(A.T, grad_out_2d)
    x_2d = x if x.ndim > 1 else x[:, None]
    grad_A = -grad_b_2d @ x_2d.T

    # check shape
    if np.ndim(b) == 1:
        grad_b = grad_b_2d.ravel()
    else:
        grad_b = grad_b_2d
    grad_A = get_grad_size(grad_A, np.shape(A))
    grad_b = get_grad_size(grad_b, np.shape(b))

    return grad_A, grad_b


solve = Op(
    name="solve",
    num_inputs=2,
    forward_fn=lambda A, b: np.linalg.solve(A, b),
    backward_fn=backward_fn_solve,
)


# logdet
logdet = Op(
    name="logdet",
    num_inputs=1,
    forward_fn=lambda A: np.log(np.linalg.det(A)),
    backward_fn=lambda grad_out, A: (get_grad_size(np.asarray(grad_out) * np.linalg.inv(A).T, np.shape(A)),),
)


# logsumexp, defining forward and backward separatelys
def forward_fn_logexpsum(x):
    x = np.asarray(x)
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def backward_fn_logexpsum(grad_out, x):
    x = np.asarray(x)
    exp_x = np.exp(x - np.max(x))
    softmax = exp_x / np.sum(exp_x)
    return (get_grad_size(np.asarray(grad_out) * softmax, np.shape(x)),)


logsumexp = Op("logsumexp", 1, forward_fn_logexpsum, backward_fn_logexpsum)


# exp
exp = Op(
    name="exp",
    num_inputs=1,
    forward_fn=lambda x: np.exp(x),
    backward_fn=lambda grad_out, x: (get_grad_size(np.asarray(grad_out) * np.exp(x), np.shape(x)),),
)


# log
log = Op(
    name="log",
    num_inputs=1,
    forward_fn=lambda x: np.log(x),
    backward_fn=lambda grad_out, x: (get_grad_size(np.asarray(grad_out) / x, np.shape(x)),),
)



def get_topological_order(output):
    """
    return nodes in topological order
    using from the code given in class
    """
    visited = set()
    topo_order = []

    def build_topo(node):
        if node in visited:
            return
        visited.add(node)
        for parent in node.parents:
            build_topo(parent)
        topo_order.append(node)

    build_topo(output)
    return topo_order



def backpropagation(output, input_nodes):
    """
    grad wrt each node
    """
    topo_order = get_topological_order(output)
    grad_map = {}
    grad_map[output] = np.ones_like(output.value)

    # traverse in reverse order
    for node in reversed(topo_order):
        # if no grad
        if node not in grad_map:
            continue

        # if end node
        if node.op is None:
            continue

        current_grad = np.asarray(grad_map[node])
        parent_values = [p.value for p in node.parents]
        local_grads = node.op.backward_fn(current_grad, *parent_values)

        # grad for each parent
        for parent_var, local_g in zip(node.parents, local_grads):
            local_g = np.asarray(local_g)
            aligned = get_grad_size(local_g, np.shape(parent_var.value))

            if parent_var not in grad_map:
                grad_map[parent_var] = np.zeros_like(parent_var.value)
            grad_map[parent_var] = grad_map[parent_var] + aligned

    result_grad = [grad_map.get(node, np.zeros_like(node.value)) for node in input_nodes] # in order

    return result_grad



# var constant
def constant(val):
    return Var(val, op=None, parents=())


# grad operator
def grad(fun):
    """
    takes function and resturns grad f
    """
    def wrapper(*args_values):
        # get all 
        input_vars = [arg if isinstance(arg, Var) else constant(arg) for arg in args_values]
        output_var = fun(*input_vars)
        gradients = backpropagation(output_var, input_vars)
        return gradients
    return wrapper


# func test as required
def func_Test(x, y, A):
    """
    Computes inner(solve(A, x), matmul(A, y)) and returns a Var node.
    """
    return inner(solve(A, x), matmul(A, y))



# compute_log_likelihood function for 2.5
def compute_log_likelihood(mu_var, Sigma_var):
    """
    calculates given log likelihood
    """
    # define necessary vars
    N, D = X.shape
    const_log2piD = constant(D * np.log(2.0 * np.pi))
    logdet_sigma = logdet(Sigma_var)
    inside_term = add(const_log2piD, logdet_sigma)
    term1 = mul(constant(-0.5 * N), inside_term)
    sum_quadratic = constant(0.0)

    # get 2nd term and add
    for n in range(N):
        x_n_var = constant(X[n])
        diff = sub(x_n_var, mu_var)
        sol = solve(Sigma_var, diff)
        quad = inner(diff, sol)
        sum_quadratic = add(sum_quadratic, quad)
    term2 = mul(constant(-0.5), sum_quadratic)

    return add(term1, term2)



# Test 2.1
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, -1.0, 0.5])
A = np.array([
    [2.0, 1.0, 0.0],
    [1.0, 3.0, 1.0],
    [0.0, 1.0, 2.0]
])
grad_out = 1.0

ops = [
    ("add", add, (x, y)),
    ("mul", mul, (x, y)),
    ("exp", exp, (x,)),
    ("log", log, (x,)),
    ("inner", inner, (x, y)),
    ("logsumexp", logsumexp, (x,)),
    ("solve", solve, (A, x)),
    ("logdet", logdet, (A,))
]

for name, op, inputs in ops:
    fwd = op.forward_fn(*inputs)
    bwd = op.backward_fn(grad_out, *inputs)
    print(f"{name:10s} | forward: {fwd} | backward: {bwd}")



# Test 2.2
x_val = np.array([1.0, 2.0, 3.0])
y_val = np.array([2.0, -1.0, 0.5])
A_val = np.array([[2.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 2.0]])

x = Var(x_val)
y = Var(y_val)
A = Var(A_val)

# forward
z1 = matmul(A, x)
z2 = inner(z1, y)
print("Forward outputs:")
print("z1 =", z1.value)
print("z2 =", z2.value)

# backward
grad_z2 = 1.0
grad_z1, grad_y = inner.backward_fn(grad_z2, z1.value, y.value)
grad_A, grad_x = matmul.backward_fn(grad_z1, A.value, x.value)
print("\nBackward outputs:")
print("grad A =\n", grad_A)
print("grad x =", grad_x)
print("grad z1 =", grad_z1)
print("grad y =", grad_y)



# Test 2.3
topo_order = get_topological_order(z2)
for node in topo_order:
    print(node)
    print(node.value)
    print(node.op)



# Test 2.3
x_val = np.array([1.0, 2.0, 3.0])
y_val = np.array([2.0, -1.0, 0.5])
A_val = np.array([[2.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 2.0]])
B_val = np.array([[1.0, 2.0, 3.0],
                  [0.0, 1.0, 4.0],
                  [5.0, 6.0, 0.0]])

x = Var(x_val)
y = Var(y_val)
A = Var(A_val)
B = Var(B_val)

# graph 1
z1_matmul = matmul(A, x)
z1 = inner(z1_matmul, y)

# Backpropagation
grads_z1 = backpropagation(z1, [A, x, y])
print("Graph 1: z1 = inner(matmul(A, x), y)")
print("grad A:\n", grads_z1[0])
print("grad x:\n", grads_z1[1])
print("grad y:\n", grads_z1[2])

# graph 2
solve_node = solve(A, x)
matmul_node = matmul(A, y)
logdet_node = logdet(B)
add_node = add(matmul_node, logdet_node)
z2 = inner(solve_node, add_node)

# Backpropagation
grads_z2 = backpropagation(z2, [A, B, x, y])
print("\nGraph 2: z2 = inner(solve(A,x), add(matmul(A,y), logdet(B)))")
print("grad A:\n", grads_z2[0])
print("grad B:\n", grads_z2[1])
print("grad x:\n", grads_z2[2])
print("grad y:\n", grads_z2[3])



# Test 2.4
x_val = np.array([1.0, 2.0, 3.0])
y_val = np.array([2.0, -1.0, 0.5])
A_val = np.array([[2.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 2.0]])

grad_func = grad(func_Test)
gradients = grad_func(x_val, y_val, A_val)
grad_x, grad_y, grad_A = gradients

print("Gradients of func_Test:")
print("grad x =", grad_x)
print("grad y =", grad_y)
print("grad A =\n", grad_A)



# Test 6
# load data -> Assume the gaussian_toy_dataset is in this directory !
mu = np.load("gaussian_toy_dataset/mean.npy")
sigma = np.load("gaussian_toy_dataset/cov.npy")
X = np.load("gaussian_toy_dataset/X.npy")
print("loaded shapes:", mu.shape, sigma.shape, X.shape)

ll_func = lambda mu_var, Sigma_var: compute_log_likelihood(mu_var, Sigma_var)
grad_ll = grad(ll_func)
grad_mu, grad_sigma = grad_ll(mu, sigma)
ll_var = compute_log_likelihood(constant(mu), constant(sigma))
forward_loglik_value = ll_var.value

# print
print("\nlog-likelihood:", forward_loglik_value)
print("\ngradient wrt mu:", np.asarray(grad_mu).shape)
print(grad_mu)
print("\ngradient wrt covariance:", np.asarray(grad_sigma).shape)
print(grad_sigma)

print("Q2 DONE RUNNING\n")
# ================================================================================

# ====================================== Q3 ======================================
print("Q3 RUNNING:\n")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# init parameters
key = jax.random.key(0)
wstar = jnp.array([3.0, -3.0])
mu_x = jnp.array([1.0, 2.0])
sigmax = jnp.array([[3.0, 1.0],[1.0, 2.0]])
sigma_epsilon = jnp.sqrt(3.0)

# sample data with size N
def sample_data(key, N):
    key_x, key_eps = jax.random.split(key)
    X = jax.random.multivariate_normal(key_x, mu_x, sigmax, (N,))
    epsilon = sigma_epsilon * jax.random.normal(key_eps, (N,))
    y = X @ wstar + epsilon
    return X, y

# linear regression
def fit_ols(X, y):
    return jnp.linalg.inv(X.T @ X) @ X.T @ y

# get risk
def get_risk(X, y, w):
    preds = X @ w
    return jnp.mean((preds - y)**2)


# ========== Q3.3 ============
dataset_sizes = [10, 100, 1000, 10000]
num_trials = 100
results = {}

# test set for true risk
key, subkey = jax.random.split(key)
X_test, y_test = sample_data(subkey, 10000)
R_star = get_risk(X_test, y_test, wstar)

for N in dataset_sizes:
    ws = []
    risks = []

    # for every trial run linear reg and get risk
    for i in range(num_trials):
        key, subkey = jax.random.split(key)
        X, y = sample_data(subkey, N)
        what = fit_ols(X, y)
        ws.append(what)
        risks.append(get_risk(X_test, y_test, what) - R_star)

    ws = jnp.stack(ws)
    mean_error = jnp.mean(ws - wstar, axis=0)
    cov_error = jnp.cov((ws - wstar).T)
    mean_risk_diff = jnp.mean(jnp.array(risks))
    
    results[N] = {"mean_error": mean_error, "cov_error": cov_error, "mean_risk_diff": mean_risk_diff}


# print results
for N, res in results.items():
    print(f"\nN = {N}")
    print(f"Mean(wN - w*): {res['mean_error']}")
    print(f"Cov(wN - w*):\n{res['cov_error']}")
    print(f"Mean[R(wN) - R(w*)]: {res['mean_risk_diff']}")


# ========== Q3.4 ============
para_diffs = []
risk_diffs = []

for N in dataset_sizes:
    # get error and multiply by root N
    mean_err = results[N]["mean_error"]
    para_diff = jnp.linalg.norm(jnp.sqrt(N) * mean_err)
    para_diffs.append(float(para_diff))

    # get risk and multiply by N
    risk_diff = N * results[N]["mean_risk_diff"]
    risk_diffs.append(float(risk_diff))

# plot the results
plt.figure(figsize=(12,5))

# plot √N(wN - w*)
plt.subplot(1,2,1)
plt.plot(dataset_sizes, para_diffs, 'o-', label=r'$\sqrt{N}(w_N - w^*)$')
plt.xscale('log')
plt.xlabel('N (log scale)')
plt.ylabel('Empirical mean norm')
plt.title('Empirical Mean of √N(wN − w*) vs N')
plt.legend()

# plot N(R(wN) - R(w*))
plt.subplot(1,2,2)
plt.plot(dataset_sizes, risk_diffs, 'o-', color='orange', label=r'$N(R(w_N) - R(w^*))$')
plt.xscale('log')
plt.xlabel('N (log scale)')
plt.ylabel('Empirical Mean')
plt.title('Empirical Mean of N(R(wN) − R(w*)) vs N')
plt.legend()

plt.tight_layout()
plt.show()

print("Q3 DONE RUNNING\n")
# ================================================================================

# ====================================== Q7 ======================================
print("Q7 RUNNING:\n")

# Q7 Lipschitz

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# given
K = 1.0
B = 10.0
D = 10.0
delta = 0.01
dataset_sizes = [10, 100, 1000, 10000, 100000]
epsilon = np.logspace(-5, 1, 1000)
const_C = 3.0 * B * np.sqrt(D)


def compute_bound(eps_array, N):
    """
    get the bound
    """
    log_ratio = np.log(const_C) - np.log(eps_array)
    logC = np.maximum(0.0, D * log_ratio)
    log_term = np.log(2.0 / delta) + logC
    second_term = np.sqrt((1.0 / (2.0 * N)) * log_term)
    bound = 2.0 * K * eps_array + second_term

    return bound


# opt epsilon for each N
results = []
bounds_by_N = {}

for N in dataset_sizes:
    b = compute_bound(epsilon, N)
    bounds_by_N[N] = b
    idx = np.argmin(b)
    eps_opt = epsilon[idx]
    bound_opt = b[idx]
    results.append({"N": N, "eps_opt": eps_opt, "bound_at_eps_opt": bound_opt})

df_results = pd.DataFrame(results)
print("optimal eps and bounds:\n")
print(df_results.to_string(index=False))


# 7.1: Bound vs epsilon
plt.figure(figsize=(8,6))
for N in dataset_sizes:
    plt.plot(epsilon, bounds_by_N[N], label=f"N={N}")
plt.xscale('log')
plt.xlabel('epsilon (log)')
plt.ylabel('bound')
plt.title('bound vs epsilon (each N)')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.show()


# 7.2: Optimal epsilon vs N
Ns = np.array(df_results['N'], dtype=float)
eps_opt = np.array(df_results['eps_opt'], dtype=float)

plt.figure(figsize=(7,6))
plt.plot(Ns, eps_opt, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('N (log)')
plt.ylabel('Opt epsilon (log)')
plt.title('Optimal epsilon vs N')
plt.grid(True, which='both', ls=':')
plt.show()

print("Q7 DONE RUNNING\n")
# ================================================================================