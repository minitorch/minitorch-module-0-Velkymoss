from math import exp


def sigmoid(x: float) -> float:
    """Implementation of the sigmoid function.

    The sigmoid function is defined as:
        sigmoid(x) = 1 / (1 + e^(-x))

    For numerical stability, this implementation uses two equivalent formulations
    based on the sign of x:
    - For x >= 0: 1 / (1 + e^(-x))
    - For x < 0: e^x / (1 + e^x)

    Args:
    ----
        x: A floating point value

    Returns:
    -------
        The sigmoid of x

    """
    if x >= 0:
        return 1 / (1 + exp(-x))
    else:
        return exp(x) / (1 + exp(x))


delta = 1e-2
threshold = 1e-15

x = 0
while True:
    diff = sigmoid(x + delta) - sigmoid(x)
    if diff < threshold:
        break
    x += 1

print(f"Upper bound for monotonicity test: {x}")
