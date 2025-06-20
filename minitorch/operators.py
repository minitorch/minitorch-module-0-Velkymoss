"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(a: float, b: float) -> float:
    """Multiplies two floating-point numbers.

    Args:
    ----
        a (float): The first number to multiply
        b (float): The second number to multiply

    Returns:
    -------
        float: The product of a and b

    """
    """Multiplies two numbers"""
    return a * b


def id(input: float) -> float:
    """Returns the input value unchanged.

    Parameters
    ----------
    input : float
        The input value to be returned.

    Returns
    -------
    float
        The same value as the input.

    """
    """Returns the input value unchanged."""
    return input


def add(a: float, b: float) -> float:
    """Add two numbers and return the result.

    Args:
    ----
        a (float): First number to add.
        b (float): Second number to add.

    Returns:
    -------
        float: The sum of a and b.

    Examples:
    --------
        >>> add(3.0, 4.0)
        7.0
        >>> add(-2.0, 2.0)
        0.0

    """
    """Adds two numbers."""
    return a + b


def neg(a: float) -> float:
    """Negate a number.

    Args:
    ----
        a (float): The number to negate.

    Returns:
    -------
        float: The negation of the input number, which is -a.

    Examples:
    --------
        >>> neg(5)
        -5
        >>> neg(-2)
        2

    """
    """Negates a number."""
    return -a


def lt(a: float, b: float) -> bool:
    """Compares two floating-point numbers and returns True if they are not equal.

    Args:
    ----
        a (float): The first number to compare.
        b (float): The second number to compare.

    Returns:
    -------
        bool: True if a is greater than b or a is less than b (i.e., a != b), otherwise False.

    """
    if a < b:
        return True
    else:
        return False


def eq(a: float, b: float) -> bool:
    """Compares two floating-point numbers for equality.

    Args:
    ----
        a (float): The first number to compare.
        b (float): The second number to compare.

    Returns:
    -------
        bool: True if both numbers are equal, False otherwise.

    """
    if a == b:
        return True
    else:
        return False


def max(a: float, b: float) -> float:
    """Returns the maximum of two float values.

    Args:
    ----
        a (float): The first value to compare.
        b (float): The second value to compare.

    Returns:
    -------
        float: The greater of the two input values. If both are equal, returns `a`.

    """
    if a > b:
        return a
    elif b > a:
        return b
    else:
        return a


def is_close(a: float, b: float) -> bool:
    """Test if two floating point numbers are close.

    Args:
    ----
        a: First floating point number
        b: Second floating point number

    Returns:
    -------
        True if the distance between a and b is less than 1e-2, False otherwise.

    """
    if abs(a - b) < 1e-2:
        return True
    else:
        return False


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


def relu(x: float) -> float:
    """Applies the rectified linear unit (ReLU) activation function.

    Args:
    ----
        x (float): Input value.

    Returns:
    -------
        float: The input value if it is greater than 0, otherwise 0.

    """
    return max(0, x)


def log(x: float) -> float:
    """Computes the natural logarithm of a given number.

    Args:
    ----
        x (float): The input value. Must be greater than 0.

    Returns:
    -------
        float: The natural logarithm of x.

    Raises:
    ------
        ValueError: If x is less than or equal to 0.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Implementation of the exponential function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The exponential of the input, e^x.

    Examples:
    --------
        >>> exp(0.0)
        1.0
        >>> exp(1.0)
        2.718281828459045

    """
    return math.exp(x)


def log_back(x: float, grad: float) -> float:
    """Computes the gradient of the natural logarithm function with respect to its input.

    Args:
    ----
        x (float): The input value to the logarithm function.
        grad (float): The gradient of the subsequent operation (upstream gradient).

    Returns:
    -------
        float: The gradient of the logarithm function with respect to x, multiplied by the upstream gradient.

    """
    return (1 / x) * grad


def inv(x: float) -> float:
    """Returns the multiplicative inverse of a given number.

    Args:
    ----
        x (float): The number to invert.

    Returns:
    -------
        float: The multiplicative inverse of x (i.e., 1/x).

    Raises:
    ------
        ZeroDivisionError: If x is zero.

    """
    if x == 0:
        raise (ZeroDivisionError)
    else:
        return 1 / x


def inv_back(x: float, grad: float) -> float:
    """Computes the gradient of the inverse function with respect to its input.

    Given the output gradient `grad` and the input value `x`, this function returns
    the gradient of the inverse operation (1/x) with respect to `x`, which is -1/x^2,
    multiplied by the upstream gradient.

    Args:
    ----
        x (float): The input value to the inverse function.
        grad (float): The upstream gradient from the next layer.

    Returns:
    -------
        float: The computed gradient with respect to `x`.

    """
    return (-1 / x**2) * grad


def relu_back(x: float, grad: float) -> float:
    """Computes the gradient of the ReLU activation function with respect to its input.

    Args:
    ----
        x (float): The input value to the ReLU function.
        grad (float): The gradient of the loss with respect to the output of the ReLU.

    Returns:
    -------
        float: The gradient of the loss with respect to the input of the ReLU.
               Returns `grad` if `x > 0`, otherwise returns 0.

    """
    if x > 0:
        return grad
    else:
        return 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn, iterable):
    pass


def zipWith(fn, list1, list2):
    pass


def reduce(fn, iterable, initial=None):
    pass


def addLists(list1, list2):
    pass


def negList(lst):
    pass


def sum(lst):
    pass


def prod(lst):
    pass
