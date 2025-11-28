"""
Unit Testing of the application's logic 

"""
from mylib.calculator import add, subtract, multiply, divide, power

def test_add():
    assert add(1, 2) == 3


def test_subtract():
    assert subtract(5, 3) == 2


def test_multiply():
    assert multiply(2, 3) == 6


def test_divide():
    assert divide(6, 3) == 2


def test_power():
    assert power(2, 3) == 8