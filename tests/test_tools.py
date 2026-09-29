import pytest

from app.tools import calculate
from app.utils.errors import ValidationException


@pytest.mark.parametrize("expr, expected", [
    ("25 * 48", 1200), ("2 ** 10", 1024), ("-3 + 4 * (2 - 1)", 1), ("7 // 2", 3), ("sqrt(16)", 4.0),
    ("round(pi, 2)", 3.14),
])
def test_calculator(expr, expected):
    assert calculate(expr) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('x')", "open('f')", "9 ** 9 ** 9", "1 / 0", "a + 1", "(-8) ** 0.5", "2 +", "True + 1",
])
def test_calculator_rejects(expr):
    with pytest.raises(ValidationException):
        calculate(expr)
