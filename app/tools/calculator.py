import ast
import math
import operator as op

from pydantic import BaseModel, Field

from app.utils.errors import ValidationException


class CalculatorInput(BaseModel):
    expression: str = Field(min_length=1, max_length=200, description="Arithmetic expression, e.g. '25 * 48' or 'sqrt(2) ** 3'")


_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow}
_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp, "sin": math.sin,
          "cos": math.cos, "tan": math.tan, "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil}
_CONSTS = {"pi": math.pi, "e": math.e}
_MAX_BITS = 100_000  # blocks 9**9**9-style CPU/memory bombs


def _eval(node: ast.AST) -> int | float:
    match node:
        case ast.Constant(value=v) if type(v) in (int, float):
            return v
        case ast.Name(id=name) if name in _CONSTS:
            return _CONSTS[name]
        case ast.UnaryOp(op=o, operand=x) if type(o) in _UNARY:
            return _UNARY[type(o)](_eval(x))
        case ast.BinOp(left=l, op=o, right=r) if type(o) in _BIN:
            a, b = _eval(l), _eval(r)
            if isinstance(o, ast.Pow) and (abs(b) > 10_000 or (
                    isinstance(a, int) and isinstance(b, int) and abs(a).bit_length() * abs(b) > _MAX_BITS)):
                raise ValidationException("Result too large")
            result = _BIN[type(o)](a, b)
            if isinstance(result, complex):
                raise ValidationException("Complex results are not supported")
            return result
        case ast.Call(func=ast.Name(id=f), args=args, keywords=[]) if f in _FUNCS:
            return _FUNCS[f](*(_eval(a) for a in args))
    raise ValidationException(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> int | float:
    """Deterministic arithmetic. Parses to an AST and walks a whitelist; never calls eval()."""
    try:
        return _eval(ast.parse(expression, mode="eval").body)
    except ValidationException:
        raise
    except (SyntaxError, ZeroDivisionError, OverflowError, ValueError, TypeError) as e:
        raise ValidationException(f"Cannot evaluate expression: {type(e).__name__}") from e


async def calculator(expression: str) -> tuple[str, list]:
    return f"{expression} = {calculate(expression)}", []
