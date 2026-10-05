"""量纲计算工作单：一次请求内按顺序执行 2～5 个步骤。

每个步骤都是一次 **求值**（``formula``，语义同 ``POST /evaluate``）或一次
**线性方程反求**（``equation`` + ``solve_for``，语义同 ``POST /solve``），
给出唯一结果名与目标单位。后续步骤可在公式/方程中按结果名引用先前步骤，
不能引用尚未执行的步骤，也不能覆盖初始变量或已有结果。

步骤间传递的是 :class:`~app.evaluator.Quantity`——以 SI 基单位口径表示的
**精确分数值**与**绝对温度标记**，不是渲染后的十进制字符串：

- 数值在整个工作单内保持有理数精确性，仅在最终响应各步 ``value`` 时才做
  有限十进制/精度校验；
- 绝对温度属性逐步骤携带：求值步骤若以绝对温度变量/结果作为唯一项，
  其结果仍是绝对温度（可继续仿射换算到 degC/degF/K）；一旦参与任何运算即
  线性化（``absolute`` 变为 False）；反求结果恒为线性量；
- 量纲、仿射与精度规则与现有求值/反求完全一致，均经由
  :func:`~app.evaluator.convert_quantity` 与
  :func:`~app.solver.convert_solution` 转换。

任一步骤非法（量纲不兼容、未知变量/符号、除零、非法仿射用法、非线性方程、
零系数、引用未执行步骤、精度溢出等）都抛出 :class:`ConversionError`
（HTTP 422），**不返回任何部分结果**。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Mapping

from .errors import ConversionError
from .evaluator import Quantity, convert_quantity, evaluate, make_quantity
from .parser import ParsedExpression, parse_expression
from .solver import convert_solution, solve

if TYPE_CHECKING:
    from .custom_units import UnitScope
    from .schemas import Variable, WorksheetStep


@dataclass(frozen=True)
class StepOutput:
    """单步执行的内部产物：结果名、目标单位与目标口径下的精确分数。"""

    name: str
    dst: ParsedExpression
    value_fr: Fraction


def invalid_request(message: str) -> ConversionError:
    """与工作单编排相关的非法情况，统一归入 ``invalid_request``（HTTP 422）。"""

    return ConversionError("invalid_request", message)


def run_worksheet(
    variables: Mapping[str, Variable],
    steps: list[WorksheetStep],
    scope: UnitScope | None,
) -> list[StepOutput]:
    """按顺序执行工作单，返回各步的内部结果；任一步失败即抛出 422。

    ``scope`` 由调用方在请求开始时按 custom_units / 定义集 / 活动别名
    构造一次：整个工作单的初始变量单位与各步目标单位都用同一份定义解析，
    别名在执行期间即使被切换也不会影响本次工作单。
    """

    # 初始变量先解析单位并折叠为 SI 口径的精确量（携带绝对温度标记）；
    # 未知单位在此即报 422，发生在任何步骤执行之前
    quantities: dict[str, Quantity] = {
        name: make_quantity(var.value, parse_expression(var.unit, scope))
        for name, var in variables.items()
    }

    outputs: list[StepOutput] = []
    for index, step in enumerate(steps):
        label = f"第 {index + 1} 步"
        dst = parse_expression(step.output_unit, scope)
        if step.is_solve:
            # solve_for 不得是已有变量（schema 已先行拒绝；此处防御式兜底，
            # 保证在非 HTTP 场景直接调用时也不会覆盖已有结果）
            if step.solve_for in quantities:
                raise invalid_request(
                    f"{label}的 solve_for {step.solve_for!r} 不得是初始变量或先前结果名"
                )
            value_si, dim = solve(step.equation, step.solve_for, quantities)
            value_fr, carried = convert_solution(value_si, dim, dst)
        else:
            # 引用未执行步骤/不存在的名称时，evaluate/solve 即抛 unknown_variable
            result = evaluate(step.formula, quantities)
            value_fr, carried = convert_quantity(result, dst)
        # 结果名唯一性由 schema 校验；此处兜底，防止直接调用时覆盖已有变量
        if step.result in quantities:
            raise invalid_request(
                f"{label}结果名 {step.result!r} 已被初始变量或先前步骤占用"
            )
        # 后续步骤按 SI 口径的精确量引用（含绝对温度标记），
        # 与本步选择的目标单位无关：value_fr 仅用于该步展示
        quantities[step.result] = carried
        outputs.append(StepOutput(name=step.result, dst=dst, value_fr=value_fr))
    return outputs
