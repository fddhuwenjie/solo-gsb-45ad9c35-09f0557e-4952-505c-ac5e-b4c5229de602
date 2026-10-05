"""POST /worksheet（量纲计算工作单）的端到端测试。

覆盖：

- 2～5 个求值/反求步骤按顺序执行，后续步骤按名引用先前结果；
- 不能引用未执行步骤，不能覆盖初始变量或已有结果（含 solve_for）；
- 步骤间传递 SI 口径精确数值与绝对温度标记；
- 任一步骤错误整单返回 422，不返回部分结果；
- 自定义单位三种来源（请求内 / 已发布定义集 / 活动别名）及互斥规则；
- 别名整个工作单只解析一次，执行期间切换不跨版本混用。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def var(value, unit):
    return {"value": value, "unit": unit}


def ev_step(result, formula, output_unit):
    return {"result": result, "formula": formula, "output_unit": output_unit}


def solve_step(result, equation, solve_for, output_unit):
    return {
        "result": result,
        "equation": equation,
        "solve_for": solve_for,
        "output_unit": output_unit,
    }


def ws(steps, variables=None, **extra):
    body = {"steps": steps}
    if variables is not None:
        body["variables"] = variables
    body.update(extra)
    return client.post("/worksheet", json=body)


def assert_steps(resp, expected):
    assert resp.status_code == 200, resp.text
    actual = [
        (s["result"], s["value"], s["unit"], s["dimension"])
        for s in resp.json()["steps"]
    ]
    assert actual == expected


def assert_err(resp, code):
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["error"] == code


# ---------------------------------------------------------------------------
# 基本顺序执行与跨步骤引用
# ---------------------------------------------------------------------------


def test_mixed_evaluate_and_solve_steps():
    resp = ws(
        [
            ev_step("s", "v*t", "km"),
            solve_step("t2", "v2*t2 = s", "t2", "h"),
            ev_step("avg", "(s+s)/(t+t2)", "km/h"),
        ],
        variables={
            "v": var("60", "km/h"),
            "t": var("2", "h"),
            "v2": var("30", "km/h"),
        },
    )
    assert_steps(
        resp,
        [
            ("s", "120", "km", "m"),
            ("t2", "4", "h", "s"),
            ("avg", "40", "km/h", "m/s"),
        ],
    )


def test_step_references_most_recent_result():
    resp = ws(
        [
            ev_step("a", "x", "m"),
            ev_step("b", "a + half", "m"),
        ],
        variables={"x": var("1", "m"), "half": var("50", "cm")},
    )
    assert_steps(
        resp,
        [
            ("a", "1", "m", "m"),
            ("b", "1.5", "m", "m"),
        ],
    )


def test_works_without_initial_variables():
    resp = ws(
        [
            ev_step("x", "2+3", "1"),
            solve_step("y", "x*y = 20", "y", "1"),
        ]
    )
    assert_steps(resp, [("x", "5", "1", "1"), ("y", "4", "1", "1")])


def test_solve_result_reused_by_later_evaluate():
    # 先反求加速度，再在公式中引用它求力
    resp = ws(
        [
            solve_step("acc", "F_ = m_*acc", "acc", "m/s^2"),
            ev_step("F2", "m_*acc", "N"),
        ],
        variables={"m_": var("2", "kg"), "F_": var("6", "N")},
    )
    assert_steps(
        resp,
        [
            ("acc", "3", "m/s^2", "m/s^2"),
            ("F2", "6", "N", "kg*m/s^2"),
        ],
    )


def test_five_steps_allowed():
    resp = ws([ev_step(f"s{i}", "1", "1") for i in range(5)])
    assert resp.status_code == 200, resp.text
    assert [s["result"] for s in resp.json()["steps"]] == [f"s{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# 步骤间精确数值与绝对温度属性传递
# ---------------------------------------------------------------------------


def test_exact_values_carry_between_steps():
    # 10 ft = 3.048 m；后续相加仍是精确值 6.096，不经过任何舍入
    resp = ws(
        [
            ev_step("a", "n", "m"),
            ev_step("b", "a + a", "m"),
        ],
        variables={"n": var("10", "ft")},
        custom_units=[{"symbol": "ft", "factor": "0.3048", "base_unit": "m"}],
    )
    assert_steps(
        resp,
        [("a", "3.048", "m", "m"), ("b", "6.096", "m", "m")],
    )


def test_absolute_temperature_attribute_carries():
    # 98.6 degF -> 310.15 K -> 37 degC：绝对温度标记逐步骤保留
    resp = ws(
        [
            ev_step("Tk", "T", "K"),
            ev_step("Tc", "Tk", "degC"),
        ],
        variables={"T": var("98.6", "degF")},
    )
    assert_steps(
        resp,
        [
            ("Tk", "310.15", "K", "K"),
            ("Tc", "37", "degC", "K"),
        ],
    )


def test_absolute_temperature_into_arithmetic_rejected():
    resp = ws(
        [
            ev_step("x", "T", "K"),
            ev_step("y", "x + a", "K"),
        ],
        variables={"T": var("0", "degC"), "a": var("1", "K")},
    )
    assert_err(resp, "affine_mixing")


def test_solve_result_is_linear_even_for_temperature_dimension():
    # 反求温差结果可以继续参与线性运算
    resp = ws(
        [
            solve_step("x", "x + a = b", "x", "dC"),
            ev_step("y", "x + a", "dC"),
        ],
        variables={"a": var("5", "dC"), "b": var("12", "dC")},
    )
    assert_steps(
        resp,
        [("x", "7", "dC", "K"), ("y", "12", "dC", "K")],
    )


def test_undetermined_zero_solution_takes_output_dimension():
    # 2*x = x 的解恒为零且量纲未定：按第一步目标单位（m）存储，后续可当长度用
    resp = ws(
        [
            solve_step("x", "2*x = x", "x", "m"),
            ev_step("y", "x + one_m", "cm"),
        ],
        variables={"one_m": var("1", "m")},
    )
    assert_steps(resp, [("x", "0", "m", "m"), ("y", "100", "cm", "m")])


# ---------------------------------------------------------------------------
# 引用与覆盖规则
# ---------------------------------------------------------------------------


def test_reference_to_not_yet_executed_step_rejected():
    resp = ws(
        [
            ev_step("x", "y", "m"),
            ev_step("y", "a", "m"),
        ],
        variables={"a": var("1", "m")},
    )
    assert_err(resp, "unknown_variable")


def test_step_result_cannot_overwrite_initial_variable():
    resp = ws(
        [
            ev_step("a", "a", "m"),
            ev_step("b", "a", "m"),
        ],
        variables={"a": var("1", "m")},
    )
    assert_err(resp, "invalid_request")


def test_duplicate_result_names_rejected():
    resp = ws(
        [
            ev_step("x", "a", "m"),
            ev_step("x", "a", "m"),
        ],
        variables={"a": var("1", "m")},
    )
    assert_err(resp, "invalid_request")


def test_solve_for_cannot_name_existing_variable():
    resp = ws(
        [
            solve_step("x", "a*x = 2", "a", "1"),
            ev_step("y", "x", "1"),
        ],
        variables={"a": var("1", "1")},
    )
    assert_err(resp, "invalid_request")


def test_solve_for_may_reference_later_step_name():
    # solve_for 用第二步才产生的名字是允许的（它只是方程内的未知量符号）
    resp = ws(
        [
            solve_step("x", "2*y = 8", "y", "1"),
            ev_step("y", "x + 1", "1"),
        ]
    )
    assert_steps(resp, [("x", "4", "1", "1"), ("y", "5", "1", "1")])


# ---------------------------------------------------------------------------
# 各步骤沿用现有求值/反求规则
# ---------------------------------------------------------------------------


def test_nonlinear_equation_step_rejected():
    resp = ws(
        [
            ev_step("a", "1", "1"),
            solve_step("x", "x*x = 4", "x", "1"),
        ]
    )
    assert_err(resp, "nonlinear_equation")


def test_no_unique_solution_step_rejected():
    resp = ws(
        [ev_step("a", "1", "1"), solve_step("x", "x = x", "x", "1")]
    )
    assert_err(resp, "no_unique_solution")


def test_solve_output_affine_unit_rejected():
    resp = ws(
        [
            solve_step("x", "x = a", "x", "degC"),
            ev_step("y", "x", "dC"),
        ],
        variables={"a": var("10", "dC")},
    )
    assert_err(resp, "affine_mixing")


def test_dimension_mismatch_fails_whole_worksheet():
    # 第一步合法，第二步量纲不兼容 -> 422（且响应不含部分结果）
    resp = ws(
        [
            ev_step("x", "a", "m"),
            ev_step("y", "x + a", "s"),
        ],
        variables={"a": var("1", "m")},
    )
    assert_err(resp, "dimension_mismatch")
    assert "steps" not in resp.json()


def test_division_by_zero_step_rejected():
    resp = ws(
        [
            ev_step("a", "1", "1"),
            ev_step("b", "a/0", "1"),
        ]
    )
    assert_err(resp, "division_by_zero")


def test_precision_overflow_any_step_fails_all():
    # 第一步本身无法有限十进制表示 -> 整单 422
    resp = ws(
        [
            ev_step("tm", "t", "min"),
            ev_step("k", "tm/tm", "1"),
        ],
        variables={"t": var("1", "s")},
    )
    assert_err(resp, "precision_overflow")

    # 第一步合法、第二步精度溢出 -> 同样整单 422
    resp = ws(
        [
            ev_step("tm", "t", "min"),
            ev_step("later", "d", "min"),
        ],
        variables={"t": var("30", "s"), "d": var("1", "s")},
    )
    assert_err(resp, "precision_overflow")


def test_unknown_symbol_fails_whole_worksheet():
    resp = ws(
        [
            ev_step("x", "a", "m"),
            ev_step("y", "x", "m"),
        ],
        variables={"a": var("1", "furlong")},
    )
    assert_err(resp, "unknown_symbol")


# ---------------------------------------------------------------------------
# 请求体校验
# ---------------------------------------------------------------------------


def test_step_count_bounds():
    assert_err(ws([ev_step("x", "1", "1")]), "invalid_request")
    assert_err(
        ws([ev_step(f"s{i}", "1", "1") for i in range(6)]),
        "invalid_request",
    )


def test_step_must_be_exactly_one_kind():
    # 同时给 formula 与 equation/solve_for
    resp = ws(
        [
            {
                "result": "x",
                "formula": "1",
                "equation": "x = 1",
                "solve_for": "x",
                "output_unit": "1",
            },
            ev_step("y", "x", "1"),
        ]
    )
    assert_err(resp, "invalid_request")
    # 两者都不给
    resp = ws(
        [
            {"result": "x", "output_unit": "1"},
            ev_step("y", "1", "1"),
        ]
    )
    assert_err(resp, "invalid_request")
    # 只给 equation 不给 solve_for
    resp = ws(
        [
            {"result": "x", "equation": "x = 1", "output_unit": "1"},
            ev_step("y", "1", "1"),
        ]
    )
    assert_err(resp, "invalid_request")


def test_equation_must_have_single_equals():
    resp = ws(
        [
            solve_step("x", "x + 1", "x", "1"),
            ev_step("y", "x", "1"),
        ]
    )
    assert_err(resp, "invalid_request")


def test_invalid_names_rejected():
    # 非法结果名
    resp = ws(
        [
            ev_step("a-b", "1", "1"),
            ev_step("z", "1", "1"),
        ]
    )
    assert_err(resp, "invalid_request")
    # 非法初始变量名
    resp = ws(
        [ev_step("x", "v", "1"), ev_step("y", "x", "1")],
        variables={"v-1": var("1", "1")},
    )
    assert_err(resp, "invalid_request")


def test_missing_fields_and_bad_value_are_422():
    # 缺 steps
    assert client.post("/worksheet", json={}).status_code == 422
    # value 非法（布尔）
    resp = ws(
        [ev_step("x", "a", "m"), ev_step("y", "x", "m")],
        variables={"a": var(True, "m")},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 自定义单位：请求内 / 定义集 / 活动别名
# ---------------------------------------------------------------------------


def test_request_local_custom_units():
    resp = ws(
        [
            ev_step("area", "n*w", "dozen^2"),
            solve_step("ratio", "ratio*n = w", "ratio", "1"),
        ],
        variables={"n": var("2", "dozen"), "w": var("3", "dozen")},
        custom_units=[{"symbol": "dozen", "factor": "12", "base_unit": "1"}],
    )
    assert_steps(
        resp,
        [
            ("area", "6", "dozen^2", "1"),
            ("ratio", "1.5", "1", "1"),
        ],
    )


def _publish(defs):
    r = client.post("/custom-unit-sets", json={"custom_units": defs})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _cu(symbol, factor, base_unit):
    return {"symbol": symbol, "factor": factor, "base_unit": base_unit}


def test_published_set_and_alias_selection():
    sid_v1 = _publish([_cu("foo", "2", "m")])
    sid_v2 = _publish([_cu("foo", "3", "m")])
    body = {
        "variables": {"a": var("1", "foo")},
        "steps": [
            ev_step("x", "a", "m"),
            ev_step("y", "x", "foo"),
        ],
    }

    r = client.post("/worksheet", json={**body, "custom_unit_set_id": sid_v1})
    assert_steps(r, [("x", "2", "m", "m"), ("y", "1", "foo", "m")])
    r = client.post("/worksheet", json={**body, "custom_unit_set_id": sid_v2})
    assert_steps(r, [("x", "3", "m", "m"), ("y", "1", "foo", "m")])

    # 别名：解析一次并贯穿整个工作单；切换后是另一次请求才看到新版
    r = client.put(
        "/custom-unit-set-aliases/live",
        json={"custom_unit_set_id": sid_v1, "expected_id": None},
    )
    assert r.status_code == 200, r.text
    r = client.post("/worksheet", json={**body, "custom_unit_set_alias": "live"})
    assert_steps(r, [("x", "2", "m", "m"), ("y", "1", "foo", "m")])
    r = client.put(
        "/custom-unit-set-aliases/live",
        json={"custom_unit_set_id": sid_v2, "expected_id": sid_v1},
    )
    assert r.status_code == 200, r.text
    r = client.post("/worksheet", json={**body, "custom_unit_set_alias": "live"})
    assert_steps(r, [("x", "3", "m", "m"), ("y", "1", "foo", "m")])


def test_alias_is_resolved_once_per_worksheet():
    from app import unit_sets

    sid_v2 = _publish([_cu("foo", "3", "m")])
    store = unit_sets.get_store()
    body = {
        "variables": {"a": var("1", "foo")},
        "steps": [
            ev_step("x", "a", "m"),
            ev_step("y", "x", "foo"),
        ],
        "custom_unit_set_alias": "once",
    }
    client.put(
        "/custom-unit-set-aliases/once",
        json={"custom_unit_set_id": sid_v2, "expected_id": None},
    )

    calls = []
    original = store.get_alias
    store.get_alias = lambda alias: calls.append(alias) or original(alias)
    try:
        r = client.post("/worksheet", json=body)
    finally:
        store.get_alias = original
    assert r.status_code == 200, r.text
    assert calls == ["once"]
    assert [s["value"] for s in r.json()["steps"]] == ["3", "1"]


def test_unknown_alias_and_nonexistent_set_are_422():
    body = {
        "variables": {"a": var("1", "m")},
        "steps": [ev_step("x", "a", "m"), ev_step("y", "x", "m")],
    }
    assert_err(
        client.post("/worksheet", json={**body, "custom_unit_set_alias": "ghost"}),
        "invalid_request",
    )
    assert_err(
        client.post(
            "/worksheet", json={**body, "custom_unit_set_id": "cus_does_not_exist"}
        ),
        "invalid_request",
    )


def test_custom_unit_sources_are_mutually_exclusive():
    sid = _publish([_cu("foo", "2", "m")])
    client.put(
        "/custom-unit-set-aliases/live",
        json={"custom_unit_set_id": sid, "expected_id": None},
    )
    body = {
        "variables": {"a": var("1", "foo")},
        "steps": [ev_step("x", "a", "m"), ev_step("y", "x", "foo")],
    }
    assert_err(
        client.post(
            "/worksheet",
            json={**body, "custom_unit_set_alias": "live",
                  "custom_units": [_cu("foo", "1", "m")]},
        ),
        "invalid_request",
    )
    assert_err(
        client.post(
            "/worksheet",
            json={**body, "custom_unit_set_alias": "live", "custom_unit_set_id": sid},
        ),
        "invalid_request",
    )
