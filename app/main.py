"""FastAPI 应用入口。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .converter import convert, to_decimal_str
from .custom_units import scope_for_request
from .errors import AliasConflictError, ConversionError
from .evaluator import convert_quantity, evaluate, make_quantity
from .parser import canonical, canonical_dim, parse_expression
from .schemas import (
    BindUnitSetAliasRequest,
    ConvertRequest,
    ConvertResponse,
    ErrorResponse,
    EvaluateRequest,
    PublishUnitSetRequest,
    PublishedUnitSetResponse,
    SolveRequest,
    UnitSetAliasInfo,
    UnitSetAliasListResponse,
    WorksheetRequest,
    WorksheetResponse,
    WorksheetStepResult,
)
from .solver import convert_solution, solve
from .unit_sets import get_store
from .units import ALIASES, PREFIXES, REGISTRY
from .worksheet import run_worksheet


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 启动时加载磁盘上的已发布自定义单位定义集（容器重建后仍可使用）
    get_store()
    yield


app = FastAPI(
    title="量纲表达式与仿射单位换算 API",
    version=__version__,
    description=(
        "解析单位表达式（乘、除、整数幂），规范化量纲，支持 SI 词头与"
        "摄氏/华氏/开尔文仿射温标；/evaluate 支持带单位的变量公式求值，"
        "/solve 支持用带单位的等式反求一个未知量（未知量须处于一次线性位置），"
        "/worksheet 支持一次请求内按顺序执行 2～5 个求值/反求步骤并引用先前结果，"
        "所有结果均为精确十进制，维度不兼容、未知符号、非法幂、除零、"
        "非法仿射用法、非线性方程、零系数或精度溢出时返回 422。"
        "可通过 POST /custom-unit-sets 把自定义单位发布为可复用定义集，"
        "再由各计算接口凭标识或活动别名选用。"
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 异常处理：领域错误与参数校验错误统一返回 422
# ---------------------------------------------------------------------------


@app.exception_handler(ConversionError)
async def conversion_error_handler(_: Request, exc: ConversionError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": {"error": exc.code, "message": exc.message}},
    )


@app.exception_handler(AliasConflictError)
async def alias_conflict_error_handler(_: Request, exc: AliasConflictError) -> JSONResponse:
    # 乐观切换冲突：当前指向与 expected_id 不一致；不改变任何指向
    detail: dict = {"error": "alias_conflict", "message": exc.message}
    if exc.current_id is not None:
        detail["current_id"] = exc.current_id
    return JSONResponse(status_code=409, content={"detail": detail})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": "invalid_request",
                "message": "请求参数校验失败",
                "errors": errors,
            }
        },
    )


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


@app.post(
    "/custom-unit-sets",
    response_model=PublishedUnitSetResponse,
    status_code=201,
    responses={422: {"model": ErrorResponse, "description": "任一条目非法，未发布"}},
    summary="发布可复用的自定义单位定义集",
)
def publish_unit_set_endpoint(req: PublishUnitSetRequest) -> PublishedUnitSetResponse:
    # 先按现有规则校验并展开全部条目：任一项非法即抛 422，且不写入任何定义集
    set_id = get_store().publish(req.custom_units)
    return PublishedUnitSetResponse(
        id=set_id,
        custom_units=[
            {"symbol": item.symbol, "factor": str(item.factor), "base_unit": item.base_unit}
            for item in req.custom_units
        ],
    )


@app.get(
    "/custom-unit-set-aliases",
    response_model=UnitSetAliasListResponse,
    summary="查询全部活动别名及其当前指向的定义集",
)
def list_unit_set_aliases_endpoint() -> UnitSetAliasListResponse:
    aliases = get_store().list_aliases()
    return UnitSetAliasListResponse(
        aliases=[
            UnitSetAliasInfo(alias=alias, custom_unit_set_id=target)
            for alias, target in aliases.items()
        ]
    )


@app.get(
    "/custom-unit-set-aliases/{alias}",
    response_model=UnitSetAliasInfo,
    responses={422: {"model": ErrorResponse, "description": "别名不存在"}},
    summary="查询单个活动别名当前指向的定义集",
)
def get_unit_set_alias_endpoint(alias: str) -> UnitSetAliasInfo:
    return UnitSetAliasInfo(
        alias=alias, custom_unit_set_id=get_store().get_alias(alias)
    )


@app.put(
    "/custom-unit-set-aliases/{alias}",
    response_model=UnitSetAliasInfo,
    responses={
        409: {"description": "当前指向与 expected_id 不一致，未切换"},
        422: {"model": ErrorResponse, "description": "目标定义集不存在或别名非法，未切换"},
    },
    summary="创建/切换活动别名（乐观并发：expected_id）",
)
def bind_unit_set_alias_endpoint(
    alias: str, req: BindUnitSetAliasRequest
) -> UnitSetAliasInfo:
    # 仅当当前指向与 expected_id 一致时生效（初建须为 null）；
    # 目标不存在返回 422；切换与落盘共同生效，失败不改写原指向。
    target = get_store().bind_alias(
        alias, req.custom_unit_set_id, req.expected_id
    )
    return UnitSetAliasInfo(alias=alias, custom_unit_set_id=target)


@app.post(
    "/convert",
    response_model=ConvertResponse,
    responses={422: {"model": ErrorResponse, "description": "换算失败"}},
    summary="单位换算",
)
def convert_endpoint(req: ConvertRequest) -> ConvertResponse:
    scope = scope_for_request(
        req.custom_units, req.custom_unit_set_id, req.custom_unit_set_alias
    )
    src = parse_expression(req.from_unit, scope)
    dst = parse_expression(req.to_unit, scope)
    value = convert(req.value, src, dst)
    return ConvertResponse(
        value=value,
        unit=canonical(dst.terms),
        dimension=canonical_dim(dst.dim),
    )


@app.post(
    "/evaluate",
    response_model=ConvertResponse,
    responses={422: {"model": ErrorResponse, "description": "求值/换算失败"}},
    summary="带单位的变量公式求值并换算到目标单位",
)
def evaluate_endpoint(req: EvaluateRequest) -> ConvertResponse:
    scope = scope_for_request(
        req.custom_units, req.custom_unit_set_id, req.custom_unit_set_alias
    )
    # 变量先解析单位并折叠为 SI 口径的精确量；未知单位在此即报 422
    quantities = {
        name: make_quantity(var.value, parse_expression(var.unit, scope))
        for name, var in req.variables.items()
    }
    result = evaluate(req.formula, quantities)
    dst = parse_expression(req.output_unit, scope)
    value_fr, _ = convert_quantity(result, dst)
    return ConvertResponse(
        value=to_decimal_str(value_fr),
        unit=canonical(dst.terms),
        dimension=canonical_dim(dst.dim),
    )


@app.post(
    "/solve",
    response_model=ConvertResponse,
    responses={422: {"model": ErrorResponse, "description": "求解失败"}},
    summary="用带单位的等式反求一个未知量",
)
def solve_endpoint(req: SolveRequest) -> ConvertResponse:
    scope = scope_for_request(
        req.custom_units, req.custom_unit_set_id, req.custom_unit_set_alias
    )
    # 变量先解析单位并折叠为 SI 口径的精确量；未知单位在此即报 422
    quantities = {
        name: make_quantity(var.value, parse_expression(var.unit, scope))
        for name, var in req.variables.items()
    }
    value_si, dim = solve(req.equation, req.solve_for, quantities)
    dst = parse_expression(req.output_unit, scope)
    value_fr, _ = convert_solution(value_si, dim, dst)
    return ConvertResponse(
        value=to_decimal_str(value_fr),
        unit=canonical(dst.terms),
        dimension=canonical_dim(dst.dim),
    )


@app.post(
    "/worksheet",
    response_model=WorksheetResponse,
    responses={422: {"model": ErrorResponse, "description": "任一步骤失败，整单无结果"}},
    summary="按顺序执行的量纲计算工作单（2～5 步，可引用先前步骤结果）",
)
def worksheet_endpoint(req: WorksheetRequest) -> WorksheetResponse:
    # 别名/定义集在此解析一次并构造同一 scope：整个工作单（初始变量、各步单位）
    # 都使用该版定义，执行期间别名被切换也不会跨版本混用
    scope = scope_for_request(
        req.custom_units, req.custom_unit_set_id, req.custom_unit_set_alias
    )
    outputs = run_worksheet(req.variables, req.steps, scope)
    # 任一步失败都会在 run_worksheet 内抛出 422，不会返回部分结果
    return WorksheetResponse(
        steps=[
            WorksheetStepResult(
                result=out.name,
                value=to_decimal_str(out.value_fr),
                unit=canonical(out.dst.terms),
                dimension=canonical_dim(out.dst.dim),
            )
            for out in outputs
        ]
    )


@app.get("/health", summary="健康检查")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/units", summary="列出全部支持的单位、词头与别名")
def list_units() -> dict:
    return {
        "units": [
            {
                "symbol": u.symbol,
                "name": u.name,
                "dimension": canonical_dim(u.dim),
                "affine": u.affine,
                "prefixable": u.prefixable,
            }
            for u in REGISTRY.values()
        ],
        "prefixes": sorted(PREFIXES),
        "aliases": ALIASES,
    }


@app.get("/", summary="服务信息")
def root() -> dict[str, str]:
    return {
        "service": "量纲表达式与仿射单位换算 API",
        "version": __version__,
        "docs": "/docs",
    }
