"""Griffe extension: link type parameters in annotations to their bound.

By default, mkdocstrings renders a type parameter used in an annotation (e.g.
`TArray` in `Float32[TArray, "..."]`) as a link to a type parameter heading,
which we don't render. This instead links it to the definition of its bound
(e.g. [`ArrayLike`][xwr.rsp.ArrayLike]) while still displaying its own name.

Only bounds which are a single name are linked; type parameters with union
bounds or constraints are left as is.
"""

import dataclasses
from collections.abc import Iterator
from typing import Any

import griffe


@dataclasses.dataclass(eq=False, slots=True)
class _BoundName(griffe.ExprName):
    """A type parameter name which resolves to its bound."""

    bound: griffe.ExprName | None = None

    @property
    def classname(self) -> str:
        return "ExprName"

    @property
    def canonical_path(self) -> str:
        return self.bound.canonical_path  # type: ignore[union-attr]

    @property
    def reference_path(self) -> str | None:
        return self.bound.reference_path  # type: ignore[union-attr]

    @property
    def canonical_name(self) -> str:
        return self.name

    @property
    def is_type_parameter(self) -> bool:
        return False


def _bound(name: griffe.ExprName) -> griffe.ExprName | None:
    """Get the bound of a type parameter, if it is a single name."""
    if not isinstance(name.parent, griffe.Object) or not name.is_type_parameter:
        return None
    owner_path = name.canonical_path.partition("[")[0]
    try:
        owner = name.parent.modules_collection[owner_path]
        bound = owner.type_parameters[name.name].bound
    except (KeyError, ValueError):
        return None
    return bound if isinstance(bound, griffe.ExprName) else None


def _rewrite(expr: Any) -> Any:
    """Replace type parameter names in an expression, in place."""
    if isinstance(expr, griffe.ExprName):
        bound = _bound(expr)
        if bound is None:
            return expr
        return _BoundName(expr.name, expr.parent, expr.member, bound=bound)
    if isinstance(expr, griffe.Expr):
        for field in dataclasses.fields(expr):
            value = getattr(expr, field.name)
            if isinstance(value, griffe.Expr):
                setattr(expr, field.name, _rewrite(value))
            elif isinstance(value, list):
                value[:] = [_rewrite(v) for v in value]
    return expr


def _walk(obj: griffe.Object) -> Iterator[griffe.Object]:
    for member in obj.members.values():
        if not member.is_alias:
            yield member  # type: ignore[misc]
            yield from _walk(member)  # type: ignore[arg-type]


class TypeParameterBoundLinks(griffe.Extension):
    """Link type parameters in annotations to their bound."""

    def on_package(self, *, pkg: griffe.Module, **kwargs: Any) -> None:
        for obj in _walk(pkg):
            if isinstance(obj, griffe.Function):
                for param in obj.parameters:
                    param.annotation = _rewrite(param.annotation)
                obj.returns = _rewrite(obj.returns)
            elif isinstance(obj, griffe.Attribute):
                obj.annotation = _rewrite(obj.annotation)
            elif isinstance(obj, griffe.TypeAlias):
                obj.value = _rewrite(obj.value)
            elif isinstance(obj, griffe.Class):
                obj.bases[:] = [_rewrite(base) for base in obj.bases]
