"""Generate ``docs/topics/README.md``: the ROS 2 topic/service/action graph.

Python import graphs cannot see how OpenRAL's layers talk at runtime — a
publisher and its subscriber share only a topic *string*. This walks the
source tree statically and joins every endpoint on its name:

* **Python** (``python/``, ``packages/``, ``tools/``): ``create_publisher`` /
  ``create_subscription`` / ``create_service`` / ``create_client`` calls and
  ``ActionServer`` / ``ActionClient`` constructions. The name argument is
  resolved through string literals, f-strings, module constants (also across
  ``from x import NAME``), class attributes, ``self._x = ...`` assignments,
  parameter defaults, ``declare_parameter(name, default)`` and
  ``openral_core.camera_topic(name, kind)`` (ADR-0108).
* **C++** (``cpp/``, ``packages/``): ``create_publisher<T>`` /
  ``create_subscription<T>`` / ``create_service<T>`` / ``create_client<T>``
  and ``rclcpp_action::create_server<T>`` / ``create_client<T>``.
* **Launch files**: ``remappings=`` pairs of string literals.

A name that is only known at runtime is listed under "Unresolved endpoints"
with its source expression, never guessed. A name resolved around a runtime
placeholder (``/openral/{robot}/reset_to_pose``) is listed under the
"Per-instance" section of its kind, apart from the concrete names. Test trees are excluded (they
publish fixtures, not the product graph). Endpoints cite ``path`` +
enclosing ``Class.method`` rather than a line number, so the page changes
only when the graph does.

Usage::

    uv run python tools/gen_ros_topic_graph.py            # rewrite docs/topics/README.md
    uv run python tools/gen_ros_topic_graph.py --check    # exit 1 if it is stale

Not part of the runtime; CI-adjacent doc tooling only.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from openral_core import CAMERA_TOPIC_PREFIX, CameraTopicKind, camera_topic

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "docs" / "topics" / "README.md"
SCAN_ROOTS = ("python", "packages", "cpp", "tools")
_EXCLUDED_PARTS = frozenset(
    {"test", "tests", "__pycache__", ".venv", "build", "install", "vendor", "fakes"}
)

# name -> (kind, role, index of the name argument, type keyword, name keyword); the
# keywords are rclpy's own (``Node.create_publisher(msg_type, topic, ...)``,
# ``create_service(srv_type, srv_name, ...)``, ``ActionServer(node, action_type,
# action_name, ...)``), the type argument sits right before the name.
_PY_METHODS: dict[str, tuple[str, str, int, str, str]] = {
    "create_publisher": ("topic", "pub", 1, "msg_type", "topic"),
    "create_subscription": ("topic", "sub", 1, "msg_type", "topic"),
    "create_service": ("service", "server", 1, "srv_type", "srv_name"),
    "create_client": ("service", "client", 1, "srv_type", "srv_name"),
}
_PY_CTORS: dict[str, tuple[str, str, int, str, str]] = {
    "ActionServer": ("action", "server", 2, "action_type", "action_name"),
    "ActionClient": ("action", "client", 2, "action_type", "action_name"),
}
# Every camera_topic() name that is not a literal is, by contract, a SensorSpec
# name — the one placeholder allowed to join rows from different call sites.
_SENSOR_PLACEHOLDER = "{sensor}"
_CPP_RE = re.compile(
    r"(?P<action>rclcpp_action::)?create_(?P<what>publisher|subscription|service|client|server)"
    r"\s*<\s*(?P<type>[\w:]+)\s*>\s*\(",
)
_CPP_PARAM_RE = re.compile(
    r"(?P<var>\w+)\s*=\s*(?:this->)?declare_parameter\s*<[^>]*>\s*\(\s*"
    r"\"(?P<param>\w+)\"\s*,\s*(?:std::string\s*\(\s*)?\"(?P<default>[^\"]*)\""
)
_CPP_ROLES = {
    "publisher": ("topic", "pub"),
    "subscription": ("topic", "sub"),
    "service": ("service", "server"),
    "client": ("service", "client"),
}
_MAX_DEPTH = 6


@dataclass(frozen=True)
class Endpoint:
    """One side of a ROS connection found in source."""

    kind: str  # topic | service | action
    role: str  # pub | sub | server | client
    name: str  # resolved name, or the source expression when unresolved
    resolved: bool
    msg_type: str
    where: str  # "path (Class.method)"


def _endpoint(
    kind: str, role: str, name: str | None, src: str, msg_type: str, where: str
) -> Endpoint:
    """Build an endpoint; a ``"x (param `p`)"`` name moves its param note onto ``where``."""
    if name is None:
        return Endpoint(kind, role, " ".join(src.split()), False, msg_type, where)
    name, sep, note = name.partition(" (param ")
    return Endpoint(
        kind, role, name, True, msg_type, f"{where} [param {note[:-1]}]" if sep else where
    )


def _module_name(path: Path) -> str | None:
    """Dotted import name of ``path``, walking up while ``__init__.py`` exists."""
    parts = [path.stem] if path.stem != "__init__" else []
    parent = path.parent
    while (parent / "__init__.py").exists():
        parts.insert(0, parent.name)
        parent = parent.parent
    return ".".join(parts) if parts else None


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def _iter_sources(suffixes: tuple[str, ...]) -> list[Path]:
    """Tracked source files only (``git ls-files``: the index, so a staged new file counts).

    Walking the filesystem would let a scratch script, a stray ``venv/`` or a colcon
    ``log/`` change the page on one machine, and CI's ``--check`` on a clean checkout
    would then disagree with what pre-commit wrote.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--", *SCAN_ROOTS],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"gen_ros_topic_graph needs a git checkout: {exc}") from exc
    files: list[Path] = []
    for rel in out.decode().split("\0"):
        path = REPO_ROOT / rel
        if (
            rel
            and path.suffix in suffixes
            and not _EXCLUDED_PARTS.intersection(Path(rel).parts)
            and path.is_file()
        ):
            files.append(path)
    return sorted(files)


class _Resolver:
    """Resolves a name-argument expression to a string, statically."""

    def __init__(self, trees: dict[Path, ast.Module]) -> None:
        self.trees = trees
        self.by_module: dict[str, Path] = {}
        for path in trees:
            mod = _module_name(path)
            if mod:
                self.by_module.setdefault(mod, path)
        self.parents: dict[ast.AST, ast.AST] = {}
        for tree in trees.values():
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    self.parents[child] = node

    def enclosing(self, node: ast.AST, kind: type | tuple[type, ...]) -> ast.AST | None:
        cur = self.parents.get(node)
        while cur is not None and not isinstance(cur, kind):
            cur = self.parents.get(cur)
        return cur

    def qualname(self, node: ast.AST) -> str:
        names: list[str] = []
        cur = self.parents.get(node)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.insert(0, cur.name)
            cur = self.parents.get(cur)
        return ".".join(names) or "<module>"

    def resolve(  # noqa: PLR0911  # reason: one explicit return per expression shape
        self, expr: ast.expr, path: Path, depth: int = 0
    ) -> str | None:
        if depth > _MAX_DEPTH:
            return None
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value
        if isinstance(expr, ast.JoinedStr):
            out, notes = [], []
            for part in expr.values:
                if isinstance(part, ast.Constant):
                    out.append(str(part.value))
                elif isinstance(part, ast.FormattedValue):
                    val = self.resolve(part.value, path, depth + 1)
                    if val is None:
                        out.append("{" + ast.unparse(part.value) + "}")
                        continue
                    # Keep a nested param note out of the middle of the name.
                    val, sep, note = val.partition(" (param ")
                    out.append(val)
                    notes += [f" (param {note}"] if sep else []
            return "".join(out) + "".join(notes[:1])
        if isinstance(expr, ast.Name):
            return self._resolve_name(expr, path, depth)
        param = _get_parameter_name(expr)
        if param is not None:
            return self._declared_default(param, path, depth)
        if isinstance(expr, ast.Attribute):
            return self._resolve_attribute(expr, path, depth)
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "str":
            return self.resolve(expr.args[0], path, depth + 1) if expr.args else None
        if isinstance(expr, ast.Call) and _callee(expr) == "camera_topic":
            return self._camera_topic(expr, path, depth)
        if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or):
            # `topic or DEFAULT` -> the default is the only statically known value.
            return self.resolve(expr.values[-1], path, depth + 1)
        return None

    def _resolve_name(self, expr: ast.Name, path: Path, depth: int) -> str | None:
        funcs = (ast.FunctionDef, ast.AsyncFunctionDef)
        scope = self.enclosing(expr, funcs)
        while isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # The last `name = <expr>` before the use wins, then a parameter default.
            value = _last_assignment(ast.walk(scope), expr.id, before=expr.lineno)
            if value is not None:
                return self.resolve(value, path, depth + 1)
            args = scope.args
            positional = args.posonlyargs + args.args
            defaults = dict(
                zip([a.arg for a in positional[::-1]], args.defaults[::-1], strict=False)
            )
            defaults.update(
                {a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True) if d}
            )
            if expr.id in defaults:
                return self.resolve(defaults[expr.id], path, depth + 1)
            params = [*positional, *args.kwonlyargs, args.vararg, args.kwarg]
            if expr.id in {a.arg for a in params if a is not None}:
                return None  # a caller's argument: no module constant may stand in
            scope = self.enclosing(scope, funcs)
        return self._module_constant(expr.id, path, depth)

    def _camera_topic(self, call: ast.Call, path: Path, depth: int) -> str | None:
        """``camera_topic(name, kind, prefix=...)`` through the real function."""
        kw = {k.arg: k.value for k in call.keywords}
        name_expr = call.args[0] if call.args else kw.get("name")
        kind_expr = call.args[1] if len(call.args) > 1 else kw.get("kind")
        name = self.resolve(name_expr, path, depth + 1) if name_expr else None
        if isinstance(kind_expr, ast.Attribute) and kind_expr.attr in CameraTopicKind.__members__:
            kind: str | None = CameraTopicKind[kind_expr.attr].value
        else:
            kind = self.resolve(kind_expr, path, depth + 1) if kind_expr else "image"
        prefix_expr = kw.get("prefix")
        prefix = self.resolve(prefix_expr, path, depth + 1) if prefix_expr else CAMERA_TOPIC_PREFIX
        if kind not in {k.value for k in CameraTopicKind} or prefix is None:
            return None
        if not name or "/" in name or " (param " in name:
            name = _SENSOR_PLACEHOLDER
        return camera_topic(name, CameraTopicKind(kind), prefix=prefix)

    def _module_constant(self, name: str, path: Path, depth: int) -> str | None:
        tree = self.trees.get(path)
        if tree is None:
            return None
        module_level = (
            n
            for n in ast.walk(tree)
            if self.enclosing(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) is None
        )
        value = _last_assignment(module_level, name, before=None)
        if value is not None:
            return self.resolve(value, path, depth + 1)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if (alias.asname or alias.name) == name:
                        target = self._import_target(node, path)
                        if target is not None:
                            return self._module_constant(alias.name, target, depth + 1)
        return None

    def _import_target(self, node: ast.ImportFrom, path: Path) -> Path | None:
        module = node.module or ""
        if node.level:
            base = _module_name(path) or ""
            pkg = base.split(".")[: -node.level] if path.stem != "__init__" else base.split(".")
            module = ".".join([*pkg, module]) if module else ".".join(pkg)
        return self.by_module.get(module) or self.by_module.get(f"{module}.__init__")

    def _resolve_attribute(self, expr: ast.Attribute, path: Path, depth: int) -> str | None:
        if not isinstance(expr.value, ast.Name):
            return None
        cls = self.enclosing(expr, ast.ClassDef)
        if expr.value.id not in ("self", "cls") or not isinstance(cls, ast.ClassDef):
            # `SomeClass.ATTR` or `module.CONST` in the same file.
            tree = self.trees.get(path)
            for node in ast.walk(tree) if tree else ():
                if isinstance(node, ast.ClassDef) and node.name == expr.value.id:
                    return self._class_attr(node, expr.attr, path, depth)
            return self._argparse_default(expr.attr, path, depth)
        return self._class_attr(cls, expr.attr, path, depth)

    def _class_attr(self, cls: ast.ClassDef, attr: str, path: Path, depth: int) -> str | None:
        value = _last_assignment(cls.body, attr, before=None)
        if value is not None:
            return self.resolve(value, path, depth + 1)
        for sub in ast.walk(cls):
            if isinstance(sub, (ast.Assign, ast.AnnAssign)):
                targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                for tgt in targets:
                    if (
                        isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"
                        and tgt.attr == attr
                        and sub.value is not None
                    ):
                        val = self.resolve(sub.value, path, depth + 1)
                        if val is not None:
                            return val
        return None

    def _argparse_default(self, dest: str, path: Path, depth: int) -> str | None:
        """``args.x`` -> ``add_argument("--x", default=...)`` in ``path``."""
        tree = self.trees.get(path)
        for node in ast.walk(tree) if tree else ():
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                continue
            flags = [str(a.value) for a in node.args if isinstance(a, ast.Constant)]
            kw = {k.arg: k.value for k in node.keywords}
            dest_kw = kw.get("dest")
            names = {f.lstrip("-").replace("-", "_") for f in flags}
            if isinstance(dest_kw, ast.Constant):
                names = {str(dest_kw.value)}
            if dest in names and "default" in kw:
                default = self.resolve(kw["default"], path, depth + 1)
                return f"{default} (param `{flags[0]}`)" if default else None
        return None

    def _declared_default(self, param: str, path: Path, depth: int) -> str | None:
        """``declare_parameter("x", default)`` in ``path`` -> ``"default (param `x`)"``."""
        tree = self.trees.get(path)
        for node in ast.walk(tree) if tree else ():
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "declare_parameter"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == param
            ):
                default = self.resolve(node.args[1], path, depth + 1)
                # An empty default means "unset unless launch passes one": not a name.
                return f"{default} (param `{param}`)" if default else None
        return None


def _last_assignment(nodes: Iterable[ast.AST], name: str, before: int | None) -> ast.expr | None:
    """Value of the last ``name = ...`` / ``name: T = ...`` in ``nodes`` (before line ``before``)."""
    best: tuple[int, ast.expr] | None = None
    for node in nodes:
        if before is not None and getattr(node, "lineno", 0) >= before:
            continue
        if isinstance(node, ast.Assign):
            hit = any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            hit = isinstance(node.target, ast.Name) and node.target.id == name
        else:
            continue
        if hit and node.value is not None and (best is None or node.lineno > best[0]):
            best = (node.lineno, node.value)
    return best[1] if best else None


def _get_parameter_name(expr: ast.expr) -> str | None:
    """The parameter name in ``self.get_parameter("x")...`` or an aliased ``gp("x")...``."""
    for node in ast.walk(expr):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            inner: ast.expr = node.func.value
            if node.func.attr == "get_parameter":
                inner = node
            elif node.func.attr != "get_parameter_value":
                continue
            if (
                isinstance(inner, ast.Call)
                and inner.args
                and isinstance(inner.args[0], ast.Constant)
                and isinstance(inner.args[0].value, str)
            ):
                return inner.args[0].value
    return None


def _message_imports(tree: ast.Module) -> dict[str, str]:
    """Local alias -> ``pkg/Type`` for every ``from pkg.msg|srv|action import Type``."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            pkg, _, sub = node.module.rpartition(".")
            if sub in ("msg", "srv", "action") and pkg:
                for alias in node.names:
                    out.setdefault(alias.asname or alias.name, f"{pkg}/{alias.name}")
    return out


def _callee(call: ast.Call) -> str | None:
    """``f`` for ``f(...)`` and ``mod.f(...)``."""
    func = call.func
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)


def _arg(call: ast.Call, index: int, keyword: str) -> ast.expr | None:
    """Argument ``index`` of ``call``, positional or by ``keyword``."""
    if len(call.args) > index:
        return call.args[index]
    return next((k.value for k in call.keywords if k.arg == keyword), None)


def extract_python(files: list[Path]) -> list[Endpoint]:
    """Every Python endpoint in ``files``."""
    trees: dict[Path, ast.Module] = {}
    for path in files:
        try:
            trees[path] = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise SystemExit(f"cannot parse {path}: {exc}") from exc
    resolver = _Resolver(trees)
    out: list[Endpoint] = []
    for path, tree in trees.items():
        rel = _rel(path)
        msg_types = _message_imports(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            spec = None
            if isinstance(func, ast.Attribute) and func.attr in _PY_METHODS:
                spec = _PY_METHODS[func.attr]
            elif isinstance(func, ast.Name) and func.id in _PY_CTORS:
                spec = _PY_CTORS[func.id]
            if spec is None:
                continue
            kind, role, index, type_kw, name_kw = spec
            name_expr = _arg(node, index, name_kw)
            type_expr = _arg(node, index - 1, type_kw)
            if name_expr is None or type_expr is None:
                continue  # a same-named helper with another signature
            type_src = ast.unparse(type_expr)
            out.append(
                _endpoint(
                    kind,
                    role,
                    resolver.resolve(name_expr, path),
                    ast.unparse(name_expr),
                    msg_types.get(type_src, type_src),
                    f"{rel} ({resolver.qualname(node)})",
                )
            )
    return out


def extract_cpp(files: list[Path]) -> list[Endpoint]:
    """Every C++ endpoint in ``files``; only string-literal names resolve."""
    out: list[Endpoint] = []
    for path in files:
        rel = _rel(path)
        text = path.read_text(encoding="utf-8")
        # `const auto x = this->declare_parameter<std::string>("x", "/default");`
        params = {
            m["var"]: f"{m['default']} (param `{m['param']}`)" for m in _CPP_PARAM_RE.finditer(text)
        }
        for match in _CPP_RE.finditer(text):
            what = match["what"]
            args = _cpp_args(text, match.end())
            if match["action"]:
                kind, role = "action", "server" if what == "server" else "client"
                # rclcpp_action overloads: (node, name, ...) or the four node
                # interfaces then name. The node form takes at most 7 args for a
                # server (4 for a client); the interface form at least 8 (5).
                interfaces = len(args) >= (8 if role == "server" else 5)
                index = 4 if interfaces else 1
            elif what == "server":
                continue
            else:
                kind, role = _CPP_ROLES[what]
                index = 0
            if len(args) <= index:
                continue
            name_src = args[index]
            literal = re.fullmatch(r'"([^"]*)"', name_src)
            name = literal[1] if literal else params.get(name_src)
            msg_type = re.sub(r"::(msg|srv|action)::", "/", match["type"])
            out.append(_endpoint(kind, role, name, name_src, msg_type, rel))
    return out


def _cpp_args(text: str, start: int) -> list[str]:
    """Top-level arguments of the C++ call whose ``(`` ends at ``start``.

    Balances ``()[]{}<>`` (``->`` is not a bracket) and skips string literals.
    """
    args: list[str] = []
    depth, i, begin = 0, start, start
    while i < len(text):
        ch = text[i]
        if ch == '"':
            i += 1
            while i < len(text) and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif ch == "-" and text.startswith("->", i):
            i += 1
        elif ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            if depth == 0:
                break
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(text[begin:i].strip())
            begin = i + 1
        elif ch == ";" and depth == 0:
            break  # never past the statement
        i += 1
    last = text[begin:i].strip()
    return [*args, last] if last or args else []


def extract_remappings(files: list[Path]) -> list[tuple[str, str, str]]:
    """``(from, to, launch file)`` for every literal remapping pair."""
    out: list[tuple[str, str, str]] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = _rel(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "remappings":
                for elt in getattr(node.value, "elts", []):
                    if isinstance(elt, ast.Tuple) and len(elt.elts) == 2:
                        src, dst = elt.elts
                        out.append((_launch_str(src), _launch_str(dst), rel))
    return sorted(set(out))


def _launch_str(expr: ast.expr) -> str:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return f"`{expr.value}`"
    return f"‹{ast.unparse(expr)}›"


_ROLE_COLUMNS = {
    "topic": ("pub", "sub", "Publishers", "Subscribers"),
    "service": ("server", "client", "Servers", "Clients"),
    "action": ("server", "client", "Servers", "Clients"),
}


def _cell(endpoints: list[Endpoint], role: str) -> str:
    return "<br>".join(sorted({e.where for e in endpoints if e.role == role})) or "—"


def render(endpoints: list[Endpoint], remaps: list[tuple[str, str, str]]) -> str:
    """The full Markdown page."""
    lines = [
        "# ROS 2 graph",
        "",
        "<!-- GENERATED by tools/gen_ros_topic_graph.py — do not edit by hand. -->",
        "",
        "Every topic, service and action OpenRAL's own nodes declare, joined on name.",
        "Extracted statically from `python/`, `packages/`, `tools/` and `cpp/` (tests excluded),",
        "so it shows the wiring the code *can* create; which parts run depends on the",
        "robot manifest and launch arguments. A [param `x`] note on an endpoint means the",
        "name is that ROS parameter's declared default, overridable at launch. A row with",
        "`—` on one side talks to something outside this repo (ros2_control,",
        "Nav2, a driver) or is a dead end worth checking. A `{placeholder}` in a",
        "per-instance name is filled at runtime (a robot id, a manifest sensor name);",
        "rows join only on identical placeholder text, except that every",
        "`openral_core.camera_topic(<name>)` is `{sensor}` — a `SensorSpec.name` by contract.",
        "",
        "Regenerate with `uv run python tools/gen_ros_topic_graph.py`;",
        "`just lint` and pre-commit fail when this page is stale.",
        "",
    ]
    sections = [
        (kind, (prefix + title).capitalize(), per_instance)
        for kind, title in (("topic", "topics"), ("service", "services"), ("action", "actions"))
        for prefix, per_instance in (("", False), ("Per-instance ", True))
    ]
    for kind, title, per_instance in sections:
        a, b, a_title, b_title = _ROLE_COLUMNS[kind]
        groups: dict[str, list[Endpoint]] = defaultdict(list)
        for e in endpoints:
            if e.kind == kind and e.resolved and ("{" in e.name) == per_instance:
                groups[e.name].append(e)
        if per_instance and not groups:
            continue
        lines += [
            f"## {title} ({len(groups)})",
            "",
            f"| Name | Type | {a_title} | {b_title} |",
            "|---|---|---|---|",
        ]
        for name in sorted(groups):
            eps = groups[name]
            types = ", ".join(f"`{t}`" for t in sorted({e.msg_type for e in eps}))
            lines.append(f"| `{name}` | {types} | {_cell(eps, a)} | {_cell(eps, b)} |")
        lines.append("")
    unresolved = sorted(
        {(e.kind, e.role, e.name, e.msg_type, e.where) for e in endpoints if not e.resolved}
    )
    lines += [
        f"## Unresolved endpoints ({len(unresolved)})",
        "",
        "Names only known at runtime (built from a manifest, a robot id or a",
        "caller's argument). Shown as their source expression.",
        "",
        "| Kind | Role | Name expression | Type | Where |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {k} | {r} | `{n}` | `{t}` | {w} |" for k, r, n, t, w in unresolved]
    lines += [
        "",
        f"## Launch remappings ({len(remaps)})",
        "",
        "| From | To | Launch file |",
        "|---|---|---|",
    ]
    lines += [f"| {src} | {dst} | {where} |" for src, dst, where in remaps]
    return "\n".join(lines).replace("\n\n\n", "\n\n") + "\n"


def build() -> str:
    """Scan the tree and render the page."""
    py_files = _iter_sources((".py",))
    endpoints = extract_python(py_files) + extract_cpp(_iter_sources((".cpp", ".hpp", ".h")))
    remaps = extract_remappings([p for p in py_files if p.name.endswith(".launch.py")])
    return render(endpoints, remaps)


def main(argv: list[str] | None = None) -> int:
    """Write the page, or with ``--check`` exit 1 if the checked-in copy is stale."""
    check = "--check" in (argv if argv is not None else sys.argv[1:])
    rendered = build()
    if check:
        if OUT_PATH.exists() and OUT_PATH.read_text(encoding="utf-8") == rendered:
            return 0
        print(
            f"{OUT_PATH.relative_to(REPO_ROOT)} is stale — run "
            "`uv run python tools/gen_ros_topic_graph.py`",
            file=sys.stderr,
        )
        return 1
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
