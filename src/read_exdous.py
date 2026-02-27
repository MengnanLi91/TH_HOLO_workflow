import re

from netCDF4 import Dataset

try:
    from rich.console import Console
    from rich.table import Table

    _CONSOLE = Console()
    _USE_RICH = True
except Exception:
    _CONSOLE = None
    _USE_RICH = False

def _short(x, n=80):
    s = repr(x)
    return s if len(s) <= n else s[: n - 3] + "..."


def _print_line(text, style=None):
    if _USE_RICH and style:
        _CONSOLE.print(text, style=style)
    else:
        print(text)


def _section(title, indent=""):
    _print_line(f"{indent}{title}", style="bold")


def _indent_lines(lines, indent):
    for line in lines:
        print(f"{indent}{line}")


def _decode_name_rows(char2d):
    """
    Decode Exodus name_* variables (typically 2D char arrays) into a list of strings.
    Works for dtype '|S1' and similar.
    """
    names = []
    for row in char2d:
        s = row.tobytes().decode("utf-8", "ignore").rstrip("\x00").strip()
        if s:
            names.append(s)
    return names

def _decode_exodus_result_variable_names(ds):
    """
    Decode the *result* variable names (what ParaView typically shows),
    if present in this Exodus/NetCDF file.
    """
    name_vars = {
        "Global": "name_glo_var",
        "Nodal (point)": "name_nod_var",
        "Element (cell)": "name_elem_var",
        "Node set": "name_nset_var",
        "Side set": "name_sset_var",
    }

    names_by_kind = {}
    for label, vname in name_vars.items():
        if vname in ds.variables:
            try:
                names = _decode_name_rows(ds.variables[vname][:])
            except Exception as e:
                names_by_kind[vname] = []
                continue
            names_by_kind[vname] = names

    return names_by_kind


def _build_name_lookup(ds):
    names_by_kind = _decode_exodus_result_variable_names(ds)
    return {
        "global": names_by_kind.get("name_glo_var", []),
        "nodal": names_by_kind.get("name_nod_var", []),
        "element": names_by_kind.get("name_elem_var", []),
        "nodeset": names_by_kind.get("name_nset_var", []),
        "sideset": names_by_kind.get("name_sset_var", []),
    }


def _decode_entity_names(ds):
    entity_vars = {
        "eb": "eb_names",
        "ns": "ns_names",
        "ss": "ss_names",
    }
    out = {}
    for key, vname in entity_vars.items():
        if vname in ds.variables:
            try:
                out[key] = _decode_name_rows(ds.variables[vname][:])
            except Exception:
                out[key] = []
        else:
            out[key] = []
    return out


def _resolve_exodus_var_name(vname, names_by_kind, entity_names):
    patterns = [
        (r"^vals_glo_var$", "global", None),
        (r"^vals_nod_var(\d+)$", "nodal", 1),
        (r"^vals_elem_var(\d+)eb(\d+)$", "element", 1),
        (r"^vals_elem_var(\d+)$", "element", 1),
        (r"^vals_nset_var(\d+)ns(\d+)$", "nodeset", 1),
        (r"^vals_nset_var(\d+)$", "nodeset", 1),
        (r"^vals_sset_var(\d+)ss(\d+)$", "sideset", 1),
        (r"^vals_sset_var(\d+)$", "sideset", 1),
    ]

    if vname in {"eb_status", "eb_prop1"} and entity_names["eb"]:
        return ", ".join(entity_names["eb"])
    if vname in {"ns_status", "ns_prop1"} and entity_names["ns"]:
        return ", ".join(entity_names["ns"])
    if vname in {"ss_status", "ss_prop1"} and entity_names["ss"]:
        return ", ".join(entity_names["ss"])
    if vname == "eb_names" and entity_names["eb"]:
        return ", ".join(entity_names["eb"])
    if vname == "ns_names" and entity_names["ns"]:
        return ", ".join(entity_names["ns"])
    if vname == "ss_names" and entity_names["ss"]:
        return ", ".join(entity_names["ss"])

    match = re.match(r"^connect(\d+)$", vname)
    if match and entity_names["eb"]:
        idx = int(match.group(1))
        if 1 <= idx <= len(entity_names["eb"]):
            return entity_names["eb"][idx - 1]

    match = re.match(r"^node_ns(\d+)$", vname)
    if match and entity_names["ns"]:
        idx = int(match.group(1))
        if 1 <= idx <= len(entity_names["ns"]):
            return entity_names["ns"][idx - 1]

    match = re.match(r"^(elem|side)_ss(\d+)$", vname)
    if match and entity_names["ss"]:
        idx = int(match.group(2))
        if 1 <= idx <= len(entity_names["ss"]):
            return entity_names["ss"][idx - 1]

    for pattern, kind, group in patterns:
        match = re.match(pattern, vname)
        if not match:
            continue
        if group is None:
            return ", ".join(names_by_kind.get(kind, [])) or None
        idx = int(match.group(group))
        names = names_by_kind.get(kind, [])
        if 1 <= idx <= len(names):
            return names[idx - 1]
        return None
    return None

def print_exodus_tree(path: str):
    ds = Dataset(path, "r")
    names_by_kind = _build_name_lookup(ds)
    entity_names = _decode_entity_names(ds)

    def walk(g, name="/", indent=""):
        _print_line(f"{indent}{name}", style="bold cyan")
        if name == "/":
            if any(names_by_kind.values()):
                _section("  Result Variable Names", indent=indent)
                if _USE_RICH:
                    table = Table(show_header=True, header_style="bold blue")
                    table.add_column("kind", style="blue")
                    table.add_column("names", style="white")
                    for kind, names in names_by_kind.items():
                        if names:
                            table.add_row(kind, ", ".join(names))
                    _CONSOLE.print(table)
                else:
                    for kind, names in names_by_kind.items():
                        if names:
                            _print_line(
                                f"{indent}    {kind}: {', '.join(names)}",
                                style="blue",
                            )
            else:
                _print_line(
                    f"{indent}  Result Variable Names: (none)",
                    style="dim",
                )

        if g.ncattrs():
            _section("  Attributes", indent=indent)
            for a in g.ncattrs():
                _print_line(
                    f"{indent}    @{a} = {_short(getattr(g, a))}",
                    style="dim",
                )

        if g.dimensions:
            _section("  Dimensions", indent=indent)
            for dname, dim in g.dimensions.items():
                size = "UNLIMITED" if dim.isunlimited() else len(dim)
                _print_line(f"{indent}    {dname} = {size}", style="green")

        if g.variables:
            _section(f"  Variables ({len(g.variables)})", indent=indent)
            if _USE_RICH:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("name", style="magenta")
                table.add_column("dtype", style="cyan")
                table.add_column("shape", style="green")
                table.add_column("resolved", style="yellow")
                for vname, var in g.variables.items():
                    resolved = _resolve_exodus_var_name(vname, names_by_kind, entity_names) or ""
                    table.add_row(vname, str(var.dtype), str(var.shape), resolved)
                _CONSOLE.print(table)
                for vname, var in g.variables.items():
                    for a in var.ncattrs():
                        _print_line(
                            f"{indent}    {vname}.@{a} = {_short(getattr(var, a))}",
                            style="dim",
                        )
            else:
                for vname, var in g.variables.items():
                    resolved = _resolve_exodus_var_name(vname, names_by_kind, entity_names)
                    label = f"{vname} dtype={var.dtype} shape={var.shape}"
                    if resolved:
                        label += f" name={resolved}"
                    _print_line(f"{indent}    {label}", style="magenta")
                    for a in var.ncattrs():
                        _print_line(
                            f"{indent}      @{a} = {_short(getattr(var, a))}",
                            style="dim",
                        )

        if getattr(g, "groups", {}):
            _section(f"  Subgroups ({len(g.groups)})", indent=indent)
            for sgname, sg in g.groups.items():
                walk(sg, name=f"{sgname}/", indent=indent + "    ")

    walk(ds)
    ds.close()

if __name__ == "__main__":
    print_exodus_tree("lid-driven-segregated_out.e")
