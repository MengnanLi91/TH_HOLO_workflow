"""Helpers for decoding and inspecting Exodus netCDF files."""

import re

from netCDF4 import Dataset

try:
    from rich.console import Console
    from rich.table import Table

    _RICH_AVAILABLE = True
except Exception:
    Console = None
    Table = None
    _RICH_AVAILABLE = False


class ExodusReader:
    """Decode Exodus metadata and optionally pretty-print dataset trees."""

    def __init__(self, use_rich: bool | None = None):
        if use_rich is None:
            use_rich = _RICH_AVAILABLE
        self.use_rich = bool(use_rich and _RICH_AVAILABLE)
        self.console = Console() if self.use_rich else None

    @staticmethod
    def _short(value, n: int = 80) -> str:
        text = repr(value)
        return text if len(text) <= n else text[: n - 3] + "..."

    def _print_line(self, text: str, style: str | None = None) -> None:
        if self.use_rich and style and self.console is not None:
            self.console.print(text, style=style)
        else:
            print(text)

    def _section(self, title: str, indent: str = "") -> None:
        self._print_line(f"{indent}{title}", style="bold")

    @staticmethod
    def decode_name_rows(char2d) -> list[str]:
        """Decode Exodus name_* arrays (typically 2D char arrays)."""
        names = []
        for row in char2d:
            decoded = row.tobytes().decode("utf-8", "ignore").rstrip("\x00").strip()
            if decoded:
                names.append(decoded)
        return names

    def decode_exodus_result_variable_names(self, ds) -> dict[str, list[str]]:
        """Decode result variable names shown in ParaView, when present."""
        name_vars = {
            "Global": "name_glo_var",
            "Nodal (point)": "name_nod_var",
            "Element (cell)": "name_elem_var",
            "Node set": "name_nset_var",
            "Side set": "name_sset_var",
        }

        names_by_kind = {}
        for _label, vname in name_vars.items():
            if vname in ds.variables:
                try:
                    names = self.decode_name_rows(ds.variables[vname][:])
                except Exception:
                    names_by_kind[vname] = []
                    continue
                names_by_kind[vname] = names
        return names_by_kind

    def build_name_lookup(self, ds) -> dict[str, list[str]]:
        """Build normalized name lookup for global/nodal/element/etc."""
        names_by_kind = self.decode_exodus_result_variable_names(ds)
        return {
            "global": names_by_kind.get("name_glo_var", []),
            "nodal": names_by_kind.get("name_nod_var", []),
            "element": names_by_kind.get("name_elem_var", []),
            "nodeset": names_by_kind.get("name_nset_var", []),
            "sideset": names_by_kind.get("name_sset_var", []),
        }

    def decode_entity_names(self, ds) -> dict[str, list[str]]:
        """Decode named entity groups like element blocks/node sets/side sets."""
        entity_vars = {
            "eb": "eb_names",
            "ns": "ns_names",
            "ss": "ss_names",
        }
        out = {}
        for key, vname in entity_vars.items():
            if vname in ds.variables:
                try:
                    out[key] = self.decode_name_rows(ds.variables[vname][:])
                except Exception:
                    out[key] = []
            else:
                out[key] = []
        return out

    def resolve_exodus_var_name(
        self,
        vname: str,
        names_by_kind: dict[str, list[str]],
        entity_names: dict[str, list[str]],
    ) -> str | None:
        """Resolve an Exodus variable key to a user-facing variable name."""
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

    def print_exodus_tree(self, path: str) -> None:
        """Print a tree view of an Exodus file with resolved variable names."""
        with Dataset(path, "r") as ds:
            names_by_kind = self.build_name_lookup(ds)
            entity_names = self.decode_entity_names(ds)

            def walk(group, name: str = "/", indent: str = "") -> None:
                self._print_line(f"{indent}{name}", style="bold cyan")
                if name == "/":
                    if any(names_by_kind.values()):
                        self._section("  Result Variable Names", indent=indent)
                        if self.use_rich and self.console is not None:
                            table = Table(show_header=True, header_style="bold blue")
                            table.add_column("kind", style="blue")
                            table.add_column("names", style="white")
                            for kind, names in names_by_kind.items():
                                if names:
                                    table.add_row(kind, ", ".join(names))
                            self.console.print(table)
                        else:
                            for kind, names in names_by_kind.items():
                                if names:
                                    self._print_line(
                                        f"{indent}    {kind}: {', '.join(names)}",
                                        style="blue",
                                    )
                    else:
                        self._print_line(
                            f"{indent}  Result Variable Names: (none)",
                            style="dim",
                        )

                if group.ncattrs():
                    self._section("  Attributes", indent=indent)
                    for attr in group.ncattrs():
                        self._print_line(
                            f"{indent}    @{attr} = {self._short(getattr(group, attr))}",
                            style="dim",
                        )

                if group.dimensions:
                    self._section("  Dimensions", indent=indent)
                    for dname, dim in group.dimensions.items():
                        size = "UNLIMITED" if dim.isunlimited() else len(dim)
                        self._print_line(f"{indent}    {dname} = {size}", style="green")

                if group.variables:
                    self._section(f"  Variables ({len(group.variables)})", indent=indent)
                    if self.use_rich and self.console is not None:
                        table = Table(show_header=True, header_style="bold magenta")
                        table.add_column("name", style="magenta")
                        table.add_column("dtype", style="cyan")
                        table.add_column("shape", style="green")
                        table.add_column("resolved", style="yellow")
                        for vname, var in group.variables.items():
                            resolved = (
                                self.resolve_exodus_var_name(
                                    vname, names_by_kind, entity_names
                                )
                                or ""
                            )
                            table.add_row(vname, str(var.dtype), str(var.shape), resolved)
                        self.console.print(table)
                        for vname, var in group.variables.items():
                            for attr in var.ncattrs():
                                self._print_line(
                                    f"{indent}    {vname}.@{attr} = "
                                    f"{self._short(getattr(var, attr))}",
                                    style="dim",
                                )
                    else:
                        for vname, var in group.variables.items():
                            resolved = self.resolve_exodus_var_name(
                                vname, names_by_kind, entity_names
                            )
                            label = f"{vname} dtype={var.dtype} shape={var.shape}"
                            if resolved:
                                label += f" name={resolved}"
                            self._print_line(f"{indent}    {label}", style="magenta")
                            for attr in var.ncattrs():
                                self._print_line(
                                    f"{indent}      @{attr} = "
                                    f"{self._short(getattr(var, attr))}",
                                    style="dim",
                                )

                if getattr(group, "groups", {}):
                    self._section(f"  Subgroups ({len(group.groups)})", indent=indent)
                    for sgname, subgroup in group.groups.items():
                        walk(subgroup, name=f"{sgname}/", indent=indent + "    ")

            walk(ds)


_DEFAULT_READER = ExodusReader()


def decode_name_rows(char2d):
    """Decode Exodus name_* arrays into a list of strings."""
    return _DEFAULT_READER.decode_name_rows(char2d)


def decode_exodus_result_variable_names(ds):
    """Decode result variable names present in an open Exodus dataset."""
    return _DEFAULT_READER.decode_exodus_result_variable_names(ds)


def build_name_lookup(ds):
    """Build normalized name lookup for global/nodal/element/etc."""
    return _DEFAULT_READER.build_name_lookup(ds)


def decode_entity_names(ds):
    """Decode Exodus entity names (element blocks/node sets/side sets)."""
    return _DEFAULT_READER.decode_entity_names(ds)


def resolve_exodus_var_name(vname, names_by_kind, entity_names):
    """Resolve an Exodus variable key to a user-facing variable name."""
    return _DEFAULT_READER.resolve_exodus_var_name(vname, names_by_kind, entity_names)


def print_exodus_tree(path: str):
    """Print a tree view of an Exodus file and resolved variable names."""
    _DEFAULT_READER.print_exodus_tree(path)


if __name__ == "__main__":
    print_exodus_tree("lid-driven-segregated_out.e")
