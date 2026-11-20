#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the agentdoc pxr documentation index from official OpenUSD sources.

usd-core ships its Boost.Python bindings without docstrings. The OpenUSD C++
headers document most of the API. This script maps every resolvable pxr Python
name to its C++ declaration and writes a compact, version-tagged index that
``nooa.agentdoc`` shows in ``doc()`` and searches with ``doc.search()``.

Run it with the usd-core build that matches the source tree on PYTHONPATH::

    PYTHONPATH=<usd-core site dir>:src python scripts/agentdoc/build_pxr_index.py \
        --source <OpenUSD checkout or tarball dir> --version 26.08 \
        --out src/nooa/agentdoc/data/pxr-26.08.json.gz

Sources used, in order:
- Python names, kinds and Boost.Python C++ signatures come from the installed
  pxr modules. The signature is read from the ArgumentError that Boost.Python
  raises when argument matching fails; the call passes 13 instances of a private
  class, which no wrapped C++ overload accepts.
- ``wrap*.cpp`` gives ``class_<Cpp>("Py")`` and ``.def("Py", &Cpp::Method)``.
- Doxygen comments (``///``, ``//!``, ``/** */``, ``///<``) above or after C++
  declarations in ``pxr/**/*.h``, looked up through C++ base classes.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import importlib
import inspect
import json
import pkgutil
import re
import sys
from pathlib import Path

# ----------------------------------------------------------------------------- headers

_API_MACRO = re.compile(r"\b[A-Z][A-Z0-9_]*_(API|API_TYPE|LOCAL)\b")
_LEADING_MACRO = re.compile(
    r"^\s*(?:[A-Z][A-Z0-9_]{2,}\b(?:\s*\([^()]*(?:\([^()]*\)[^()]*)*\))?\s*)+"
)
_ACCESS = re.compile(r"^\s*(?:public|private|protected)\s*:(?!:)\s*")
_IDENT = re.compile(r"[A-Za-z_]\w*")


class Decl:
    __slots__ = ("docs", "loc", "kind")

    def __init__(self, loc: str, kind: str):
        self.docs: list[str] = []
        self.loc = loc
        self.kind = kind


class HeaderIndex:
    """Doc comments of C++ declarations, keyed by (scope class name, member name)."""

    def __init__(self) -> None:
        self.decls: dict[tuple[str, str], Decl] = {}
        self.bases: dict[str, list[str]] = collections.defaultdict(list)
        self.named_class_docs: dict[str, str] = {}

    def add(self, scope: str, name: str, doc: str | None, loc: str, kind: str) -> None:
        decl = self.decls.get((scope, name))
        if decl is None:
            decl = self.decls[(scope, name)] = Decl(loc, kind)
        elif doc and not decl.docs:
            decl.loc, decl.kind = loc, kind  # the documented declaration wins
        if doc and doc not in decl.docs:
            decl.docs.append(doc)

    def parse_file(self, path: Path, rel: str) -> None:
        text = path.read_text(errors="ignore")
        _Parser(self, text, rel).run()


def _strip_comment_markers(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(///<|//!<|///|//!|/\*\*<|/\*!<|/\*\*|/\*!|\*/|\*(?!/))", "", line)
        line = re.sub(r"\*/\s*$", "", line)
        lines.append(line.rstrip())
    return "\n".join(lines).strip("\n")


class _Parser:
    def __init__(self, index: HeaderIndex, text: str, rel: str):
        self.ix = index
        self.s = text
        self.rel = rel
        self.scopes: list[tuple[str, str]] = []  # (kind, name): class/enum/ns/body
        self.stmt: list[str] = []
        self.stmt_line = 0
        self.pending: list[str] = []
        self.last: tuple[str, str] | None = None
        self.line = 1
        self.has_code = False
        self.pending_end = 0

    # current class scope name ('' at namespace level)
    def scope(self) -> str:
        for kind, name in reversed(self.scopes):
            if kind in ("class", "enumclass"):
                return name
            if kind == "enum":
                continue
        return ""

    def in_body(self) -> bool:
        return any(kind == "body" for kind, _ in self.scopes)

    def run(self) -> None:
        s, n, i = self.s, len(self.s), 0
        at_line_start = True
        while i < n:
            c = s[i]
            if c == "\n":
                self.line += 1
                at_line_start = True
                self.stmt.append(" ")
                i += 1
                continue
            if at_line_start and c == "#":
                j = i
                while j < n:
                    k = s.find("\n", j)
                    if k < 0:
                        k = n
                    if s[k - 1 : k] == "\\":
                        j = k + 1
                        self.line += 1
                        continue
                    break
                i = k
                continue
            if not c.isspace():
                at_line_start = False
            if s.startswith("//", i):
                k = s.find("\n", i)
                k = n if k < 0 else k
                body = s[i:k]
                if not self.in_body() and (
                    body.startswith("///") and not body.startswith("////") or body.startswith("//!")
                ):
                    self.doc_comment(
                        body,
                        trailing=body.startswith(("///<", "//!<")),
                        start=self.line,
                        end=self.line,
                    )
                i = k
                continue
            if s.startswith("/*", i):
                k = s.find("*/", i + 2)
                k = n if k < 0 else k + 2
                body = s[i:k]
                start = self.line
                self.line += body.count("\n")
                if not self.in_body() and (
                    body.startswith("/**") and not body.startswith("/**/") or body.startswith("/*!")
                ):
                    self.doc_comment(
                        body, trailing=body.startswith(("/**<", "/*!<")), start=start, end=self.line
                    )
                i = k
                continue
            if c in "\"'":
                j = i + 1
                if c == '"' and s[i - 1 : i] == "R":
                    m = re.match(r'R"([^(]*)\(', s[i - 1 :])
                    if m:
                        end = s.find(")" + m.group(1) + '"', i)
                        j = n if end < 0 else end + len(m.group(1)) + 2
                        self.line += s[i:j].count("\n")
                        self.stmt.append('""')
                        i = j
                        continue
                while j < n and s[j] != c:
                    if s[j] == "\\":
                        j += 1
                    elif s[j] == "\n":
                        break
                    j += 1
                self.stmt.append('""')
                self.has_code = True
                i = j + 1
                continue
            if c == "{":
                self.open_brace()
            elif c == "}":
                self.close_brace()
            elif c == ";":
                self.end_stmt()
            elif c == "," and self.scopes and self.scopes[-1][0] in ("enum", "enumclass_body"):
                self.enumerator()
            else:
                if not self.has_code and not c.isspace():
                    self.stmt_line = self.line
                    self.has_code = True
                self.stmt.append(c)
            i += 1

    def doc_comment(self, raw: str, trailing: bool, start: int = 0, end: int = 0) -> None:
        text = _strip_comment_markers(raw)
        if trailing:
            # Documents the declaration before it (an enumerator or member).
            if self.has_code and self.scopes and self.scopes[-1][0] in ("enum", "enumclass_body"):
                self.enumerator(doc=text)
                return
            if self.last is not None:
                self.ix.add(
                    self.last[0],
                    self.last[1],
                    text,
                    self.ix.decls[self.last].loc,
                    self.ix.decls[self.last].kind,
                )
            return
        # Only the comment run directly above a declaration documents it; adjacent
        # comment lines form one block.
        if self.pending and start <= self.pending_end + 1:
            self.pending[-1] += "\n" + text
        else:
            if self.pending and not self.has_code:
                self.pending = []
            self.pending.append(text)
        self.pending_end = end

    def take_pending(self) -> str | None:
        if not self.pending:
            return None
        blocks = self.pending
        self.pending = []
        kept = []
        for text in blocks:
            named = re.search(r"^\s*[\\@](?:class|struct)\s+(\w+)", text, re.M)
            if named:
                self.ix.named_class_docs.setdefault(named.group(1), text)
            if re.search(
                r"^\s*[\\@](file|defgroup|name|addtogroup|page|mainpage)\b|^\s*@[{}]\s*$",
                text,
                re.M,
            ) and not re.search(r"[\\@](param|returns?)\b", text):
                continue
            kept.append(text)
        return "\n\n".join(kept) or None

    def clean_stmt(self) -> str:
        text = " ".join("".join(self.stmt).split())
        self.stmt = []
        self.has_code = False
        text = _ACCESS.sub("", text)
        text = _API_MACRO.sub("", text)
        text = re.sub(r"\[\[[^\]]*\]\]", "", text)
        prev = None
        while prev != text:
            prev = text
            text = _ACCESS.sub("", text)
            m = _LEADING_MACRO.match(text)
            if m and not re.match(r"\s*[A-Z][A-Z0-9_]*\s*\(", text[m.end() :] or "x"):
                text = text[m.end() :]
            elif m and m.end() == len(text):
                text = ""
        return text.strip()

    def loc(self) -> str:
        return f"{self.rel}:{self.stmt_line or self.line}"

    def register(self, scope: str, name: str, kind: str, doc: str | None) -> None:
        self.ix.add(scope, name, doc, self.loc(), kind)
        self.last = (scope, name)

    def decl_name(self, text: str) -> tuple[str, str] | None:
        """(kind, name) of a declaration statement, or None."""
        if not text or text.startswith(("friend ", "return ", "static_assert", "using namespace")):
            return None
        text = re.sub(r"^template\s*<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>\s*", "", text)
        m = re.match(r"^using\s+(\w+)\s*=", text)
        if m:
            return "typedef", m.group(1)
        if text.startswith("typedef "):
            m = re.search(r"(\w+)\s*(\[[^\]]*\])?\s*$", text) or re.search(
                r"\(\s*\*\s*(\w+)\s*\)", text
            )
            return ("typedef", m.group(1)) if m else None
        # first '(' outside template brackets
        depth = 0
        paren = -1
        eq = -1
        for k, ch in enumerate(text):
            if ch == "<":
                depth += 1
            elif ch == ">" and depth:
                depth -= 1
            elif ch == "(" and depth == 0:
                paren = k
                break
            elif (
                ch == "="
                and depth == 0
                and eq < 0
                and not text.startswith("operator", max(0, k - 8))
            ):
                eq = k
        if paren >= 0 and (eq < 0 or eq > paren):
            head = text[:paren].rstrip()
            m = re.search(r"(operator\s*(?:\(\)|[^\w\s(]+|\w[\w\s:<>*&]*?))$|(~?\w+)$", head)
            if not m:
                return None
            name = (m.group(1) or m.group(2)).replace(" ", "")
            if name in ("if", "for", "while", "switch", "sizeof", "decltype", "alignas"):
                return None
            return "function", name
        head = text[:eq] if eq >= 0 else text
        head = re.sub(r"\[[^\]]*\]\s*$", "", head.strip())
        head = re.sub(r"\{.*$", "", head).strip()
        m = re.search(r"(\w+)\s*$", head)
        if not m or m.group(1) in ("public", "private", "protected", "const", "default", "delete"):
            return None
        if not re.search(r"\w[\w:<>,*&\s]*\s+[*&]*\w+\s*$", head) and eq < 0:
            return None
        return "variable", m.group(1)

    def open_brace(self) -> None:
        text = self.clean_stmt()
        if self.in_body():
            self.scopes.append(("body", ""))
            return
        doc = self.take_pending()
        m = re.match(
            r"^(?:template\s*<.*>\s*)?(class|struct|union)\s+(?:alignas\([^)]*\)\s*)?(\w+)(.*)$",
            text,
        )
        if m and "(" not in m.group(3).split(":")[0]:
            name = m.group(2)
            self.register(self.scope(), name, "class", doc)
            bases = re.findall(
                r"(?:public|protected|private|virtual|,|:)\s*(?:virtual\s+)?(?:public\s+|protected\s+|private\s+)?([\w:]+)(?:\s*<[^{]*?>)?",
                m.group(3),
            )
            for base in bases:
                base = base.split("::")[-1]
                if (
                    base
                    and base not in ("public", "protected", "private", "virtual", "final")
                    and base != name
                ):
                    if base not in self.ix.bases[name]:
                        self.ix.bases[name].append(base)
            self.scopes.append(("class", name))
            return
        m = re.match(r"^(?:typedef\s+)?enum\s+(class|struct)?\s*(\w+)?", text)
        if m:
            name = m.group(2) or ""
            if name:
                self.register(self.scope(), name, "enum", doc)
            self.scopes.append(("enumclass_body" if m.group(1) else "enum", name))
            return
        if re.match(r"^(?:inline\s+)?namespace\b", text) or text.startswith('extern ""'):
            self.scopes.append(("ns", ""))
            return
        if text:
            d = self.decl_name(text)
            if d and d[0] == "function":
                self.register(self.scope(), d[1], "function", doc)
            elif d and d[0] == "variable":
                self.register(self.scope(), d[1], "variable", doc)
        self.scopes.append(("body", ""))

    def close_brace(self) -> None:
        if self.scopes:
            kind, _name = self.scopes[-1]
            if kind in ("enum", "enumclass_body"):
                self.enumerator()
            self.scopes.pop()
        self.stmt = []
        self.has_code = False
        if not self.in_body():
            self.pending = [] if not self.scopes or self.scopes[-1][0] != "class" else self.pending

    def enumerator(self, doc: str | None = None) -> None:
        text = self.clean_stmt()
        m = re.match(r"^(\w+)", text)
        if not m:
            if doc and self.last:
                self.ix.add(*self.last, doc, self.ix.decls[self.last].loc, "enumerator")
            return
        name = m.group(1)
        kind, enum_name = self.scopes[-1]
        own = self.take_pending()
        docs = "\n".join(d for d in (own, doc) if d) or None
        scope = enum_name if kind == "enumclass_body" else self.scope()
        self.register(scope, name, "enumerator", docs)

    def end_stmt(self) -> None:
        text = self.clean_stmt()
        if self.in_body():
            return
        if self.scopes and self.scopes[-1][0] in ("enum", "enumclass_body"):
            return
        if not text:
            return
        if re.match(r"^(class|struct|union|enum)\s+\w+\s*$", text):
            return  # forward declaration keeps a following doc's attachment
        doc = self.take_pending()
        d = self.decl_name(text)
        if d:
            self.register(self.scope(), d[1], d[0], doc)


# ----------------------------------------------------------------------------- doxygen text

_INLINE_CMDS = r"p|a|e|em|c|b|arg|ref|link"


def clean_doxygen(raw: str) -> str:
    """Doxygen comment text -> readable plain text (paragraphs, lists, code blocks)."""
    text = raw.replace("\r", "")
    text = re.sub(
        r"^\s*[\\@](class|struct|ingroup|file|anchor|addtogroup|defgroup|weakgroup|internal|nosubgrouping|hideinitializer|showinitializer)\b[^\n]*$",
        "",
        text,
        flags=re.M,
    )
    text = re.sub(r"^\s*@[{}]\s*$", "", text, flags=re.M)
    text = re.sub(r"[\\@]overload\b", "", text)
    text = re.sub(r"[\\@]brief\b\s*", "", text)
    text = re.sub(r"[\\@](?:sub)*section\s+\w+\s*", "", text)
    text = re.sub(r"[\\@]ref\s+([\w:.()~]+)\s+\"([^\"]+)\"", r"\2", text)
    text = re.sub(r"[\\@](?:ref|link)\s+([\w:.()~]+)", r"\1", text)
    text = re.sub(r"[\\@]endlink\b", "", text)
    text = re.sub(
        r"[\\@](?:p|a|e|em|c|b|arg)\s+([\w:.()<>&*,\[\]-]+?)(?=[\s,;)!?]|\.(?:\s|$)|$)", r"\1", text
    )
    text = re.sub(
        r"</?(?:b|i|em|tt|code|strong|br|p|ul|ol|li|sup|sub|center|pre)\s*/?>", "", text, flags=re.I
    )
    text = (
        text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("&quot;", '"')
    )
    text = re.sub(r"[\\@]param\s*(?:\[[\w,]+\])?\s+(\w+)\s*", r"\n- \1: ", text)
    text = re.sub(r"[\\@]tparam\s+(\w+)\s*", r"\n- \1 (template): ", text)
    text = re.sub(r"[\\@](?:returns?|retval)\b\s*", "\nReturns: ", text)
    for cmd, label in (
        ("sa", "See also"),
        ("see", "See also"),
        ("note", "Note"),
        ("warning", "Warning"),
        ("deprecated", "Deprecated"),
        ("pre", "Precondition"),
        ("post", "Postcondition"),
        ("throws?", "Raises"),
        ("exception", "Raises"),
        ("todo", "Todo"),
        ("bug", "Bug"),
        ("remarks?", "Note"),
        ("attention", "Note"),
        ("details", ""),
        ("par", ""),
    ):
        text = re.sub(rf"[\\@]{cmd}\b\s*", f"\n{label}: " if label else "\n", text)
    text = re.sub(r"[\\@]li\b\s*", "\n- ", text)
    text = re.sub(
        r"[\\@](?:snippet|include|dontinclude|image|copydoc|copydetails|copybrief|htmlonly|endhtmlonly|verbatim|endverbatim)\b[^\n]*",
        "",
        text,
    )
    # Code blocks.
    out: list[str] = []
    in_code = False
    para: list[str] = []

    def flush() -> None:
        if para:
            out.append(" ".join(para))
            para.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if re.match(r"^[\\@]code\b|^~~~|^```", stripped):
            flush()
            in_code = not in_code if not stripped.startswith(("\\code", "@code")) else True
            continue
        if re.match(r"^[\\@]endcode\b", stripped):
            in_code = False
            continue
        if in_code:
            out.append(
                "    " + line.rstrip()[min(len(line) - len(line.lstrip()), 1) :]
                if line.strip()
                else ""
            )
            continue
        if not stripped:
            flush()
            out.append("")
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            # Markdown table row (schema attribute tables): "key: value".
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if all(not cell or set(cell) <= set("-: ") for cell in cells):
                continue
            flush()
            out.append(": ".join(cell for cell in cells if cell))
            continue
        if re.match(
            r"^(- |\* |\d+\. |Returns: |See also: |Note: |Warning: |Deprecated: |Raises: |Precondition: |Postcondition: |Todo: )",
            stripped,
        ):
            flush()
            para.append(re.sub(r"^\* ", "- ", stripped))
            continue
        para.append(stripped)
    flush()
    text = "\n".join(out)
    text = re.sub(r"[\\@](\w+)", r"\1", text)  # any remaining command: keep its word
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text


def first_sentence(text: str, limit: int = 160) -> str:
    para = text.split("\n\n", 1)[0].split("\n", 1)[0].strip()
    m = re.match(r"(.+?[.!?])(\s|$)", para)
    s = m.group(1) if m and len(m.group(1)) >= 12 else para
    if len(s) > limit:
        s = s[: limit - 1].rsplit(" ", 1)[0] + "…"
    return s


# ----------------------------------------------------------------------------- signatures


class _NoMatch:
    """Never converts to a C++ argument type."""


_NAMESPACE = re.compile(
    r"\bpxrInternal_v\w+__pxrReserved__::|\bpxr::|\b(?:pxr_)?boost::python::(?:api::)?|\bstd::(?:__\w+::)?"
)


def boost_signatures(func) -> list[str]:
    """C++ overload signatures Boost.Python reports for ``func`` (no call reaches C++)."""
    try:
        func(*([_NoMatch()] * 13))
    except BaseException as error:  # noqa: BLE001
        msg = str(error)
        if "did not match C++ signature" not in msg:
            return []
        lines = msg.split("did not match C++ signature:", 1)[1].strip().splitlines()
        return [line.strip() for line in lines if line.strip()]
    return []


_PYTYPE_CACHE: dict[str, str] = {}


def py_type(cpp: str, cpp_to_py: dict[str, str]) -> str:
    t = _NAMESPACE.sub("", cpp).strip()
    t = re.sub(r"\b(const|volatile|struct|class|typename)\b", "", t)
    t = re.sub(r"\{lvalue\}", "", t)
    t = t.replace("&", "").replace("*", "").strip()
    t = " ".join(t.split())
    simple = {
        "TfToken": "str",
        "string": "str",
        "basic_string<char, char_traits<char>, allocator<char> >": "str",
        "char": "str",
        "bool": "bool",
        "double": "float",
        "float": "float",
        "GfHalf": "float",
        "half": "float",
        "int": "int",
        "unsigned int": "int",
        "long": "int",
        "unsigned long": "int",
        "long long": "int",
        "unsigned long long": "int",
        "size_t": "int",
        "unsigned char": "int",
        "short": "int",
        "unsigned short": "int",
        "int64_t": "int",
        "uint64_t": "int",
        "uint32_t": "int",
        "int32_t": "int",
        "object": "object",
        "list": "list",
        "dict": "dict",
        "tuple": "tuple",
        "void": "None",
        "_object": "object",
        "SdfPath": "Sdf.Path",
        "VtValue": "object",
        "VtDictionary": "dict",
        "UsdTimeCode": "Usd.TimeCode",
        "SdfAssetPath": "Sdf.AssetPath",
        "TfType": "Tf.Type",
        "TfTokenVector": "list[str]",
        "SdfPathVector": "list[Sdf.Path]",
    }
    if t in simple:
        return simple[t]
    m = re.match(
        r"^(?:TfRefPtr|TfWeakPtr|TfDeclarePtrs<\w+>::\w+|shared_ptr|unique_ptr)<\s*(\w+)\s*>$", t
    )
    if m:
        return py_type(m.group(1), cpp_to_py)
    m = re.match(r"^(\w+?)(RefPtr|Ptr|Handle|ConstPtr|ConstHandle)$", t)
    if m and m.group(1) in cpp_to_py:
        return cpp_to_py[m.group(1)]
    m = re.match(r"^VtArray<\s*(\w+)\s*>$", t)
    if m:
        inner = m.group(1)
        name = {
            "int": "Int",
            "float": "Float",
            "double": "Double",
            "TfToken": "Token",
            "string": "String",
            "bool": "Bool",
            "unsigned int": "UInt",
            "GfHalf": "Half",
            "long": "Int64",
            "unsigned char": "UChar",
        }.get(inner)
        if name is None and inner.startswith("Gf"):
            name = inner[2:]
        if name is None and inner in cpp_to_py:
            module, _, leaf = cpp_to_py[inner].partition(".")
            return f"{module}.{leaf}Array"
        if name:
            return f"Vt.{name}Array"
    m = re.match(r"^(?:vector|set|list)<\s*(.+?)(?:\s*,\s*allocator<.*>)?\s*>$", t)
    if m:
        return f"list[{py_type(m.group(1), cpp_to_py)}]"
    if re.match(r"^function<", t):
        return "Callable"
    m = re.match(r"^(?:optional)<\s*(.+?)\s*>$", t)
    if m:
        return f"{py_type(m.group(1), cpp_to_py)} | None"
    if t in cpp_to_py:
        return cpp_to_py[t]
    if "::" in t:
        head, tail = t.rsplit("::", 1)
        if head in cpp_to_py:
            return f"{cpp_to_py[head]}.{tail}"
    return t


def _split_args(text: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def py_signature(
    cpp_sig: str, cpp_to_py: dict[str, str], owner_cpp: str | None
) -> tuple[str, bool]:
    """('(self, a: int, b: str = "")', is_instance_method) from one Boost signature line."""
    m = re.match(r"^(\w+)\((.*)\)(?:\s*->\s*(.+))?$", cpp_sig)
    if not m:
        return "", False
    args = []
    instance = False
    ret = m.group(3)
    for k, arg in enumerate(_split_args(m.group(2))):
        arg = arg.strip()
        default = None
        if "=" in arg:
            arg, default = arg.split("=", 1)
            arg, default = arg.strip(), default.strip()
            default = _NAMESPACE.sub("", default)
        lvalue = "{lvalue}" in arg
        am = re.match(r"^(.*?)(?:\s+(\w+))?$", arg.replace("{lvalue}", "").strip())
        ctype, name = (am.group(1), am.group(2)) if am else (arg, None)
        if name and not re.search(r"[\w>]\s+\w+$", arg.replace("{lvalue}", "").strip()):
            name = None
        if (
            k == 0
            and (
                lvalue or (owner_cpp and _NAMESPACE.sub("", ctype).strip(" &*").endswith(owner_cpp))
            )
            and not name
        ):
            args.append("self")
            instance = True
            continue
        if (
            k == 0
            and ctype.strip() in ("_object*", "_object *", "PyObject*")
            and m.group(1) == "__init__"
        ):
            args.append("self")
            instance = True
            continue
        ptype = py_type(ctype, cpp_to_py)
        if not name:
            name = f"arg{k}"
        piece = f"{name}: {ptype}"
        if default is not None:
            piece += f" = {default}"
        args.append(piece)
    sig = "(" + ", ".join(args) + ")"
    if ret:
        sig += " -> " + py_type(ret, cpp_to_py)
    return sig, instance


# ----------------------------------------------------------------------------- wrap files


def parse_wraps(pxr_root: Path) -> tuple[dict, dict, dict]:
    """class_<Cpp>("Py") maps, .def("Py", &Cls::Meth) maps and .def("Py", FreeFunc)."""
    classes: dict[tuple[str, str], str] = {}
    methods: dict[tuple[str, str], tuple[str, str]] = {}
    funcs: dict[tuple[str, str], str] = {}
    for path in pxr_root.rglob("wrap*.cpp"):
        lib = path.parent.name
        src = path.read_text(errors="ignore")
        typedefs = {
            m.group(2): m.group(1).split("::")[-1]
            for m in re.finditer(r"typedef\s+([\w:]+)\s+(\w+)\s*;", src)
        }
        for m in re.finditer(r"using\s+(\w+)\s*=\s*([\w:]+)\s*;", src):
            typedefs[m.group(1)] = m.group(2).split("::")[-1]
        for m in re.finditer(r"class_\s*<\s*([\w:]+)[^\"]{0,300}?\(\s*\"(\w+)\"", src):
            cpp = m.group(1).split("::")[-1]
            cpp = typedefs.get(cpp, cpp)
            if cpp not in ("This", "Type"):
                classes.setdefault((lib, m.group(2)), cpp)
        for m in re.finditer(
            r"\.def\(\s*\"(\w+)\"\s*,\s*(?:\([^)]*\)\s*)?&?\s*([\w:]+)::(~?\w+)", src
        ):
            cls = m.group(2).split("::")[-1]
            methods.setdefault((lib, m.group(1)), (typedefs.get(cls, cls), m.group(3)))
        for m in re.finditer(
            r"(?<![\w.])def\(\s*\"(\w+)\"\s*,\s*(?:\([^)]*\)\s*)?&?\s*([\w:]+)", src
        ):
            funcs.setdefault((lib, m.group(1)), m.group(2).split("::")[-1])
        for m in re.finditer(
            r"\.add_(?:static_)?property\(\s*\"(\w+)\"\s*,\s*(?:\w+\s*\(\s*)?&?\s*([\w:]+)::(\w+)",
            src,
        ):
            cls = m.group(2).split("::")[-1]
            methods.setdefault((lib, m.group(1)), (typedefs.get(cls, cls), m.group(3)))
    return classes, methods, funcs


# ----------------------------------------------------------------------------- build


def lc_first(name: str) -> str:
    return name[:1].lower() + name[1:]


def iter_pxr_modules():
    import pxr

    for info in sorted(pkgutil.iter_modules(pxr.__path__), key=lambda m: m.name):
        if info.name.startswith("_") or not info.name[:1].isupper():
            continue
        try:
            yield info.name, importlib.import_module(f"pxr.{info.name}")
        except Exception as error:  # noqa: BLE001
            print(f"skip pxr.{info.name}: {error}", file=sys.stderr)


def build(source: Path, version: str, *, max_text: int) -> dict:
    roots = [p for p in [source / "pxr", *source.glob("*/pxr")] if p.is_dir()]
    if not roots:
        raise SystemExit(f"no pxr/ directory under {source}")
    pxr_root = roots[0]
    hix = HeaderIndex()
    for path in sorted(pxr_root.rglob("*.h")):
        rel = path.relative_to(pxr_root).as_posix()
        hix.parse_file(path, rel)
    wrap_classes, wrap_methods, wrap_funcs = parse_wraps(pxr_root)
    libdir = {d.name for d in pxr_root.rglob("*") if d.is_dir()}

    modules = list(iter_pxr_modules())
    # Python class -> C++ class name.
    py_to_cpp: dict[str, str] = {}
    cpp_to_py: dict[str, str] = {}
    classes: list[tuple[str, str, type]] = []

    def cpp_for_class(mod: str, qual: str) -> str | None:
        lib = lc_first(mod)
        leaf = qual.rsplit(".", 1)[-1]
        outer = py_to_cpp.get(f"{mod}.{qual.rsplit('.', 1)[0]}") if "." in qual else None
        cands = [
            f"{mod}{leaf}",
            wrap_classes.get((lib, leaf)),
            f"{mod}{leaf}Type",
            leaf,
            f"{mod}_{leaf}",
        ]
        if outer:
            cands.insert(0, leaf)
        for cand in cands:
            if not cand:
                continue
            decl = hix.decls.get((outer or "", cand)) or hix.decls.get(("", cand))
            if (
                decl is not None and decl.kind in ("class", "enum", "typedef")
            ) or cand in hix.bases:
                return cand
        # nested scopes (UsdStage::InitialLoadSet)
        if outer and (outer, leaf) in hix.decls:
            return f"{outer}::{leaf}"
        return None

    def walk_classes(mod: str, owner, prefix: str, depth: int = 0) -> None:
        for name, value in sorted(vars(owner).items()):
            if not isinstance(value, type) or name.startswith("_") or "_Test" in name:
                continue
            if getattr(value, "__module__", "") != f"pxr.{mod}":
                continue
            qual = f"{prefix}{name}"
            if f"{mod}.{qual}" in py_to_cpp:
                continue
            cpp = cpp_for_class(mod, qual)
            py_to_cpp[f"{mod}.{qual}"] = cpp or ""
            if cpp:
                cpp_to_py.setdefault(
                    cpp.split("::")[-1] if "::" not in cpp else cpp, f"{mod}.{qual}"
                )
            classes.append((mod, qual, value))
            if depth < 2:
                walk_classes(mod, value, qual + ".", depth + 1)

    for mod, module in modules:
        walk_classes(mod, module, "")

    def class_doc(cpp: str | None) -> tuple[str, str]:
        if not cpp:
            return "", ""
        scope, _, leaf = cpp.rpartition("::")
        decl = hix.decls.get((scope, leaf))
        docs = list(decl.docs) if decl else []
        if leaf in hix.named_class_docs and hix.named_class_docs[leaf] not in docs:
            docs.insert(0, hix.named_class_docs[leaf])
        return "\n\n".join(docs), (decl.loc if decl else "")

    def member_doc(cpp_cls: str | None, names: list[str]) -> tuple[str, str, str]:
        """(raw doc, loc, declaring class) for a member, searching base classes."""
        if not cpp_cls:
            return "", "", ""
        seen: set[str] = set()
        todo = [cpp_cls]
        first_loc = ("", "")
        while todo:
            cls = todo.pop(0)
            if cls in seen:
                continue
            seen.add(cls)
            for name in names:
                decl = hix.decls.get((cls, name))
                if decl is not None:
                    if decl.docs:
                        return "\n\n".join(decl.docs[:3]), decl.loc, cls
                    first_loc = first_loc if first_loc[0] else (decl.loc, cls)
            todo.extend(hix.bases.get(cls.split("::")[-1], []))
        return "", first_loc[0], first_loc[1]

    entries: list[list] = []
    seen_names: set[str] = set()
    stats = collections.Counter()

    def to_py_names(text: str) -> str:
        def repl(m: re.Match) -> str:
            cpp = m.group(1)
            rest = m.group(2) or ""
            py = cpp_to_py.get(cpp)
            if not py:
                return m.group(0)
            return py + rest.replace("::", ".")

        text = re.sub(
            r"\b([A-Z]\w*Tokens)->(\w+)",
            lambda m: f"{cpp_to_py.get(m.group(1) + 'Type', m.group(1))}.{m.group(2)}",
            text,
        )
        return re.sub(r"\b([A-Z][A-Za-z]+[A-Z]\w*)((?:::~?\w+)*)", repl, text)

    def add(name, kind, sig, raw, loc, docstring=None):
        if name in seen_names:
            return
        seen_names.add(name)
        pending.append([name, kind, sig, raw, loc, docstring])

    pending: list[list] = []

    def finish(name, kind, sig, raw, loc, docstring):
        text = to_py_names(clean_doxygen(raw)) if raw else ""
        if not text and docstring:
            text = inspect.cleandoc(docstring)
        if len(text) > max_text:
            cut = text[:max_text].rsplit("\n", 1)[0] if "\n" in text[:max_text] else text[:max_text]
            text = cut.rstrip() + "\n[…truncated; see the OpenUSD C++ docs]"
        summary = first_sentence(text) if text else ""
        if re.fullmatch(r'"[^"\n]*"', summary) and "\n\n" in text:
            # Token docs open with the literal ("face"); summarize the description.
            summary = first_sentence(text.split("\n\n", 1)[1])
        stats[kind + (":doc" if text else ":nodoc")] += 1
        # text is the whole description (it starts with the summary sentence).
        entries.append([name, kind, sig or "", summary, text if text != summary else "", loc])

    def sigs_for(func, owner_cpp: str | None) -> tuple[str, bool | None]:
        sigs = boost_signatures(func)
        if not sigs:
            try:
                return str(inspect.signature(func)), None
            except (TypeError, ValueError):
                return "", None
        out = []
        instance = None
        for line in sigs[:4]:
            sig, inst = py_signature(line, cpp_to_py, owner_cpp)
            if sig and sig not in out:
                out.append(sig)
                instance = inst if instance is None else instance
        if len(sigs) > 4:
            out.append(f"... {len(sigs) - 4} more overloads")
        return " | ".join(out), instance

    for mod, module in modules:
        lib = lc_first(mod)
        mdoc = (module.__doc__ or "").strip()
        add(mod, "module", "", "", f"{lib}/" if lib in libdir else "", mdoc)
        names = sorted(n for n in dir(module) if not n.startswith("_"))
        for name in names:
            try:
                value = getattr(module, name)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(value, type) or inspect.ismodule(value):
                continue
            vmod = getattr(value, "__module__", None) or ""
            if callable(value):
                if not (
                    vmod.startswith(f"pxr.{mod}")
                    or type(value).__name__ == "function"
                    and vmod.startswith("pxr")
                ):
                    continue
                cands = [wrap_funcs.get((lib, name)), f"{mod}{name}", name, f"{mod}_{name}"]
                raw, loc = "", ""
                for cand in cands:
                    if not cand:
                        continue
                    decl = hix.decls.get(("", cand))
                    if decl is not None and (decl.docs or not loc):
                        raw, loc = "\n\n".join(decl.docs[:3]), decl.loc
                        if raw:
                            break
                sig, _ = sigs_for(value, None)
                for cand in cands:
                    if cand and cand != name and ("", cand) in hix.decls:
                        cpp_to_py.setdefault(cand, f"{mod}.{name}")
                add(f"{mod}.{name}", "function", sig, raw, loc, getattr(value, "__doc__", None))
            elif isinstance(value, (str, int, float)) and type(value).__module__ == "builtins":
                continue
        for _mod, qual, cls in [c for c in classes if c[0] == mod]:
            cpp = py_to_cpp.get(f"{mod}.{qual}")
            raw, loc = class_doc(cpp)
            init_sig = ""
            if "__init__" in vars(cls):
                init_sig, _ = sigs_for(vars(cls)["__init__"], cpp.split("::")[-1] if cpp else None)
                init_sig = init_sig.replace("(self, ", "(").replace("(self)", "()")
            own_doc = cls.__doc__ if cls.__doc__ and cls.__doc__ != type.__doc__ else None
            kind = (
                "enum"
                if any(b.__name__ == "Enum" and b.__module__ == "pxr.Tf" for b in cls.__mro__[1:])
                else "class"
            )
            add(f"{mod}.{qual}", kind, init_sig, raw, loc, own_doc)
            cpp_leaf = cpp.split("::")[-1] if cpp else None
            scope_cpp = (
                cpp if cpp and "::" not in cpp else (cpp.replace("::", "::") if cpp else None)
            )
            for mname, mval in sorted(vars(cls).items()):
                if mname.startswith("_") or isinstance(mval, type):
                    continue
                full = f"{mod}.{qual}.{mname}"
                func = mval.__func__ if isinstance(mval, (staticmethod, classmethod)) else mval
                wm = wrap_methods.get((lib, mname))
                cands = [mname]
                if wm and wm[1] != mname:
                    cands.append(wm[1])
                lookup_cls = scope_cpp
                if type(func).__name__ == "StaticProperty":
                    # Class-level constant (Tokens, Sdf.ValueTypeNames, Sdf.Path.emptyPath).
                    try:
                        cval = getattr(cls, mname)
                    except Exception:  # noqa: BLE001
                        cval = None
                    if cval is not None and not callable(cval):
                        names = [mname, mname.rstrip("_"), f"Get{mname[:1].upper()}{mname[1:]}"]
                        raw, loc, _decl = member_doc(lookup_cls, names)
                        if isinstance(cval, str):
                            kind, sig = "token", f" = {cval!r}" if len(cval) < 60 else ""
                        else:
                            kind = "constant"
                            sig = (
                                ": "
                                + type(cval).__module__.removeprefix("pxr.")
                                + "."
                                + type(cval).__qualname__
                            )
                        add(full, kind, sig, raw, loc)
                        continue
                if isinstance(func, property) or type(func).__name__ in (
                    "property",
                    "StaticProperty",
                ):
                    getter = getattr(func, "fget", None)
                    if wm:
                        cands = [wm[1], mname]
                    else:
                        cands += [
                            f"Get{mname[:1].upper()}{mname[1:]}",
                            mname[:1].upper() + mname[1:],
                        ]
                    raw, loc, _decl = member_doc(lookup_cls, cands)
                    sig = ""
                    if getter is not None:
                        gs, _ = sigs_for(getter, cpp_leaf)
                        rm = re.search(r"->\s*(.+)$", gs.split(" | ")[0]) if gs else None
                        sig = f": {rm.group(1)}" if rm else ""
                    add(full, "property", sig, raw, loc, getattr(func, "__doc__", None))
                    continue
                if callable(func):
                    raw, loc, _decl = member_doc(lookup_cls, cands)
                    sig, instance = sigs_for(func, cpp_leaf)
                    kind = (
                        "method"
                        if (
                            instance
                            or isinstance(mval, (staticmethod, classmethod)) is False
                            and instance is None
                        )
                        else "staticmethod"
                    )
                    if isinstance(mval, staticmethod):
                        kind = "staticmethod"
                    add(
                        full,
                        kind,
                        sig,
                        raw,
                        loc,
                        getattr(func, "__doc__", None) if inspect.isfunction(func) else None,
                    )
                    continue
                if isinstance(mval, (str, int, float)) or type(mval).__module__.startswith("pxr"):
                    names = [mname, mname.rstrip("_")]
                    raw, loc, _decl = member_doc(lookup_cls, names)
                    if not raw and cpp and cpp.endswith("Type"):
                        pass
                    value_repr = repr(mval) if isinstance(mval, (str, int, float)) else ""
                    kind = "token" if isinstance(mval, str) else "constant"
                    sig = f" = {value_repr}" if value_repr and len(value_repr) < 60 else ""
                    add(full, kind, sig, raw, loc)

    for item in pending:
        finish(*item)

    from pxr import Usd

    return {
        "format": 1,
        "package": "pxr",
        "version": version,
        "usd_version": ".".join(str(v) for v in Usd.GetVersion()),
        "source": f"OpenUSD {source.name}: C++ header comments (pxr/**/*.h) and wrap*.cpp",
        "columns": ["name", "kind", "signature", "summary", "text", "source"],
        "stats": dict(sorted(stats.items())),
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source", required=True, type=Path, help="OpenUSD source dir (contains pxr/)"
    )
    parser.add_argument("--version", required=True, help="OpenUSD version label, e.g. 26.08")
    parser.add_argument("--out", required=True, type=Path, help="output .json.gz")
    parser.add_argument(
        "--max-text", type=int, default=1500, help="cap on each entry's text (chars)"
    )
    parser.add_argument("--allow-version-mismatch", action="store_true")
    args = parser.parse_args(argv)
    from pxr import Usd

    installed = Usd.GetVersion()
    major, _, minor = args.version.partition(".")
    if (installed[1], installed[2]) != (
        int(major),
        int(minor or 0),
    ) and not args.allow_version_mismatch:
        raise SystemExit(
            f"installed pxr is {installed}, sources are {args.version}: pass --allow-version-mismatch"
        )
    data = build(args.source, args.version, max_text=args.max_text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode()
    args.out.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    print(
        f"{args.out}: {len(data['entries'])} entries, {len(payload)} bytes json, "
        f"{args.out.stat().st_size} bytes gz",
        file=sys.stderr,
    )
    print(json.dumps(data["stats"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
