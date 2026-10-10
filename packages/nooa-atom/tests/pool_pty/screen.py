# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small 40x120 ANSI screen replay, adapted from the read-only prior audit.

Convenience text projection, not an independent screenshot oracle. Raw ANSI,
wire messages and client logs remain the primary evidence.
"""

import re
import unicodedata


class Screen:
    def __init__(self):
        self.lines = [[" "] * 120 for _ in range(40)]
        self.r = 0
        self.c = 0
        self.pending = ""
        self.unknown = set()
        self.top = 0
        self.bottom = 39

    def feed(self, text):
        self.pending += text
        while self.pending:
            ch = self.pending[0]
            if ch == "\x1b":
                m = re.match(r"\x1b\[([0-9;?:><=]*)([ -/]*)([@-~])", self.pending)
                if m:
                    args, inter, op = m.groups()
                    self.pending = self.pending[m.end() :]
                    if any(x in args for x in "?><="):
                        continue
                    v = [int(x) if x else 0 for x in args.split(";")]
                    n = v[0] or 1
                    if op in "Hf":
                        self.r = (v[0] or 1) - 1
                        self.c = ((v[1] if len(v) > 1 else 1) or 1) - 1
                    elif op == "d":
                        self.r = n - 1
                    elif op == "G":
                        self.c = n - 1
                    elif op == "A":
                        self.r -= n
                    elif op == "B":
                        self.r += n
                    elif op == "C":
                        self.c += n
                    elif op == "D":
                        self.c -= n
                    elif op == "E":
                        self.r += n
                        self.c = 0
                    elif op == "F":
                        self.r -= n
                        self.c = 0
                    elif op == "r":
                        self.top = (v[0] or 1) - 1
                        self.bottom = ((v[1] if len(v) > 1 else 40) or 40) - 1
                        self.r = 0
                        self.c = 0
                    elif op in ("S", "T"):
                        n = min(n, self.bottom - self.top + 1)
                        part = self.lines[self.top : self.bottom + 1]
                        blank = [[" "] * 120 for _ in range(n)]
                        self.lines[self.top : self.bottom + 1] = (
                            (part[n:] + blank) if op == "S" else (blank + part[:-n])
                        )
                    elif op == "J":
                        if v[0] in (2, 3):
                            self.lines = [[" "] * 120 for _ in range(40)]
                        elif v[0] == 0:
                            self.lines[self.r][self.c :] = [" "] * (120 - self.c)
                            for i in range(self.r + 1, 40):
                                self.lines[i] = [" "] * 120
                        else:
                            self.unknown.add("J" + args)
                    elif op == "K":
                        a, b = (
                            (0, 120)
                            if v[0] == 2
                            else ((0, self.c + 1) if v[0] == 1 else (self.c, 120))
                        )
                        self.lines[self.r][a:b] = [" "] * (b - a)
                    elif op == "P":
                        n = min(n, 120 - self.c)
                        self.lines[self.r][self.c :] = self.lines[self.r][self.c + n :] + [" "] * n
                    elif op == "X":
                        self.lines[self.r][self.c : min(120, self.c + n)] = [" "] * min(
                            n, 120 - self.c
                        )
                    elif op not in ("m", "h", "l", "n", "q", "t", "u"):
                        self.unknown.add(op + args)
                    self.r = max(0, min(39, self.r))
                    self.c = max(0, min(119, self.c))
                    continue
                if len(self.pending) < 2:
                    break
                if self.pending[1] in "]P":
                    m = re.search(r"\x07|\x1b\\", self.pending[2:])
                    if not m:
                        break
                    self.pending = self.pending[2 + m.end() :]
                    continue
                if self.pending[1] == "[":
                    break
                if self.pending[1] == "M":
                    if self.r == self.top:
                        self.lines[self.top : self.bottom + 1] = [[" "] * 120] + self.lines[
                            self.top : self.bottom
                        ]
                    else:
                        self.r = max(0, self.r - 1)
                    self.pending = self.pending[2:]
                    continue
                self.unknown.add(repr(self.pending[:2]))
                self.pending = self.pending[2:]
                continue
            self.pending = self.pending[1:]
            if ch == "\r":
                self.c = 0
            elif ch == "\n":
                self.r += 1
                if self.r >= 40:
                    self.lines.pop(0)
                    self.lines.append([" "] * 120)
                    self.r = 39
            elif ch == "\b":
                self.c = max(0, self.c - 1)
            elif ch == "\t":
                self.c = min(119, ((self.c // 8) + 1) * 8)
            elif ord(ch) >= 32:
                if self.c >= 120:
                    self.c = 0
                    self.r += 1
                    if self.r >= 40:
                        self.lines.pop(0)
                        self.lines.append([" "] * 120)
                        self.r = 39
                self.lines[self.r][self.c] = ch
                self.c += 2 if unicodedata.east_asian_width(ch) in "WF" else 1

    def text(self):
        return "\n".join("".join(row).rstrip() for row in self.lines) + "\n"
