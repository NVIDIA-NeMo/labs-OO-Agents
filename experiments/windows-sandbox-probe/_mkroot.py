"""Build a private runtime root: interpreter + stdlib + DLLs + site-packages.

The base interpreter directory denies DACL writes (Errno 5), so the container
cannot be granted access to it in place. A user-owned copy is the only way to
make the interpreter reachable from inside the AppContainer.
"""
import os, shutil, sys, time

SRC_BASE = r"D:\dev\uv\python\cpython-3.13-windows-x86_64-none"
SRC_SP = r"E:\labs-OO-Agents\.venv\Lib\site-packages"
DST = r"E:\labs-OO-Agents\experiments\windows-sandbox-probe\.rt"

if os.path.exists(DST):
    shutil.rmtree(DST)
t0 = time.perf_counter()
os.makedirs(DST, exist_ok=True)

# Top-level files of the base install (python.exe, python313.dll, ...) and Lib/.
for name in os.listdir(SRC_BASE):
    src = os.path.join(SRC_BASE, name)
    dst = os.path.join(DST, name)
    if os.path.isdir(src):
        if name in ("Lib", "DLLs", "libs", "include"):
            shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns(
                "__pycache__", "test", "tests", "idlelib", "tkinter", "turtledemo"))
    else:
        shutil.copy2(src, dst)

# site-packages: only what an import probe needs; skip caches and huge trees.
os.makedirs(os.path.join(DST, "Lib", "site-packages"), exist_ok=True)
shutil.copytree(SRC_SP, os.path.join(DST, "Lib", "site-packages"),
                symlinks=True, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

elapsed = time.perf_counter() - t0
total = 0
for root, dirs, files in os.walk(DST):
    for f in files:
        try:
            total += os.path.getsize(os.path.join(root, f))
        except OSError:
            pass
print(f"root={DST}")
print(f"files={sum(len(f) for _,_,f in os.walk(DST))} bytes={total/1e6:.1f}MB build_s={elapsed:.2f}")
print("python.exe:", os.path.exists(os.path.join(DST, "python.exe")))
