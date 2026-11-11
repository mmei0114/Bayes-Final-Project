"""Read-only server probe; run with the user's chosen remote base interpreter."""
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

packages = {}
for name in ('numpy', 'scipy', 'matplotlib'):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
limits = {}
for name in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes',
             '/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpu/cpu.cfs_quota_us'):
    if Path(name).exists():
        limits[name] = Path(name).read_text().strip()
print(json.dumps(dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                      cpu_affinity=len(os.sched_getaffinity(0)), packages=packages,
                      resource_limits=limits), indent=2))
