"""Read-only resource and package probe in the user's remote base environment."""
import importlib.metadata as md
import json
import os
import platform
import sys
from pathlib import Path

packages = {}
for name in ('numpy', 'scipy', 'matplotlib', 'torch', 'tensorflow', 'keras', 'bayesflow', 'cmdstanpy', 'arviz'):
    try:
        packages[name] = md.version(name)
    except md.PackageNotFoundError:
        packages[name] = None
limits = {}
for name in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes',
             '/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
             '/sys/fs/cgroup/cpu/cpu.cfs_period_us'):
    if Path(name).exists():
        limits[name] = Path(name).read_text().strip()
gpu = None
if packages['torch']:
    import torch
    gpu = dict(cuda_available=torch.cuda.is_available(), compiled_cuda=torch.version.cuda,
               supported_arches=torch.cuda.get_arch_list())
    if torch.cuda.is_available():
        gpu.update(name=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                   total_memory=torch.cuda.get_device_properties(0).total_memory)
tf_gpu = None
if packages['tensorflow']:
    import tensorflow as tf
    tf_gpu = dict(version=tf.__version__, physical_devices=[str(g) for g in tf.config.list_physical_devices('GPU')])
    if tf_gpu['physical_devices']:
        with tf.device('/GPU:0'):
            test=tf.matmul(tf.ones((1024,1024)),tf.ones((1024,1024)))
        tf_gpu.update(matrix_product_device=test.device, matrix_product_value=float(test[0,0]),
                      build_info=tf.sysconfig.get_build_info())
print(json.dumps(dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                      libc=platform.libc_ver(), cpu_affinity=len(os.sched_getaffinity(0)),
                      packages=packages, resource_limits=limits, torch_gpu=gpu, tensorflow_gpu=tf_gpu), indent=2))
