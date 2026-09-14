"""Provenance collection without initializing a device."""
import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from ops.host.dispatch import ROOT, PINNED_COMMIT, verify_dependency


def manifest():
    versions = {}
    for package in ('torch', 'torch-npu', 'catlass'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                              capture_output=True, text=True)
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
            'python': platform.python_version(), 'platform': platform.platform(),
            'packages': versions, 'project_commit': revision.stdout.strip() or None,
            'catlass_commit': PINNED_COMMIT, 'donor_sha256': verify_dependency(),
            'device_executed': False, 'target': 'Ascend950',
            'limitations': ['NPU compilation/correctness/performance not validated by static checks',
                            'CAST_ROUND is not a promise of baseline RINT tie behavior']}
