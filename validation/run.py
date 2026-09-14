"""Validation entry; defaults to planning, requires --execute-npu for device checks."""
import sys
from performance.run import main


if __name__ == '__main__':
    raise SystemExit(main([*sys.argv[1:], '--validate-only']))
