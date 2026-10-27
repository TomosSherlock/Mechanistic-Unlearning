

set -ex



pip check
python -c "from importlib.metadata import version; assert(version('pyzmq')=='27.1.0')"
pytest tests -k "not test_process_teardown"  --ignore=tests/test_log.py
exit 0
