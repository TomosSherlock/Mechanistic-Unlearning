

set -ex



pip check
pytest -m "not new_console and not wheel and not large" -k "not test_process_teardown"
exit 0
