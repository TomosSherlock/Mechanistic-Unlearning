

set -ex



pip check
python -c "from importlib.metadata import version; assert(version('debugpy')=='1.8.16')"
python -m pytest -ra -vvv -n auto --dist=loadfile --timeout=60 --timeout-method=thread --durations=25 -o pythonpath=gh_src gh_src/tests/debugpy  --ignore=gh_src/tests/debugpy/test_flask.py --ignore=gh_src/tests/debugpy/test_django.py --ignore=gh_src/tests/debugpy/test_gevent.py --ignore=gh_src/tests/debugpy/test_numpy.py -k "not (attach_pid or django or flask or gevent or numpy)"
exit 0
