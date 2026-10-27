

set -ex



pip check
python -c "from importlib.metadata import version; assert(version('psutil')=='7.0.0')"
pytest psutil/tests  --ignore=psutil/tests/test_scripts.py -v -k "not (test_against_findmnt or test_comparisons or test_disk_partitions_mocke or test_cmdline or test_name or test_users or test_pmap or test_python2 or test_proc_name or test_against_df or test_all or test_long_name or test_proc_cmdline)"
exit 0
