

set -ex



unset PIP_NO_INDEX
python -m pip install --target alt-site backports.tarfile
export PYTHONPATH="${SP_DIR}:alt-site"
python -c "import backports.tarfile"
exit 0
