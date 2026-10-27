

set -ex



conda-token --help
anaconda auth --help
python -c "from anaconda_auth import __version__; assert __version__ == \"0.12.3\""
python -m anaconda_auth._conda.config --verify
exit 0
