

set -ex



pip check
python -m conda info | grep -q "user-agent.*aau/"
exit 0
