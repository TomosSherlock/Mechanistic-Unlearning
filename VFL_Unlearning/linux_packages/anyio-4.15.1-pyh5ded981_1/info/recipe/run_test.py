import sys
import platform
from subprocess import call


FAIL_UNDER = 0

if platform.system() == "Linux":
    FAIL_UNDER = 84

COV = ["coverage"]
RUN = ["run", "--source=anyio", "--branch", "-m"]
PYTEST = ["pytest", "-vv", "--color=yes", "--tb=long"]
REPORT = ["report", "--show-missing", "--skip-covered", f"--fail-under={FAIL_UNDER}"]

SKIPS = [
    "bind_link_local",
    "block_device",
    "connection_refused",
    "happy_eyeballs",
    "ipv6",
    # https://github.com/conda-forge/anyio-feedstock/pull/78
    "tcp_listener_retry_after_partial_failure",
    "tcp_listener_dualstack_disabled",
    "tcp_listener_same_port",
    # https://github.com/conda-forge/anyio-feedstock/pull/88
    "sourceless_install",
]

SKIP_OR = " or ".join(SKIPS)
K = ["-k", f"not ({SKIP_OR})"]


if __name__ == "__main__":
    sys.exit(
        # run the tests
        call([*COV, *RUN, *PYTEST, *K])
        # maybe run coverage
        or call([*COV, *REPORT])
    )
