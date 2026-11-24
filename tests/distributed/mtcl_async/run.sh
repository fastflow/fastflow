#!/usr/bin/env bash
# Builds and runs the MTCL sender and receiver regression tests.
# Each scenario runs in its own process; build output goes to a temporary
# directory that is removed on exit.
#
# Exit status: 0 all passed, 1 a scenario failed, 2 build or harness error.
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [sender|receiver] [--case NAME] [--control] [--repeat N]
  sender    scenarios: close, overwrite
  receiver  scenarios: frame-size, buffer-capacity (built with AddressSanitizer)
Without a suite, runs both. Without --case, runs every scenario of the suite,
control mode first. --control runs control mode only.
Environment: MTCL_HOME, CEREAL_HOME (searched when unset), CXX, TEST_TIMEOUT (seconds, default 10).
EOF
}

test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$test_dir/../../.." && pwd)

suites=(sender receiver)
selected_case=""
control_only=false
repeat=1
while (($#)); do
    case "$1" in
        sender|receiver) suites=("$1"); shift ;;
        --case)    [[ $# -ge 2 ]] || { usage >&2; exit 2; }; selected_case=$2; shift 2 ;;
        --control) control_only=true; shift ;;
        --repeat)  [[ $# -ge 2 && $2 =~ ^[1-9][0-9]*$ ]] || { usage >&2; exit 2; }; repeat=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

scenarios_of() {
    case "$1" in
        sender)   echo "close overwrite" ;;
        receiver) echo "frame-size buffer-capacity" ;;
    esac
}

if [[ -n $selected_case ]]; then
    [[ ${#suites[@]} -eq 1 && " $(scenarios_of "${suites[0]}") " == *" $selected_case "* ]] || {
        echo "--case needs a suite and one of its scenarios." >&2; usage >&2; exit 2;
    }
fi

find_dependency() {  # find_dependency FILE CANDIDATE... -> first candidate containing FILE
    local file=$1; shift
    for candidate in "$@"; do
        if [[ -f $candidate/$file ]]; then echo "$candidate"; return 0; fi
    done
    return 1
}
MTCL_HOME=${MTCL_HOME:-$(find_dependency include/mtcl.hpp "$repo_dir/ff/distributed/MTCL" \
    "$repo_dir/../FastFlow-Python/extern/fastflow/ff/distributed/MTCL" "$repo_dir/../MTCL" "${HOME:-}/MTCL" || true)}
CEREAL_HOME=${CEREAL_HOME:-$(find_dependency include/cereal/cereal.hpp "$repo_dir/../FastFlow-Python/extern/cereal" \
    "$repo_dir/../cereal" "${HOME:-}/cereal" /usr /usr/local || true)}
[[ -f ${MTCL_HOME:-}/include/mtcl.hpp ]] || { echo "Set MTCL_HOME to an MTCL checkout (with include/mtcl.hpp)." >&2; exit 2; }
[[ -f ${CEREAL_HOME:-}/include/cereal/cereal.hpp ]] || { echo "Set CEREAL_HOME to the cereal root (with include/cereal/cereal.hpp)." >&2; exit 2; }
command -v timeout >/dev/null || { echo "GNU timeout is required." >&2; exit 2; }

build_dir=$(mktemp -d "${TMPDIR:-/tmp}/fastflow-mtcl-async.XXXXXX")
trap 'rm -rf -- "$build_dir"' EXIT
# FastFlow includes "MTCL/include/mtcl.hpp": an initialized in-tree submodule
# wins, otherwise this link to MTCL_HOME resolves it.
mkdir -p "$build_dir/include"
ln -s "$(cd -- "$MTCL_HOME" && pwd)" "$build_dir/include/MTCL"
echo "MTCL: $MTCL_HOME (unless the in-tree submodule is initialized)"
echo "cereal: $CEREAL_HOME"

# The receiver must stop at the first invalid access. A failing scenario
# throws past svc()'s buffer cleanup, so leak reports are disabled.
export ASAN_OPTIONS="${ASAN_OPTIONS:+$ASAN_OPTIONS:}detect_leaks=0:halt_on_error=1:abort_on_error=0:exitcode=1"

build() {  # build SUITE
    local flags=(-std=c++17 -O0 -g -pthread -Wall)
    [[ $1 == receiver ]] && flags+=(-fsanitize=address -fno-omit-frame-pointer -fno-pie -no-pie)
    "${CXX:-c++}" "${flags[@]}" -I"$repo_dir" -I"$build_dir/include" -I"$CEREAL_HOME/include" \
        "$test_dir/test_$1.cpp" -o "$build_dir/test_$1" || { echo "Build of test_$1 failed." >&2; exit 2; }
}

passed=0
failed=0
status=0
for suite in "${suites[@]}"; do
    build "$suite"
    read -r -a cases <<<"${selected_case:-$(scenarios_of "$suite")}"
    modes=(control regression)
    $control_only && modes=(control)
    for ((iteration = 1; iteration <= repeat; ++iteration)); do
        for mode in "${modes[@]}"; do
            for scenario in "${cases[@]}"; do
                args=(--case "$scenario")
                [[ $mode == control ]] && args+=(--control)
                echo "[$iteration/$repeat] $suite $scenario ($mode)"
                result=0
                timeout --kill-after=2s "${TEST_TIMEOUT:-10}s" "$build_dir/test_$suite" "${args[@]}" || result=$?
                if ((result == 0)); then
                    passed=$((passed + 1))
                    continue
                fi
                failed=$((failed + 1))
                if ((result == 1)); then
                    ((status == 2)) || status=1
                else
                    echo "Harness or process error: exit $result (124 means timeout)." >&2
                    status=2
                fi
            done
        done
    done
done
echo "Results: $passed passed, $failed failed."
exit "$status"
