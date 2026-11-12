#!/usr/bin/env bash
# Build once, run each scenario in a separate process, leave no build artifacts.
set -euo pipefail
trap 'echo "Harness setup failed (line $LINENO)." >&2; exit 2' ERR

test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$test_dir/../.." && pwd)
# AddressSanitizer must stop at the first invalid access. Receiver exceptions
# bypass svc()'s manual buffer cleanup, so leak reports are outside this test.
export ASAN_OPTIONS="${ASAN_OPTIONS:+$ASAN_OPTIONS:}detect_leaks=0:halt_on_error=1:abort_on_error=0:exitcode=1"
selected=all
control_only=false
repeat=1
usage() {
    echo "Usage: $0 [--case frame-size|buffer-capacity] [--control] [--repeat N]"
    echo "Default: both controls, then both regressions; a regression failure exits 1."
    echo "Builds with AddressSanitizer (required for buffer-capacity); requires GNU timeout."
    echo "Dependencies: MTCL_HOME, CEREAL_HOME; optional CXX and DRECEIVER_TEST_TIMEOUT."
}
while (($#)); do
    case "$1" in
        --case)
            [[ $# -ge 2 && ( $2 == frame-size || $2 == buffer-capacity ) ]] || { usage >&2; exit 2; }
            selected=$2; shift 2 ;;
        --control) control_only=true; shift ;;
        --repeat)
            [[ $# -ge 2 && $2 =~ ^[1-9][0-9]*$ ]] || { usage >&2; exit 2; }
            repeat=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

if [[ -z ${MTCL_HOME:-} ]]; then
    for candidate in "$repo_dir/ff/distributed/MTCL" \
                     "$repo_dir/../FastFlow-Python/extern/fastflow/ff/distributed/MTCL" \
                     "$repo_dir/../MTCL" "${HOME:-}/MTCL"; do
        if [[ -f $candidate/include/mtcl.hpp ]]; then MTCL_HOME=$candidate; break; fi
    done
fi
if [[ -z ${CEREAL_HOME:-} ]]; then
    for candidate in "$repo_dir/../FastFlow-Python/extern/cereal" \
                     "$repo_dir/../cereal" "${HOME:-}/cereal" /usr /usr/local; do
        if [[ -f $candidate/include/cereal/cereal.hpp ]]; then CEREAL_HOME=$candidate; break; fi
    done
fi
[[ -f ${MTCL_HOME:-}/include/mtcl.hpp ]] || {
    echo "Set MTCL_HOME to an initialized MTCL checkout (containing include/mtcl.hpp)." >&2; exit 2;
}
[[ -f ${CEREAL_HOME:-}/include/cereal/cereal.hpp ]] || {
    echo "Set CEREAL_HOME to the cereal root (containing include/cereal/cereal.hpp)." >&2; exit 2;
}
command -v timeout >/dev/null || { echo "GNU timeout is required." >&2; exit 2; }

build_dir=$(mktemp -d "${TMPDIR:-/tmp}/fastflow-dreceiver-async.XXXXXX")
trap 'rm -rf -- "$build_dir"' EXIT
# A staging include directory supports MTCL_HOME with any directory basename.
# The receiver's quoted include first selects an initialized in-tree submodule;
# otherwise it resolves this link. No production files are copied or changed.
mkdir -p "$build_dir/include"
ln -s "$(cd -- "$MTCL_HOME" && pwd)" "$build_dir/include/MTCL"
echo "MTCL fallback: $MTCL_HOME (an initialized in-tree submodule takes precedence)"
echo "cereal: $CEREAL_HOME"
if ! "${CXX:-c++}" -std=c++17 -O0 -g -pthread -fsanitize=address -fno-omit-frame-pointer -fno-pie -no-pie \
    -I"$repo_dir" -I"$build_dir/include" -I"$CEREAL_HOME/include" \
    "$test_dir/test_dreceiverMTCL_async.cpp" -o "$build_dir/test_dreceiverMTCL_async"; then
    echo "Harness build failed." >&2
    exit 2
fi

cases=(frame-size buffer-capacity)
[[ $selected == all ]] || cases=("$selected")
passed=0
failed=0
result=0
for ((iteration=1; iteration<=repeat; ++iteration)); do
    modes=(control regression)
    $control_only && modes=(control)
    for mode in "${modes[@]}"; do
        for scenario in "${cases[@]}"; do
            args=(--case "$scenario")
            [[ $mode != control ]] || args+=(--control)
            echo "[$iteration/$repeat] $scenario ($mode)"
            if timeout --kill-after=2s "${DRECEIVER_TEST_TIMEOUT:-10}s" \
                    "$build_dir/test_dreceiverMTCL_async" "${args[@]}"; then
                passed=$((passed + 1))
            else
                status=$?
                failed=$((failed + 1))
                [[ $result != 0 ]] || result=1
                if [[ $status != 1 ]]; then
                    echo "Harness/process error: exit $status (124 means timeout)." >&2
                    result=2
                fi
            fi
        done
    done
done
echo "Results: $passed passed, $failed failed."
exit "$result"
