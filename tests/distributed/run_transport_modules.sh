#!/usr/bin/env bash
# Run each contract scenario in a fresh process, using real local MTCL handles.
set -euo pipefail
test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$test_dir/../.." && pwd)
mtcl_root=${MTCL_HOME:-$repo_dir/ff/distributed/MTCL}
cereal_root=${CEREAL_HOME:-/usr}
[[ -f $mtcl_root/include/mtcl.hpp && -f $cereal_root/include/cereal/cereal.hpp ]] || {
    echo "Set MTCL_HOME and CEREAL_HOME to initialized dependencies." >&2
    exit 2
}
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/fastflow-transport-modules.XXXXXX")
trap 'rm -rf -- "$build_dir"' EXIT
mkdir -p "$build_dir/include"
ln -s "$(cd -- "$mtcl_root" && pwd)" "$build_dir/include/MTCL"
"${CXX:-c++}" -std=c++17 -O0 -g -pthread \
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie -no-pie \
    -I"$repo_dir" -I"$build_dir/include" -I"$cereal_root/include" \
    "$test_dir/test_transport_modules.cpp" -o "$build_dir/test_transport_modules"
export ASAN_OPTIONS="${ASAN_OPTIONS:+$ASAN_OPTIONS:}detect_leaks=0:halt_on_error=1"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:+$UBSAN_OPTIONS:}halt_on_error=1:print_stacktrace=1"
for scenario in protocol tokens batch separate producer-end drain; do
    timeout --kill-after=2s "${TRANSPORT_TEST_TIMEOUT:-15}s" \
        "$build_dir/test_transport_modules" "$scenario"
done
