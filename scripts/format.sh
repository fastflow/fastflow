#!/bin/sh
#
# A simple script for invoking clang-format on every cpp/hpp files
#
# ref: https://cliutils.gitlab.io/modern-cmake/chapters/features/utilities.html
#

# Run format
git ls-files -- '*.cpp' '*.h' '*.hpp' | xargs clang-format -i -style=file

# Print diff
git diff --exit-code --color
