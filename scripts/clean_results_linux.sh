#!/usr/bin/env bash
set -euo pipefail
rm -rf results/neurreps_sha_test_s32_main \
       results/neurreps_sha_test_s32_capacity \
       results/neurreps_sha_test_s32_padding_free
echo "Removed V10 result directories. DATA/ was not touched."
