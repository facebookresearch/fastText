#!/bin/usr/env sh
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

LG=$(basename --suffix=".txt" "${1}")

./filter_utf8 < "shard/${LG}.txt" \
    | ./dedup > "shard/${LG}.dedup"
