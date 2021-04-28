#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# This script illustrates how to run the build tests locally
# This requires docker

tail -n 15 .circleci/config.yml | sed s/.\\+\"\\\(\.\\+\\\)\"/\\1/g | xargs -P 4 -o -I {} bash -c "circleci build --job {} && (>&2 echo "{}")" > /dev/null
