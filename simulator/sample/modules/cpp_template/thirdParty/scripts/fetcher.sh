#!/bin/bash -e
# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

prev=$(pwd)
current=$(cd $(dirname $0); pwd)
cd $current/../deps

if [ -e tmpfile ]; then
    rm -f tmpfile
fi

# ここに依存ライブラリをダウンロードする処理を記述する

cd $prev
