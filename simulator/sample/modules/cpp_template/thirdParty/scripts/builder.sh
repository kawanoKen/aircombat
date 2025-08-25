#!/bin/bash -e
# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

prev=$(pwd)
current=$(cd $(dirname $0); pwd)
cd $current/../deps

# ここにダウンロードしてきた依存ライブラリをビルドする処理を記述する。

cd $prev
