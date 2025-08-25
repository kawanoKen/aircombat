rem Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

set prev=%cd%
set current=%~dp0
cd %current%\deps

if exist tmpfile (
    del tmpfile
)

rem ここに依存ライブラリをダウンロードする処理を記述する

cd %prev%
