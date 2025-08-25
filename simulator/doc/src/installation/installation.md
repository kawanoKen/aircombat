# 環境構築方法 {#page_installation}

## 推奨動作環境(主要な項目のみ) {#section_installation_recommended_environment}

本シミュレータの推奨動作環境は以下の通りである。

- OS : Ubuntu 22.04 LTS
- Boost : 1.88.0
- cereal : 1.3.2
- CMake : 3.22.1 (>=3.10)
- Doxygen : 1.14.0 (取扱説明書を生成する場合のみ)
- Eigen : 3.4.0
- gcc : 11.4.0 (C++20)
- gymnasium : 0.28.1〜1.0.0 (rayのバージョンに依存する)
- Magic Enum C++ : 0.9.7
- nlohmann's json : 3.12.0
- NLopt : 2.7.1
- pybind11 : 2.13.6
- Python : 3.10
- PyTorch : 2.7.0 (>=2.1.0)
- ray[default,tune,rllib] : 2.7.0〜2.46.0
- thread-pool : 5.0.0

上記のうちC++依存ライブラリの一覧は
\subpage page_cpp_third_party_dependency
に記載している。

## インストール方法 {#section_installation_procedure}

### Python仮想環境の起動 {#section_installation_python_venv}

　本シミュレータのビルドスクリプトは"python"コマンドでインタプリタを起動する仕様となっている。
そのため、venvやconda等を用いて仮想環境を利用することを推奨する。

```sh
cd your/path/to/create/virtual/env
python3 -m venv env_name
source env_name/bin/activate
```

### setuptoolsのインストール {#section_installation_install_setuptools}

後述のインストール用スクリプトはビルド時にsetuptools>=62.4が必要なため、もし未インストールの場合は事前にインストールしておく。

```sh
pip install "setuptools>=62.4"
```

### シミュレータのビルド・インストール {#section_installation_install_simulator}

リポジトリ直下で、

```sh
python install_all.py -i -c # 本体部分
python install_all.py -i -p # core_plugins以下に存在するプラグイン全て
python install_all.py -i -u # user_plugins以下に存在するプラグイン全て
#python install_all.py -a #-aは-c -p -uを全て指定するのと同じ効果
```

の順に実行すると、本体、core plugin、user pluginの順にビルド・インストールされる。

### 深層学習用フレームワークのインストール {#section_installation_install_deep_learning_framework}

シミュレータ及びサンプルの全ての機能を使用するためにはPyTorchとTensorflow(Tensorboard)が必要であるため、
各ユーザーは自身の所望のバージョンのこれらのフレームワークを別途インストールしておく必要がある。
なお、PyTorchについては一部のプラグインのrequirementsに含まれているため、未インストールだった場合はpip経由で
自動的にインストールされる。

### 第4回空戦AIチャレンジ用サンプルのビルド・インストール {#section_installation_install_sample_for_4th_acac}

リポジトリ直下で、-mオプションをつけて

```sh
python install_all.py -i -m "./sample/modules/R7ContestSample"
```

を実行するとR7ContestSampleパッケージがuser pluginとしてインストールされる。

### ドキュメントの生成

#### Doxygenのインストール

本シミュレータはC++20の機能を使用しているため、Ubuntu 22.04のaptリポジトリから得られるDoxygenでは正しくドキュメントを生成できない。最新版のバイナリをGitHub等からダウンロードしてインストールすること。

```sh
# graphviz、pdf2svgをインストール
sudo apt-get install -y graphviz pdf2svg

# LATEXをインストール
sudo apt-get install -y texlive-ful

# LATEXのインストール中に、
# "Pregenerating ConTeXt MarkIV format. This may take some time..."
# というメッセージが出たところで止まってしまうが、
# Enter を連打(長押し)することでインストールが完了する。

# Doxygenの最新版のバイナリをダウンロード
curl https://github.com/doxygen/doxygen/releases/download/Release_1_14_0/doxygen-1.14.0.linux.bin.tar.gz -o doxygen-1.14.0.linux.bin.tar.gz -L

tar -xf doxygen-1.14.0.linux.bin.tar.gz
cd doxygen-1.14.0
sudo make
sudo make install
```

#### ドキュメント生成スクリプトの実行

上記のインストールが完了した後、仮想環境を起動した状態で、以下を実行するとdoc/html以下似ドキュメントが生成され、そのトップページはdoc/html/core/index.htmlとして生成される。

```sh
bash ./doc/scripts/generate.sh
```

### install_all.pyのオプション一覧 {#section_installation_install_all_py_options}

```txt
usage: install_all.py [-h] [--clean] [-s] [-b] [-i] [-a] [-c] [-p] [-u] [-m [MANUAL_PATHS ...]] [-o] [-f [FIND_LINKS ...]] [-n] [--debug] [--msys]

options:
  -h, --help            show this help message and exit
  --clean               use when you want to clean and rebuild
  -s, --sdist           use when you want to generate sdists
  -b, --bdist_wheel     use when you want to generate wheels
  -i, --install         use when you want to install packges
  -a, --all_packages    use when you want to clean and rebuild core package (ASRCAISim1 itself) and all (core/user) plugins
  -c, --core            use when you want to clean and rebuild core package (ASRCAISim1 itself)
  -p, --core-plugins    use when you want to clean and rebuild core plugins
  -u, --user-plugins    use when you want to clean and rebuild use plugins
  -m [MANUAL_PATHS ...], --manual-paths [MANUAL_PATHS ...]
                        use when you want to designate paths manually
  -o, --offline         use when you want to build in an offline environment
  -f [FIND_LINKS ...], --find-links [FIND_LINKS ...]
                        use to give the find-links option passed to pip when you are in an offline environment
  -n, --no-build-isolation
                        use when you want not to isolate build environment
  --debug               use when you want to build C++ module as debug mode
  --msys                use when you want to use MSYS
```

## 動作確認 {#section_installation_check_if_installed_correctly}

### シミュレータ本体の動作確認 {#section_installation_check_simulator}

sample/scripts/R7Contest/MinimumEvaluationに移動し、

```sh
python evaluator.py "Rule-Fixed" "Rule-Fixed" -n 1 -v -l "./result"
```

を実行すると、オープン部門のシナリオでルールベースの初期行動判断モデルどうしの戦闘が可視化しながら1回行われ、その結果を保存したログが./resultに保存される。GUIの無い環境で使用したい場合は、-vのオプションを省略することで可視化を無効化できる。
ユース部門のシナリオで実行したい場合は-yオプションを指定する。

### HandyRLを用いた学習のサンプル {#section_installation_check_HandyRL_sample}

#### 学習の実施 {#section_installation_check_HandyRL_train}

sample/scripts/R7Contest/HandyRLSampleに移動し、

```sh
python main.py R7_contest_open_sample_M.yaml --train
```

を実行すると、オープン部門のシナリオで1体で陣営全体を操作する行動判断モデルについて、Self-Playによる学習が行われる。
学習結果は./results/Open/Multi/YYYYmmddHHMMSS以下に保存される。
なお、"M"を"S" に変更すると、1 体で1機を操作する行動判断モデルとなり、保存先は"Multi"の代わりに"Single"となる。
また、ユース部門の場合は上記のopen,Openをyouth,Youthと読み替えること。

#### 学習済モデルの評価 {#section_installation_check_HandyRL_evaluation}

学習済モデルは、./results/Open/Multi/YYYYmmddHHMMSS/policies/checkpoints以下に保存されている。
このモデルの評価は上記のevaluator.pyを使用することで可能である。
sample/scripts/R7Contest/MinimumEvaluation/candidates.jsonを開き、例えば

```json
"test":{
    "userModuleID":"HandyRLSample01M",
    "args":{"weightPath":"<学習済の.pthファイルのフルパス>"}
}
```

のように候補を追加し、sample/scripts/R7Contest/MinimumEvaluation上で

```sh
python evaluator.py "test" "Rule-Fixed" -n 10 -v -l "eval_test.csv"
```

と実行すると、初期行動判断モデルとの対戦による学習済モデルの評価を行うことができる。
ユース部門の場合は上記のopen,Openをyouth,Youthと読み替えること。また、-yオプションの指定を忘れないこと。

## Dockerコンテナの使用 {#section_installation_docker_sample}

dockerディレクトリ以下に、本シミュレータを利用するdockerイメージの構築サンプルを配置している。

### イメージのビルド {#section_installation_build_docker_image}

build_docker_image.shを用いる。GUI環境の要否に応じてcli,vnc,xrdpのいずれかをコマンドライン引数に与える。

```sh
bash build_docker_image.sh [cli|vnc|xrdp]
```

### コンテナの起動 {#section_installation_run_container}

run_container.shに起動コマンドの例を記述している。

```sh
bash run_container.sh [cli|vnc|xrdp]
```

### コンテナの中身について {#section_installation_container_description}

シミュレータは/opt/data/ASRCAISim1に格納されており、/opt/data/venv/acac_4thにある仮想環境にインストールされている。

```sh
source /opt/data/venv/acac_4th/bin/activate
```

で仮想環境を起動できる。

## アンインストール方法 {#section_installation_uninstall}

pip uninstallでもよいが、プラグインを一括でアンインストールするためのスクリプトuninstall.pyを用意している。

```sh
python uninstall_all.py -u # user plugin全て
python uninstall_all.py -p # core plugin全て
python uninstall_all.py -c # 本体部分
#python uninstall_all.py -a #-aは-c -p -uを全て指定するのと同じ効果
```

### uninstall_all.pyのオプション一覧 {#section_uninstall_all_py_options}

```txt
usage: uninstall_all.py [-h] [-a] [-c] [-p] [-u]

options:
  -h, --help          show this help message and exit
  -a, --all_packages  use when you want to clean and rebuild core package (ASRCAISim1 itself) and all (core/user) plugins
  -c, --core          use when you want to clean and rebuild core package (ASRCAISim1 itself)
  -p, --core-plugins  use when you want to clean and rebuild core plugins
  -u, --user-plugins  use when you want to clean and rebuild use plugins
```
