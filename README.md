# 配布データと応募用ファイル作成方法の説明

本コンペティションで配布されるデータと応募用ファイルの作成方法や投稿する際の注意点などについて説明する.

1. [配布データ](#配布データ)
1. [応募用ファイルの作成について](#応募用ファイルの作成について)
1. [コンペティションサイトで取得できる対戦ログデータ](#コンペティションサイトで取得できる対戦ログデータ)
1. [投稿時の注意点](#投稿時の注意点)

※本ドキュメントはChromeのMarkdown Viewerを導入した状態でChromeブラウザでみることを推奨する.

## 配布データ

配布されるデータは以下の通り.

- [シミュレータ環境](#シミュレータ環境)
- [応募用サンプルファイル](#応募用サンプルファイル)

### シミュレータ環境

実行プログラム等を含む空戦を再現するシミュレータ環境. `simulator_dist.zip`が本体で, 解凍すると, 以下のようなディレクトリ構造のデータが作成される.

```bash
simulator_dist
├── Agents                            : 行動判断モジュールを格納したディレクトリ
├── common                            : コンペティションで定義されている戦闘場面などの設定ファイルを格納したディレクトリ
├── dist                              : シミュレータ本体を含めた依存ライブラリのwhlファイルを格納したディレクトリ
├── dockerfiles                       : Dockerによる環境構築に使われるベースイメージファイルを格納したディレクトリ
│   ├── Dockerfile.cpu                : CPU用の環境
│   ├── Dockerfile.gpu                : GPU用の環境
│   └── Dockerfile.macos              : MacOS用の環境
├── docs                              : 説明資料を格納したディレクトリ
│   └── html
│       └── core
│           └── index.html            : 説明資料のトップページ
├── figs                              : 説明図などを格納したディレクトリ
├── simulator                         : シミュレータ本体
├── src                               : 対戦などを行うためのモジュールを実装したプログラム
├── docker-compose.yml                : 仮想環境を構築するための定義ファイル
├── make_agent.ipynb                  : エージェントを作成するための簡単な手順を示したノートブック
├── README.md                         : このファイル
├── replay.py                         : 戦闘シーンを動画として作成するプログラム
└── validate.py                       : エージェント同士の対戦を実行するプログラム
```

#### 説明資料

説明資料はDoxygenで生成されており, [docs/html/core/index.html](docs/html/core/index.html)にトップページがある.

ページ構成は以下の通り.

```bash
index.html
├─ 環境構築方法
├─ シミュレーションに関する説明
├─ 行動判断モデルの学習のための機能
├─ 基本的な空対空戦闘場面のシミュレーション
├─ プラグインについて
└─ 第4回空戦AIチャレンジ向けチュートリアル
```

なお, 説明資料中のファイルパスは`simulator`ディレクトリを基準とした相対パスで書かれている.

シミュレータにデフォルトで実装されている初期行動判断モデルについては[ここ](docs/html/core/a198027.html)を参照. なお, 本モデルは本コンペティションにおいてベンチマークとして参戦する.

### 応募用サンプルファイル

応募用のサンプルファイル. 実体は`sample_submit.zip`で, 解凍すると以下のようなディレクトリ構造のデータが作成される.

```bash
sample_submit
├── __init__.py
├── args.json
└── config.json
```

詳細や作成方法については[応募用ファイルの作成方法](#応募用ファイルの作成方法)を参照すること.

## 応募用ファイルの作成について

応募用ファイルは学習済みモデルなどを含めた, 行動判断を実行するためのソースコード一式をzipファイルでまとめたものとする.

### 環境構築

まずは実行テストなどを行う準備として, 環境構築を行う. 主に手持ちのPCのローカル環境で行う方法とDockerでコンテナ型仮想環境を構築する方法がある. 評価システムで運用されている環境を再現する意味ではDockerにより構築する方が望ましい.

#### ローカル環境で構築する場合

推奨OSは`Ubuntu 22.04`. WindowsOSの場合は, `wsl`を導入したうえで, Ubuntu環境を構築しておくとよい. 導入方法については[こちら](https://learn.microsoft.com/ja-jp/windows/wsl/install)を参照されたい. また, AnacondaなどのPython環境(Pythonの推奨バージョンは3.10)を構築しておくことが必要である. 構築しておらず, Anacondaを導入する場合は[ここ](https://www.anaconda.com/download)より, 自身のOSから判断してインストーラをダウンロードしてインストールできる. そして以下のPythonライブラリがインストールされているとする(torch以外は厳密にバージョンを合わせる必要はないが, なるべく合わせておく).

```plaintext
torch==2.7.0, !=2.0.1
tensorflow==2.13.0
pandas==2.1.4
scikit-learn==1.6.1
lightgbm==4.5.0
timeout-decorator==0.5.0
opencv-python-headless==4.11.0.86
```

インストールしていない場合は`pip`などによりインストールしておくこと. 例えば, `torch`は[ここ](https://pytorch.org/get-started/previous-versions/), `tensorflow`は[ここ](https://www.tensorflow.org/install)を参照.


`./dist`以下にシミュレータ本体とその他依存パッケージのビルド済みのwhlファイルがある. 以下のコマンドを実行して, 実際にシミュレータ環境を構築されたい.

```bash
$ cd dist/linux # linux, x86の場合
$ pip install -r requirements.txt
...
```

ソースからビルドしたい場合(時間がかかる)は`simulator/README.md`又はドキュメントの[環境構築方法](docs/html/core/a197995.html)のページを参照すること.

##### MacOS(Apple Silicon)での構築

制約はあるが, MacOSで環境構築することは可能. `./dist/macos`にビルド済みのwhlファイルを用意してあるので, インストールする場合は利用すること.

ソースからビルドする場合, 以下の点に注意. 

- CMakeのバージョンは3.30未満のものを使用すること.

3.30以降だとFindBoost関係の互換性がなくなりビルドに失敗してしまうため, 以下を参考に, 3.29.6のcmake.rbをダウンロードし, `brew install ./cmake.rb`によりインストールする.
https://github.com/Homebrew/homebrew-core/blob/master/Formula/c/cmake.rb

- HandyRLサンプルはMPSを使用できるが, rayがMPSを認識しないので, yamlのtrain_argsのnum_gpusを0にすること.

PyTorchはMPSに対応しているためHandyRLサンプルの該当部分はCUDAとMPSで分岐するよう修正しているが, rayがMPSをGPUとして認識しないため, yamlにおけるrayのリソース指定(num_gpus)は0にする必要がある旨コメントを追記してある.


#### Dockerで構築する場合

Dockerで環境構築する場合, Docker Engineなど, Dockerを使用するために必要なものがない場合はまずはそれらを導入しておく. [Docker Desktop](https://docs.docker.com/get-docker/)を導入すると必要なものがすべてそろうので, 自身の環境に合わせてインストーラをダウンロードして導入しておくことが望ましい. デフォルトでは, NVIDIA GPU上での実行を前提とした環境(ベースイメージの元となっているDockerfileは`dockerfiles/Dockerfile.gpu`)となっている. GPU環境を持っていないならば, CPU上での実行を前提とした環境(ベースイメージの元となっているDockerfileは`dockerfiles/Dockerfile.cpu`)を構築する. 具体的には, `docker-compose.yml`の`dockerfile`の項目を以下のように編集する. なお, 評価システムではx86 CPU上での実行となる.

```yml
services:
  dev1:
    build:
      context: .
      dockerfile: dockerfiles/Dockerfile.cpu
    container_name: atla4
    ports:
      - "8080:8080"
    volumes:
      - .:/workspace
    tty: true
```

そして, 以下のコマンドを実行することで立ち上げる.

```bash
$ docker compose up -d
...
```

`docker-compose.yml`は編集するなりして, 自身が使いやすいように改造してもよい. 無事にコンテナが走ったら, 必要なデータなどをコンテナへコピーする.

```bash
$ docker cp /path/to/some/file/or/dir {コンテナ名}: {コンテナ側パス}
... 
```

そして, 以下のコマンドでコンテナの中に入り, 分析や開発を行う.

```bash
$ docker exec -it {コンテナ名} bash
...
```

`コンテナ名`には`docker-compose.yml`の`services`->`dev1`->`container_name`に記載の値を記述する. `.`をコンテナ側の`/workspace`へバインドマウントした状態(`.`でファイルの編集などをしたらコンテナ側にも反映される. 逆もしかり.)で, `/workspace`からスタートする. さらにGUI機能も導入されているため, GUIでシミュレータを動作可能(後述).

##### MacOS(Apple Silicon)での構築

`Dockerfile.cpu`はx86を前提としているので, 代わりにarm64用の`Dockerfile.macos`を用いること(`docker-compose.yml`の`services`->`dev1`->`dockerfile`の項目を`Dockerfile.macos`と記載する). ただし, DockerはMPSに対応していないので, MPSを用いる場合はローカルで環境構築を行うこと.

また, 以下のような制約がある.
- テスト実行において, コマンドライン上で可視化をオンにしても表示されないが, VNCを使えば可視化できる.
- MPSを認識しないのでcpu-onlyではあるが, 強化学習サンプルは動作する.
- x86イメージでも, Apple Virtualization frameworkではなくDocker VMMで起動すれば一応動作する. ただし, VNCを使っても可視化できず, rayが使えないので強化学習サンプルは動作しないため, x86環境での投稿前動作確認程度にしか使えない.

##### CUDA環境の利用

ホスト側にCUDA環境に対応したGPUがある場合, CUDA環境に対応したイメージからコンテナを構築可能. 実際にGPUがコンテナ内で有効になっているかどうかは以下のコマンドで確認できる.

```bash
# コンテナに入った後
$ python -c "import torch; print(torch.cuda.is_available())"
True
$ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
...{GPUデバイスのリスト}...
```

#### その他の方法で構築する場合

その他の方法としては, 例えば[Google Colab](https://colab.research.google.com/?hl=ja)がある. Googleアカウントを持っていないならば, 事前に作成しておく必要がある. そのうえで配布シミュレータ`simulator_dist.zip`を`/content`直下にアップロードして, 新規のノートブック上で解凍し, [ローカル環境で構築する場合](#ローカル環境で構築する場合)と同様の手順で環境を構築すれば分析を進めることができる. なお, OSやPythonのバージョンは推奨バージョンと必ずしも一致しないことに注意すること.

#### GUI機能について

[VSCode](https://code.visualstudio.com/)を導入済みであれば, VSCodeから(コンテナに接続などして)実際に分析や開発などを行うことが可能. [Dockerで構築する場合](#dockerで構築する場合), GUIとしては`lxde`と, VNC(Virtual Network Computing)とnoVNCを導入した状態となる. よって, HTTPサーバーで公開すれば, クライアント側はブラウザでアクセスするだけで原理的にはリモートデスクトップ環境を使えるようになる. Dockerから構築した場合は`jupyter lab`も利用可能となっている.

##### VSCodeの利用

例えばWindowsOSの場合は[Remote-WSL](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl)を導入しておくと, VSCodeからlinux環境(wsl)にアクセスすることができる. 導入後, コマンドパレットで`wsl`と打つと一覧に`WSL: New Window`などとでるので, それを選択することで機能が使える. また, [VSCode](https://code.visualstudio.com/)には立ち上げたDockerコンテナに接続して開発を行える[拡張機能](https://code.visualstudio.com/docs/devcontainers/containers)がある. まずVSCode [設定]→[拡張機能]→[Dev Container]:インストールする. その後コマンドパレットで`dev..`と打つと一覧に`Dev Containers: Attach to Running Container`が出るのでそれを選択して, 立ち上げたコンテナ名を選択すると, 別ウィンドウでワークスペースが立ち上がる. Pythonやjupyterの拡張機能をインストールし, Pythonインタープリタとjupyterのカーネルはベースイメージのものを選択すると作業環境ができる. これらを導入することで, VSCodeの高度なインテリセンス機能などを活用できる.

##### VNCの利用

[Dockerで構築する場合](#dockerで構築する場合), 以下の操作によりVNCサーバーを立ち上げてGUI操作ができるようになる.

```bash
$ USER={ユーザー名} vncserver :1 -geometry 1500x700 -depth 24
...
```

一旦立ち上がったがすぐkillされた場合は以下のコマンドで再度サーバーを立ち上げてみる.

```bash
$ tigervncserver -xstartup /usr/bin/startlxde -geometry 1500x700 -depth 24
...
```

ログインユーザーとして, ディスプレイ:1に解像度1500x700, 24bit colorのVNCデスクトップを立ち上げている. 解像度と色は好みで設定する. パスワード入力が求められるので, 6-8文字の簡単なパスワードを設定する.

起動しているかどうかは以下のコマンドで確認できる.

```bash
$ vncserver -list
...
```

削除するときは`vncserver -kill :1`を実行すればよい. ログは`/root/.vnc/*.log`に出力されているので, `tail -F /root/.vnc/*.log`で監視できる.

次にnoVNCを設定したポートに公開する.

```bash
websockify -D --web=/usr/share/novnc/ {ポート番号} localhost:5901
```

`ポート番号`にはVNCサーバー側のポート番号(Dockerで構築した場合は`docker-compose.yml`で定義したコンテナ側のポート番号(`ports`の`:`の右側の値))を指定する. VNCでは"5900+ディスプレイ番号"のポートにサーバーが立つので, ここでは`localhost:5901`を指定している.

以上の設定のあと, ホスト側でブラウザを立ち上げて`localhost:{ホスト側のポート番号}/vnc.html`にアクセスすると, noVNCの画面が立ち上がってGUIデスクトップが表示される. 画面左のメニューで各種設定や, クリップボードの確認・編集も可能.

##### jupyter labの利用

コンテナには`jupyter lab`がインストールされている. 以下のコマンドで`jupyter lab`を立ち上げることができる.

```bash
# コンテナに入った後
$ jupyter lab --port={ポート番号} --ip=0.0.0.0 --allow-root
...
copy and paste one of these URLs:
        ...
     or ...

```

`ポート番号`には`docker-compose.yml`で定義したコンテナ側のポート番号(`ports`の`:`の右側の値)を指定する. 表示されるURLをブラウザに張り付ければjupyterの環境で作業が可能となる. Desktopから直接入ってもよい(最初にtokenを聞かれるので, URLの`token=`に続く文字列を入力する).

#### 動作確認プログラムの実行

以下のコマンドにより実際に動画付きでシミュレータが動くかどうかを確認する. 成功すると動画が再生されて勝敗結果等が出力される.

```bash
$ cd /path/to/workspace
$ python validate.py --visualize 1
...
```

VSCodeからコンテナに接続した際にターミナルを開いて実行したときは以下のように動画付きで結果が出力される.

![vscode_test](./figs/vscodetest.gif)

以下はnovncによりリモートデスクトップで実行したときの実行画面.

![run_test](./figs/vnctest.gif)

動画が表示されない場合, `--visualize`を0, `--replay`を1として実行して, `replay.py`によりmp4形式の対戦動画を作成して(存在する`.dat`ファイル全てに対して作成される)動画を再生する.

```bash
$ python validate.py --visualize 0 --movie 1
...
$ xvfb-run python replay.py
...
```

### 行動判断を行うプログラムの実装方法

提出すべき行動判断を実行するためのソースコード一式は以下のようなディレクトリ構造となっていることを想定している.

```bash
.
├── __init__.py              必須: 実行時に最初に呼ばれるinitファイル
├── args.json                必須: アルゴリズムに渡す引数に関する情報をまとめたjson形式のファイル
└── ...                      任意: 追加で必要なファイルやディレクトリ
```

- エージェントに対する設定に関しては例えば"config.json"などの名前でファイルとして保存しておいて, 実行時に読み込んで使用する想定である.
- `__init__.py`で定義されたメソッドに渡す情報は`args.json`を想定している.
  - 名前は必ず`args.json`とすること.
- 学習済みモデルなど, 実行に必要なその他ファイルも全て含まれている想定(必要に応じてディレクトリを作成してもよい).
- C++版の独自クラスを作成する場合は, 評価環境がx86のUbuntuなので, そこで動作するようにビルドすること.

#### `__init__.py`の実装方法

以下のメソッドを実装すること.

##### getUserAgentClass

Agentクラスオブジェクトを返すメソッド. 以下の条件を満たす.

- 引数argsを指定する.
  - `args.json`の内容が渡される想定である.
- Agentクラスオブジェクトを返す.
  - Agentクラスオブジェクトは自作してもよいし, もともと実装されているものを直接importして用いてもよい. 実装例については`make_agent.ipynb`や`simulator/sample/modules/R7ContestSample/R7ContestSample`以下のファイルを参照すること. R7ContestSampleに関する解説は[第4回空戦AIチャレンジ向け強化学習Agentサンプル](docs/html/core/a198025.html)に記載がある.

##### getUserAgentModelConfig

Agentモデル登録用にmodelConfigを表すjsonを返すメソッド. 以下の条件を満たす.

- 引数argsを指定する.
  - `args.json`の内容が渡される想定である.
- `modelConfig`を返す. なお, `modelConfig`とは, Agentクラスのコンストラクタに与えられる二つのjson(dict)のうちの一つであり、設定ファイルにおいて

    ```json
    {
        "Factory":{
            "Agent":{
                "modelName":{
                    "class":"className",
                    "config":{...}
                }
            }
        }
    }
    ```

    の`config`の部分に記載される`{...}`の`dict`が該当する.  

##### isUserAgentSingleAsset

Agentの種類(一つのAgentインスタンスで1機を操作するのか, 陣営全体を操作するのか)を返すメソッド. 以下の条件をみたす.

- 引数argsを指定する.
  - `args.json`の内容が渡される想定である.
- 1機だけならば`True`, 陣営全体ならば`False`を返す想定である.

##### getUserPolicy

`StandalonePolicy`を返すメソッド. 以下の条件を満たす.

- 引数argsを指定する.
  - `args.json`の内容が渡される想定である.
- `StandalonePolicy`を返す.

以下は`__init__.py`の実装例.

```Python
import os,json
import ASRCAISim1
from ASRCAISim1.policy import StandalonePolicy

#①Agentクラスオブジェクトを返す関数を定義
def getUserAgentClass(args={}):
    from BasicAirToAirCombatModels01 import BasicAACRuleBasedAgent01
    return BasicAACRuleBasedAgent01

#②Agentモデル登録用にmodelConfigを表すjsonを返す関数を定義
def getUserAgentModelConfig(args={}):
    configs=json.load(open(os.path.join(os.path.dirname(__file__),"config.json"),"r"))
    modelType="Fixed"
    if(args is not None):
        modelType=args.get("type",modelType)
    return configs.get(modelType,"Fixed")

#③Agentの種類(一つのAgentインスタンスで1機を操作するのか、陣営全体を操作するのか)を返す関数を定義
def isUserAgentSingleAsset(args={}):
    #1機だけならばTrue,陣営全体ならばFalseを返すこと。
    return True

#④StandalonePolicyを返す関数を定義
class DummyPolicy(StandalonePolicy):
    def step(self,observation,reward,done,info,agentFullName,observation_space,action_space):
        return None #action_space.sample()

def getUserPolicy(args={}):
    return DummyPolicy()

```

応募用サンプルファイル`sample_submit.zip`も参照すること. また, `simulator/sample/scripts/R7Contest/MinimumEvaluation`以下にある`HandyRLSample01S`や`HandyRLSample01M`などのサンプルモジュールも参考にされたい. なお, `HandyRL`についてはサンプルで使用しているNNクラスに対応した`StandalonePolicy`の派生クラスが提供されている(`simulator/core_plugins/HandyRLUtility/HandyRLUtility`以下を参照.).

#### 強化学習サンプル

このシミュレータにはHandyRL(の改変版)を用いた強化学習サンプルが同梱されている. 動かしたい場合は[ここ](docs/html/core/a198026.html)を参照すること.

### 実行テスト

モデル学習などを行い, 行動判断を行うプログラムが実装できたら, 初期行動判断モデル(`./Agents/BenchMark`以下に実装されている)との対戦を実行する.

```bash
$ python validate.py  --myagent-id agent_id --myagent-module-path /path/to/MyAgent --result-dir /path/to/results
...
```

- `--myagent-id`には好きな自分のエージェント名を指定する. デフォルトは`0`.
- `--myagent-module-path`には実装したプログラム(`__init__.py`など)が存在するパス名を指定する. デフォルトは`Agents/MyAgent`
- `--result-dir`には対戦結果ログを格納するディレクトリを指定する. デフォルトは`./results`
- `--youth`はユース部門の設定で実行する場合は`1`, オープン部門の場合は`0`とする. デフォルトは`0`. **参加予定の部門の設定になっていることを確認すること**

実行に成功すると対戦が行われ, `{result_dir}/{agent_id}`以下に対戦結果ログ等が格納される.

- デフォルトでは`./results/validation_results.json`が対戦結果ログとして保存される. 対戦時の終了までにかかった時間や勝利数やスコアなどの情報が含まれる.
- `--replay`を`1`にすると対戦動画情報(`.dat`ファイル)が保存される. デフォルトは`0`.
  - さらに`--visualize`を`1`にすると実行中に動画が再生されるが, 実行に時間がかかる. デフォルトは`0`.
- `--num-validation`に指定した値の回数だけ対戦が行われる. デフォルトは`3`.
- `--time-out`に指定した値(単位は[秒])までに行動を返さなかった場合は行動空間からランダムサンプリングされる. デフォルトは`3.0`.
- `--memory-limit`に指定した値(単位は[GB])で対戦実行時の使用メモリの上限を設定する. 超えた場合はエラーとなり, 対戦は無効となる. デフォルトは`7.5`.
- `--random-test`を`1`にするとシミュレーションを行わずランダムに勝敗を付ける. デフォルトは`0`であるが, デバッグをしたいときなどに`1`にする.
- `--control-init`は初期配置の指定の仕方を決める. デフォルトは`random`で, こちらがコンペティションにおける設定.
- `--make-log`を`1`にすると対戦実行時に各ステップで返される自分の陣営における`observations`, `infos`とエージェントが返す`actions`のログを記録する. デフォルトは`1`でこちらがコンペティションにおける設定.
  - `{result_dir}/{agent_id}`以下に`log_{fight_n}.json`として保存される.
- `--max-size`に指定した値(単位は[GB])で対戦実行時のログデータがメモリを占有する大きさの上限を決める. 上限を超えると`observations`, `infos`, `actions`のログは記録として残らない. デフォルトは`0.08`.
  - 実行後標準出力で`Log size for fight {n}: ...[GB]`などと表示されるので, サイズを確認できる.
  - 上限を超えた場合は`log_{fight_n}.json`は保存されない.
- `--color`に"Red"または"Blue"を指定することで自分の陣営の色を指定する. "Red"を指定した場合は西軍, "Blue"を指定した場合は東軍となる. デフォルトは"Red".

`validation_results.json`のフォーマットは以下の通り.

```json
{
 "own": "Red",
 "details": {
  "1": {
   "finishedTime": 23.604158639907837,
   "step_count": 1200,
   "numAlives": {
    "Blue": 1.0,
    "Red": 0.0
   },
   "endReason": "ELIMINATION",
   "error": false,
   "red_error_message": "",
   "blue_error_message": "",
   "scores": {
    "Blue": 2.0,
    "Red": 1.0
   },
   "config": {
    ...
   }
  },
  ...
 },
 "win": 1,
 "loss": 0,
 "draw": 0,
 "status": "success",
 "message": "ok"
}
```

`own`は自分の陣営の色, `details`には各戦闘の`finishedTime`(対戦プログラムの実行にかかった時間. 単位は秒), `step_count`(対戦が終わるまでにかかったステップ数), `numAlives`(生き残った戦闘機と護衛対象機の数), `endReason`(終了理由), `error`(プログラム的なエラーを起こしたか否か), `red_error_message`(赤側が起こしたエラーの内容. ない場合は空.), `blue_error_message`(青側が起こしたエラーの内容. ない場合は空.) `scores`(各陣営のスコア), `config`(戦闘場面に関する初期条件など)が含まれ, `win`は勝利回数, `loss`は敗北の回数, `draw`は引き分けの回数, `status`はプログラム実行のステータスで,"success"の場合は成功, "error"の場合は失敗, `message`にはプログラムが正常に終了した場合は"ok", それ以外の場合はエラーの内容が記載される. `conig`において`AssetConfigDispatcher`->`{Blue,Red}InitialState`->`elements`の`value`->`instanceConfig`で初期配置に関する情報を確認できる.

`log_{n}.json`のフォーマットは以下の通り.

```json
{
 "observations":
 [
  {
    "ptime":0.1,
    "data":
    {
      "1":...,
      "2":...
    }
  },
  ...
 ],
 "actions":
 [
  {
    "ptime":0.1,
    "data":
    {
      "1":...,
      "2":...
    }
  },
  ...
 ],
 "infos":
 [
  {
    "ptime":0.1,
    "data":
    {
      "1":...,
      "2":...
    }
  },
  ...
 ]
}
```

`observations`は前処理後の観測値, `actions`は取った行動, `infos`はスコアなどの情報が各ステップごとに格納されている. `ptime`がシミュレーション内の時刻で`data`が実際のデータ. なお, `ptime`の間隔は`modelConfig`で設定した設定値によって変わる.

なお, BenchMark以外と対戦したい場合は別のエージェントを作成してそのモジュールディレクトリを`--opponent-module-path`に指定して実行すればよい. また,`--opponent-id`には対戦相手の名前を指定できる.デフォルトは`1`.

#### replay.pyについて

`validate.py`で`--myagent-id`を指定していた場合には`replay.py`にも同じ名前で`--myagent-id`を指定すること. また,`y`又は`--youth`を指定するとユース部門, 指定しなければオープン部門として描画される.


### 応募用ファイルの作成

上記の[ディレクトリ構造](#ディレクトリ構造)となっていることを確認して, zipファイルとして圧縮する.

```bash
$ cd /path/to/Agents
$ zip -r submit ./MyAgent
...
```

実行後, 作業ディレクトリにおいて`submit.zip`が作成される.

## コンペティションサイトで取得できる対戦ログデータ

コンペティションサイトに投稿したアルゴリズムの対戦ログデータをサイトで取得することができる. 内容は`validate.py`を実行したときに得られるログデータとほぼ同じ. 1対戦のログデータ(`detail`->`logs`)のメモリを占有する大きさの上限は0.08[GB]で, 超えた場合は対戦実行時に各ステップで返される自分の陣営における`observations`, `infos`とエージェントが返す`actions`のログは残らない(対戦自体は有効)が, 勝敗結果や最終スコアなどの集約した情報は残る. フォーマットは以下の通り.

```json
[
  {
    "score": 1248.614892485913,
    "record": {
      "win": 0,
      "loss": 3,
      "draw": 1,
      "nocontest": 0
    }
  },
  {
    "own": "Red",
    "result": "Lose",
    "detail": {
      "finishedTime": 9.92264437675476,
      "step_count": 899,
      "numAlives": {
        "Blue": 2.0,
        "Red": 1.0
      },
      "endReason": "TIMEUP",
      "error": false,
      "logs": {
        "observations": [...],
        "actions": [...],
        "infos": [...]
      },
      "scores": {
        "Blue": 0.5748469871489575,
        "Red": 0.0
      }
    }
  },
  ...
]
```

"score"で現在のレーティング, "record"でこれまでの戦績を確認できる("win"が勝利数, "loss"が敗北数, "draw"が引き分け数, "nocontest"が無効試合数). コンペティションサイトの投稿画面において取得可能である. ただし, 対戦バッチが実行されるたびに内容が上書きされるため, 注意が必要である. なお, 投稿後検証処理時に初期行動判断モデルとの対戦が行われるが, そちらの対戦ログ(`logs`の内容)は含まれないので, 情報が欲しい場合はローカル環境で実行して確認すること.

## 投稿時の注意点

投稿する前に自身のローカル環境で実行テストを行い, エラーなく実行できるか確認すること. 投稿時にエラーが出た場合, 以下のことについて確認してみる.

- 提出するプログラム内でインターネット接続を行うような処理を含めていないか. 評価システム上でインターネット接続はできない.
- 実行時間がかかりすぎていないか. 時間がかかりすぎるとランタイムエラーとなることがある. 使用メモリなども見直すこと.
- コンペティションサイトで対戦ログデータをダウンロードして確認したい場合, 上限があるので, ローカル環境でテストして上限を超えないか確認しておくこと.

## 更新情報

- [2025/07/18] MacOSで環境構築できるように改修した. また, `simulator/sample/modules/R7ContestSample/R7ContestSample/R7ContestPyAgentSample01.py`のL1422を`ofs+=3`に修正した. 必要に応じてR7ContestSampleを再インストールすること.
- [2025/07/14] Dockerfile.{cpu, gpu}において, protobufのバージョンをtensorflowのバージョンと互換性のあるものに変更することで, 実行時に例外が発生することを解消した.
