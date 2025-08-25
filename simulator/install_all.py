# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#
# ASRCAISim1の本体及びプラグインのビルドとインストールを楽にするためのツール
# A tool which enables build and install of ASRCAISim1 and its plugins.
#
import argparse
import glob
import os
import shutil
import subprocess
import sys
from setuptools import find_packages

# オフライン環境で必ず参照したいローカルリポジトリがあればここに書いておくと毎回指定しなくて済む。
# If your environment is offline and there are local repositories you always use, write here so that you can skip -f option.
default_offline_repo_dirs = []

base_dir = os.path.dirname(os.path.abspath(__file__))

def split_paths(arg):
    return arg.split(';')

parser=argparse.ArgumentParser()
parser.add_argument("--clean",action="store_true",help="use when you want to clean and rebuild")
parser.add_argument("-s","--sdist",action="store_true",help="use when you want to generate sdists")
parser.add_argument("-b","--bdist_wheel",action="store_true",help="use when you want to generate wheels")
parser.add_argument("-i","--install",action="store_true",help="use when you want to install packges")
parser.add_argument("-a","--all_packages",action="store_true",help="use when you want to clean and rebuild core package (ASRCAISim1 itself) and all (core/user) plugins")
parser.add_argument("-c","--core",action="store_true",help="use when you want to clean and rebuild core package (ASRCAISim1 itself)")
parser.add_argument("-p","--core-plugins",action="store_true",help="use when you want to clean and rebuild core plugins")
parser.add_argument("-u","--user-plugins",action="store_true",help="use when you want to clean and rebuild use plugins")
parser.add_argument("-m","--manual-paths",nargs="*",default=[],help="use when you want to designate paths manually")
parser.add_argument("-o","--offline",action="store_true",help="use when you want to build in an offline environment")
parser.add_argument("-f","--find-links",nargs="*",default=[],help="use to give the find-links option passed to pip when you are in an offline environment")
parser.add_argument("-n","--no-build-isolation",action="store_true",help="use when you want not to isolate build environment")
parser.add_argument("--debug",action="store_true",help="use when you want to build C++ module as debug mode")
parser.add_argument("--msys",action="store_true",help="use when you want to use MSYS")

args=parser.parse_args()

if not (args.clean or args.sdist or args.bdist_wheel):
    args.install = True

if not (args.core or args.core_plugins or args.user_plugins or len(args.manual_paths)>0):
    args.all_packages = True

build_flags = []
offline_flags = []
if args.offline:
    offline_flags.append("--no-index")
    offline_repo_dirs = default_offline_repo_dirs + args.find_links
    for d in offline_repo_dirs:
        offline_flags.append("--find-links="+d)
if args.no_build_isolation:
    offline_flags.append("--no-build-isolation")

os.environ['ASRCAISim1_enable_debug_build'] = str(args.debug)
os.environ['ASRCAISim1_build_with_msys'] = str(args.msys)
os.environ['ASRCAISim1_disable_automatic_import_of_core_plugins'] = 'True'

# ASRCAISim1本体
if args.all_packages or args.core:
    dist_dir = os.path.join(base_dir,"dist")
    if not os.path.exists(dist_dir):
        os.mkdir(dist_dir)
    if args.clean:
        subprocess.check_call([sys.executable,"setup.py","clean"])
        for dist in glob.glob(os.path.join(dist_dir, "ASRCAISim1-*")):
            os.remove(dist)
        for dist in glob.glob(os.path.join(dist_dir, "ASRCAISim1-*".lower())):
            os.remove(dist)
    if args.sdist or args.bdist_wheel or args.install:
        # wheel,installの場合もsdistを挟む
        subprocess.check_call([sys.executable,"setup.py","sdist"])
    if args.bdist_wheel or args.install:
        # cachingが上手く働かない環境もあるようなのでinstallの場合もwheelを挟む
        subprocess.check_call([sys.executable,"-m","pip","wheel",".","-w",dist_dir,"--prefer-binary","--no-deps"] + offline_flags + build_flags)
    if args.install:
        subprocess.check_call([sys.executable,"-m","pip","uninstall","-y","ASRCAISim1"])
        try:
            subprocess.check_call([sys.executable,"-m","pip","cache","remove","ASRCAISim1"])
            subprocess.check_call([sys.executable,"-m","pip","cache","remove","ASRCAISim1".lower()])
        except:
            pass
        subprocess.check_call([sys.executable,"-m","pip","install","ASRCAISim1","--find-links="+dist_dir,"--prefer-binary"] + offline_flags + build_flags)

# core plugins
if args.all_packages or args.core_plugins:
    core_plugins = [
        os.path.abspath(p)
        for p in glob.glob(os.path.join(base_dir, "core_plugins/*"))
    ]
    dist_dir = os.path.join(base_dir,"dist")
    if not os.path.exists(dist_dir):
        os.mkdir(dist_dir)
    if args.clean:
        for plugin_dir in core_plugins:
            os.chdir(plugin_dir)
            if os.path.exists(os.path.join(plugin_dir, "setup.py")):
                subprocess.check_call([sys.executable,"setup.py","clean"])
            else:
                try:
                    subprocess.check_call([sys.executable,"-m","ASRCAISim1.scripts.clean_at","."])
                except:
                    os.chdir(os.path.join(base_dir,"ASRCAISim1"))
                    script = "import sys; from scripts.clean_at import clean_at_standalone; clean_at_standalone(sys.argv[1])"
                    subprocess.check_call([sys.executable,"-c",script,plugin_dir])
                    os.chdir(plugin_dir)
            plugin_name = find_packages()[0].partition('.')[0]
            for dist in glob.glob(os.path.join(dist_dir, plugin_name+"-*")):
                os.remove(dist)
            for dist in glob.glob(os.path.join(dist_dir, plugin_name.lower()+"-*")):
                os.remove(dist)
        os.chdir(base_dir)
    if args.sdist or args.bdist_wheel or args.install:
        # core plugin間の依存が有り得るためwheel,installの場合もsdistを挟む
        for plugin_dir in core_plugins:
            os.chdir(plugin_dir)
            if os.path.exists(os.path.join(plugin_dir, "setup.py")):
                subprocess.check_call([sys.executable,"setup.py","sdist","--dist-dir="+dist_dir])
            else:
                script = "from setuptools import setup; setup()"
                subprocess.check_call([sys.executable,"-c",script,"sdist","--dist-dir="+dist_dir])
        os.chdir(base_dir)
    if args.bdist_wheel or args.install:
        # cachingが上手く働かない環境もあるようなのでinstallの場合もwheelを挟む
        for plugin_dir in core_plugins:
            plugin_name = find_packages(plugin_dir)[0].partition('.')[0]
            try:
                subprocess.check_call([sys.executable,"-m","pip","cache","remove",plugin_name])
                subprocess.check_call([sys.executable,"-m","pip","cache","remove",plugin_name.lower()])
            except:
                pass
        os.chdir(dist_dir) # base_dirだとインストール先でなく直下のASRCAISim1本体がインポートされてしまうので移動
        subprocess.check_call([sys.executable,"-m","pip","wheel","-w",dist_dir,"--find-links="+dist_dir,"--prefer-binary","--no-deps"] + offline_flags + build_flags + core_plugins)
        os.chdir(base_dir)
    if args.install:
        plugin_names = [find_packages(plugin_dir)[0].partition('.')[0] for plugin_dir in core_plugins]
        if len(plugin_names) > 0:
            subprocess.check_call([sys.executable,"-m","pip","uninstall","-y"]+plugin_names)
        subprocess.check_call([sys.executable,"-m","pip","install","--find-links="+dist_dir,"--prefer-binary"] + offline_flags + build_flags + plugin_names)
 
 # user plugins
if args.all_packages or args.user_plugins:
    user_plugins = list(set([
        os.path.dirname(os.path.abspath(p))
        for p in (
            glob.glob(os.path.join(base_dir, "user_plugins/**/pyproject.toml"), recursive = True) +
            glob.glob(os.path.join(base_dir, "user_plugins/**/setup.cfg"), recursive = True) + 
            glob.glob(os.path.join(base_dir, "user_plugins/**/setup.py"), recursive = True)
        )
        if not "thirdParty" in p
    ]))
    dist_dir = os.path.join(base_dir,"dist")
    if not os.path.exists(dist_dir):
        os.mkdir(dist_dir)
    if args.clean:
        for plugin_dir in user_plugins:
            os.chdir(plugin_dir)
            if os.path.exists(os.path.join(plugin_dir, "setup.py")):
                subprocess.check_call([sys.executable,"setup.py","clean"])
            else:
                try:
                    subprocess.check_call([sys.executable,"-m","ASRCAISim1.scripts.clean_at","."])
                except:
                    os.chdir(os.path.join(base_dir,"ASRCAISim1"))
                    script = "import sys; from scripts.clean_at import clean_at_standalone; clean_at_standalone(sys.argv[1])"
                    subprocess.check_call([sys.executable,"-c",script,plugin_dir])
                    os.chdir(plugin_dir)
            plugin_name = find_packages()[0].partition('.')[0]
            for dist in glob.glob(os.path.join(dist_dir, plugin_name+"-*")):
                os.remove(dist)
            for dist in glob.glob(os.path.join(dist_dir, plugin_name.lower()+"-*")):
                os.remove(dist)
        os.chdir(base_dir)
    if args.sdist or args.bdist_wheel or args.install:
        # user plugin間の依存が有り得るためwheel,installの場合もsdistを挟む
        for plugin_dir in user_plugins:
            os.chdir(plugin_dir)
            if os.path.exists(os.path.join(plugin_dir, "setup.py")):
                subprocess.check_call([sys.executable,"setup.py","sdist","--dist-dir="+dist_dir])
            else:
                script = "from setuptools import setup; setup()"
                subprocess.check_call([sys.executable,"-c",script,"sdist","--dist-dir="+dist_dir])
        os.chdir(base_dir)
    if args.bdist_wheel or args.install:
        # cachingが上手く働かない環境もあるようなのでinstallの場合もwheelを挟む
        for plugin_dir in user_plugins:
            plugin_name = find_packages(plugin_dir)[0].partition('.')[0]
            try:
                subprocess.check_call([sys.executable,"-m","pip","cache","remove",plugin_name])
                subprocess.check_call([sys.executable,"-m","pip","cache","remove",plugin_name.lower()])
            except:
                pass
        os.chdir(dist_dir) # base_dirだとインストール先でなく直下のASRCAISim1本体がインポートされてしまうので移動
        #build identifierが有効な場合にbuild isolationを行うと異なる値となってしまうため、wheelを挟む
        subprocess.check_call([sys.executable,"-m","pip","wheel","-w",dist_dir,"--find-links="+dist_dir,"--prefer-binary","--no-deps"] + offline_flags + build_flags + user_plugins)
        os.chdir(base_dir)
    if args.install:
        plugin_names = [find_packages(plugin_dir)[0].partition('.')[0] for plugin_dir in user_plugins]
        if len(plugin_names) > 0:
            subprocess.check_call([sys.executable,"-m","pip","uninstall","-y"]+plugin_names)
        subprocess.check_call([sys.executable,"-m","pip","install","--find-links="+dist_dir,"--prefer-binary"] + offline_flags + build_flags + plugin_names)

# manual paths
if len(args.manual_paths)>0:
    manual_paths = list(set(sum([[
        os.path.dirname(os.path.abspath(p))
        for p in (
            glob.glob(os.path.join(m, "**/pyproject.toml"), recursive = True) +
            glob.glob(os.path.join(m, "**/setup.cfg"), recursive = True) + 
            glob.glob(os.path.join(m, "**/setup.py"), recursive = True)
        )
        if not "thirdParty" in p
    ] for m in args.manual_paths],[])))
    dist_dir = os.path.join(base_dir,"dist")
    if not os.path.exists(dist_dir):
        os.mkdir(dist_dir)
    if args.clean:
        for plugin_dir in manual_paths:
            os.chdir(plugin_dir)
            if os.path.exists(os.path.join(plugin_dir, "setup.py")):
                subprocess.check_call([sys.executable,"setup.py","clean"])
            else:
                try:
                    subprocess.check_call([sys.executable,"-m","ASRCAISim1.scripts.clean_at","."])
                except:
                    os.chdir(os.path.join(base_dir,"ASRCAISim1"))
                    script = "import sys; from scripts.clean_at import clean_at_standalone; clean_at_standalone(sys.argv[1])"
                    subprocess.check_call([sys.executable,"-c",script,plugin_dir])
                    os.chdir(plugin_dir)
            plugin_name = find_packages()[0].partition('.')[0]
            for dist in glob.glob(os.path.join(dist_dir, plugin_name+"-*")):
                os.remove(dist)
            for dist in glob.glob(os.path.join(dist_dir, plugin_name.lower()+"-*")):
                os.remove(dist)
        os.chdir(base_dir)
    if args.sdist or args.bdist_wheel or args.install:
        # user plugin間の依存が有り得るためwheel,installの場合もsdistを挟む
        for plugin_dir in manual_paths:
            os.chdir(plugin_dir)
            if os.path.exists(os.path.join(plugin_dir, "setup.py")):
                subprocess.check_call([sys.executable,"setup.py","sdist","--dist-dir="+dist_dir])
            else:
                script = "from setuptools import setup; setup()"
                subprocess.check_call([sys.executable,"-c",script,"sdist","--dist-dir="+dist_dir])
        os.chdir(base_dir)
    if args.bdist_wheel or args.install:
        # cachingが上手く働かない環境もあるようなのでinstallの場合もwheelを挟む
        for plugin_dir in manual_paths:
            plugin_name = find_packages(plugin_dir)[0].partition('.')[0]
            try:
                subprocess.check_call([sys.executable,"-m","pip","cache","remove",plugin_name])
                subprocess.check_call([sys.executable,"-m","pip","cache","remove",plugin_name.lower()])
            except:
                pass
        os.chdir(dist_dir) # base_dirだとインストール先でなく直下のASRCAISim1本体がインポートされてしまうので移動
        #build identifierが有効な場合にbuild isolationを行うと異なる値となってしまうため、wheelを挟む
        subprocess.check_call([sys.executable,"-m","pip","wheel","-w",dist_dir,"--find-links="+dist_dir,"--prefer-binary","--no-deps"] + offline_flags + build_flags + manual_paths)
        os.chdir(base_dir)
    if args.install:
        plugin_names = [find_packages(plugin_dir)[0].partition('.')[0] for plugin_dir in manual_paths]
        if len(plugin_names) > 0:
            subprocess.check_call([sys.executable,"-m","pip","uninstall","-y"]+plugin_names)
        subprocess.check_call([sys.executable,"-m","pip","install","--find-links="+dist_dir,"--prefer-binary"] + offline_flags + build_flags + plugin_names)
