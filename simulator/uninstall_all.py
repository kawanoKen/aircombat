# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import argparse
import subprocess
import sys
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

parser=argparse.ArgumentParser()
parser.add_argument("-a","--all_packages",action="store_true",help="use when you want to clean and rebuild core package (ASRCAISim1 itself) and all (core/user) plugins")
parser.add_argument("-c","--core",action="store_true",help="use when you want to clean and rebuild core package (ASRCAISim1 itself)")
parser.add_argument("-p","--core-plugins",action="store_true",help="use when you want to clean and rebuild core plugins")
parser.add_argument("-u","--user-plugins",action="store_true",help="use when you want to clean and rebuild use plugins")
args=parser.parse_args()

if not (args.core or args.core_plugins or args.user_plugins):
    args.all_packages = True

core_plugins = [ep.name for ep in entry_points(group='ASRCAISim1.core_plugins')]
user_plugins = [ep.name for ep in entry_points(group='ASRCAISim1.user_plugins')]

if args.all_packages or args.core or args.user_plugins:
    if len(user_plugins) > 0:
        subprocess.check_call(["pip","uninstall","-y"]+user_plugins)
if args.all_packages or args.core or args.core_plugins:
    if len(core_plugins) > 0:
        subprocess.check_call(["pip","uninstall","-y"]+core_plugins)
if args.all_packages or args.core:
    subprocess.check_call(["pip","uninstall","ASRCAISim1", "-y"])
