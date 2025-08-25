"""!
Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
@package ASRCAISim1

@~japanese-en

@par core pluginsの自動検索と読み込み

    setup.pyでASRCAISim1.core_pluginsのentry pointを登録したパッケージはcore pluginとして、
    独立にインストール可能で、かつASRCAISim1本体と透過的にマージできるものとして扱う。
    また、マージされていない状態の各pluginへの参照としてASRCAISim1.plugins.{plugin name}でも参照できるようにしている。
    つまり、pluginのattributeは
    1. {plugin name}.{attribute name}
    2. ASRCAISim1.{attribute name}
    3. ASRCAISim1.plugins.{plugin name}.{attribute name}

    のいずれの形式でも参照できるものとして扱う。

    名称が衝突しないように気をつけて設計する必要があり、不具合の温床となるリスクはあるものの、柔軟性を優先したものである。

    core pluginのC++部分については、本体と同じasrc::coreの名前空間を用いるものとし、利用者にとっては透過的にアクセスできるものとして扱う。
    core pluginをリンクする際はCMakeLists.txtのtarget_link_librariesでASRCAISim1::Coreに加えてASRCAISim1::{plugin name}も指定する。
    もし個別の指定が面倒ならばASRCAISim1::Allとすればインストール済の全てcore pluginをリンクできる(AllにはASRCAISim1::Coreも含むためCoreを置換する。)

    なお、環境変数"ASRCAISim1_disable_automatic_import_of_core_plugins"を1にすることで自動読み込みを無効化できる。
    
@par 旧バージョン(1.x)との互換性及び新バージョン(2.x)への移行
    - 従来addonsをインポートしていた部分を、以下の3種類のいずれかで置換する。

        なお、当分の間、後方互換性のために従来どおりaddons経由でのインポートも使用可能としている。

        - 旧
            from ASRCAISim1.addons.{plugin name}(.{submodule name}) import {attribute name}
        - 新
            1. from ASRCAISim1.plugins.{plugin name}(.{submodule name}) import {attribute name}  ※"addons"を"plugins"に置換
            2. from ASRCAISim1(.{submodule name}) import {attribute name}  ※"addons.{plugin name}"を削除
            3. from {plugin name}(.{submodule name}) import {attribute name}   ※"ASRCAISim1.addons"を削除
    - C++部分でaddonsのヘッダファイルをインクルードしていた場合は、"addons/{plugin name}/"の階層を除き、ASRCAISim1直下にあるものとしてパスを指定する。

       こちらについては別途namespaceの変更等により再ビルドが必須になっているため、後方互換性は確保していない。

       - 旧 #include <ASRCAISim1/addons/{plugin name}/({sub directory name}/){header filename}>
       - 新 #include <ASRCAISim1/({sub directory name}/){header filename}>

@~english
@par core plugins auto search & import
    Packages whose setup.py registers ASRCAISim1.core_plugins entry point are treated as core plugins.
    Core plugins can be installed separately from ASRCAISim1, and they can be merged as if their attributes are also attributes of ASRCAISim1.
    The "unmerged" core plugins can also be accesible by ASRCAISim1.plugins.{plugin name}.
    
    Therefore you can access attributes of these plugins by:
    1. {plugin name}.{attribute name}
    2. ASRCAISim1.{attribute name}
    3. ASRCAISim1.plugins.{plugin name}.{attribute name}
   
    Although it has a risk of unintended behavior due to name conflict, we prioritized flexibility and accept that risk.

   C++ part of core plugins must use asrc::core namespace so that users can access them by the same as ASRCAISim1 itself.
   When you want to link a core plugin, add ASRCAISim1::{plugin name} next to ASRCAISim1::Core in target_link_libraries in CMakeLists.txt.
   If you feel bothered to add each plugin, you can instead write ASRCAISim1::All (Note: All also includes Core and you should remove ASRCAISim1::Core.)

   Auto import of core plugins can be disabled by setting environment variable "ASRCAISim1_disable_automatic_import_of_core_plugins" to 1.
   
@par Compatibility and migration from old version (1.x) to new version (2.x)
    - Replace import statements with "addons" by either of the following format.

        Note: For the time being, you can use old import statements via "addons" to keep backward compatibility.

        - (old) from ASRCAISim1.addons.{plugin name}(.{submodule name}) import {attribute name}
        - (new)
            1. from ASRCAISim1.plugins.{plugin name}(.{submodule name}) import {attribute name}  ※"addons"を"plugins"に置換
            2. from ASRCAISim1(.{submodule name}) import {attribute name}  ※"addons.{plugin name}"を削除
            3. from {plugin name}(.{submodule name}) import {attribute name}   ※"ASRCAISim1.addons"を削除
    - If you include headers of "addons" in C++ part, remove "addons/{plugin name}" and treat headers as if they are located directly under ASRCAISim1の階層を除き、ASRCAISim1 include directory.

        Note: C++ part requires rebuild due to the change of namespace etc., we don't keep backward compatibility.
        - (old) #include <ASRCAISim1/addons/{plugin name}/({sub directory name}/){header filename}>
        - (new) #include <ASRCAISim1/({sub directory name}/){header filename}>
"""
import importlib
import inspect
import os
import sys
import types
from typing import Union

class _ASRCDynamicModuleWrapper(types.ModuleType):
    """! @brief 透過的アクセス機能を提供するための、マージ後のモジュールを表すクラス。
    """

    def __init__(self, original: types.ModuleType, fullname: str):
        super().__init__(fullname, original.__doc__)
        assert not isinstance(original, _ASRCDynamicModuleWrapper)
        self.__name__ = fullname
        self._fullname = fullname
        self._original_name = original.__name__
        self._merged_modules = {original.__name__: original}
        self._validate()
    def _validate(self):
        original = self._merged_modules[self._original_name]
        for key in dir(original):
            if not key in [
                '__class__',
                '__dict__',
                '__name__',
            ]:
                orig_attr = getattr(original, key)
                if (
                    isinstance(orig_attr, types.ModuleType) and 
                    orig_attr.__name__.startswith(self._original_name)
                ):
                    suffix = orig_attr.__name__[len(self._original_name):]
                    orig_attr = _ASRCDynamicModuleWrapper(orig_attr, self._fullname + suffix)
                    sys.modules[self._fullname + suffix] = orig_attr
                setattr(self, key, orig_attr)
        if hasattr(original, '__path__'):
            self.__path__ = original.__path__
        else:
            self.__path__ = []
    def merge(self, other: types.ModuleType):
        # 既にマージ済ならば何もしない。
        if other.__name__ in self._merged_modules:
            return
        assert not isinstance(other, _ASRCDynamicModuleWrapper)
        #print("merged: ",self._original_name," <- ",other)
        self._merged_modules[other.__name__] = other
        if hasattr(self, '__all__'):
            all_attr = list(getattr(self, '__all__'))
            if hasattr(other, '__all__'):
                all_attr=list(set(all_attr+list(other.__all__)))
            else:
                other_all_attr = [key for key in dir(other) if not key.startswith('_')]
                all_attr=list(set(all_attr+list(other_all_attr)))
        else:
            if hasattr(other, '__all__'):
                all_attr = [key for key in dir() if not key.startswith('_')]
                all_attr=list(set(all_attr+list(other.__all__)))
            else:
                all_attr = None
        for key in dir(other):
            # '_'で始まるattributeと、core pluginに共通するattributeはマージ対象外とする。
            plugin_name = other.__name__.partition('.')[0]
            if key.startswith('_') or key in [
                'lib'+plugin_name,
                'factoryHelper',
            ]:
                continue
            other_attr = getattr(other, key)
            try:
                self_attr = super().__getattr__(key)
                if self_attr is other_attr:
                    continue
                if isinstance(self_attr, _ASRCDynamicModuleWrapper) and isinstance(other_attr, types.ModuleType):
                    self_attr.merge(other_attr)
                else:
                    # pybind11でMAKE_OPAQUEしたSTLコンテナに関するクラスは重複する可能性があるため、除外して無視する。
                    if not (
                        inspect.isclass(self_attr) and
                        inspect.isclass(other_attr) and
                        (
                            (self_attr.__name__.startswith('std::') and other_attr.__name__.startswith('std::')) or
                            (self_attr.__name__.startswith('ItemsView') and other_attr.__name__.startswith('ItemsView')) or
                            (self_attr.__name__.startswith('KeysView') and other_attr.__name__.startswith('KeysView')) or
                            (self_attr.__name__.startswith('ValuesView') and other_attr.__name__.startswith('ValuesView'))
                        )
                    ):
                        raise TypeError(
                            f"Only ModuleType can be merged. You tried to merge {other_attr}({type(other_attr)})" + 
                            f"into {self_attr}({type(self_attr)}) as '{key}' for the module '{self._fullname}''"
                        )
            except AttributeError:
                if (
                    isinstance(other_attr, types.ModuleType) and 
                    other_attr.__name__.startswith(other.__name__)
                ):
                    if not isinstance(other_attr, _ASRCDynamicModuleWrapper):
                        fullname = self._fullname + '.' + key
                        other_attr = _ASRCDynamicModuleWrapper(other_attr, fullname)
                        sys.modules[fullname] = other_attr
                setattr(self, key, other_attr)
        if not all_attr is None:
            setattr(self, '__all__', all_attr)

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError as e:
            if key.startswith('__'):
                raise e
            try:
                ret = importlib.import_module(self._fullname + '.' + key)
                return ret
            except ImportError:
                raise AttributeError(
                    f"module '{self._original_name}' has no attribute '{key}'"
                )
    def _add_submodule(self, name, module: types.ModuleType):
        # 既にマージ済ならば何もしない。
        assert not isinstance(module, _ASRCDynamicModuleWrapper)
        try:
            self_attr = super().__getattr__(name)
        except AttributeError:
            fullname = self._fullname + '.' + name
            ret = _ASRCDynamicModuleWrapper(module, fullname)
            sys.modules[fullname] = ret
            return ret
        assert isinstance(self_attr, _ASRCDynamicModuleWrapper)
        self_attr.merge(module)
        return self_attr

def __add_finder(_plugins):
    """! @brief 透過的アクセス機能を提供するための、独自のモジュール検索&読み込み機能の実装。
        sys.meta_pathを用いる。
    """

    class _ASRCLoader(importlib.abc.Loader):
        """! @brief 透過的アクセス機能を提供するための、独自のモジュール読み込みクラス。
        """

        @classmethod
        def try_to_create_spec(cls, fullname, base_spec):
            """! @brief モジュールの完全なパスがASRCAISim1で始まるものについて、このクラスで使用できるModuleSpecの生成を試みる。
                生成出来なかった場合は標準のfind_specで得られるModuleSpecをそのまま返す。
                @param [in] fullname モジュールの完全なパス
                @param [in] base_spec 標準のfind_specで得られたbase_spec(Noneの可能性もあり)
            """
            if fullname == 'ASRCAISim1':
                return base_spec
            prefix = 'ASRCAISim1.'
            if fullname.startswith(prefix):
                if fullname.startswith(prefix + 'plugins'):
                    #core plugin
                    if fullname == prefix + 'plugins':
                        return importlib.machinery.ModuleSpec(fullname, _ASRCLoader(base_spec, fullname))
                    else:
                        prefix = prefix + 'plugins.'
                        true_name = fullname[len(prefix):]
                        plugin_name = true_name.partition('.')[0]
                        true_spec = importlib.util.find_spec(true_name)
                        if _plugins.has_core_plugin(plugin_name):
                            return importlib.machinery.ModuleSpec(
                                fullname,
                                _ASRCLoader(base_spec, true_name),
                                origin = true_spec.origin
                            )
                        else:
                            #return base_spec
                            raise ImportError(
                                f'ASRCAISim1 does not have a core plugin {plugin_name}'
                            )
                elif fullname.startswith(prefix + 'addons'):
                    # For backward compatibility. It will be removed in the future.
                    # core plugin
                    if fullname == prefix + 'addons':
                        return importlib.machinery.ModuleSpec(fullname, _ASRCLoader(base_spec, fullname))
                    else:
                        prefix = prefix + 'addons.'
                        true_name = fullname[len(prefix):]
                        plugin_name = true_name.partition('.')[0]
                        true_spec = importlib.util.find_spec(true_name)
                        if _plugins.has_core_plugin(plugin_name):
                            return importlib.machinery.ModuleSpec(
                                fullname,
                                _ASRCLoader(base_spec, true_name),
                                origin = true_spec.origin
                            )
                        else:
                            #return base_spec
                            raise ImportError(
                                f'ASRCAISim1 does not have a core plugin {plugin_name}'
                            )
                else:
                    # submodule
                    if base_spec is not None:
                        # ASRCAISim1本体の中に見つかった場合
                        return importlib.machinery.ModuleSpec(
                            fullname,
                            _ASRCLoader(base_spec, fullname),
                            origin = base_spec.origin
                        )
                    else:
                        # ASRCAISim1本体の中に見つからなかった場合はcore pluginから探す
                        for plugin_name in _plugins.core_plugins():
                            true_name = plugin_name + '.' + fullname[len(prefix):]
                            ret = importlib.util.find_spec(true_name)
                            if ret is not None:
                                return importlib.machinery.ModuleSpec(
                                    fullname,
                                    _ASRCLoader(None, true_name),
                                    origin = ret.origin
                                )
                        return base_spec
            elif _plugins.has_core_plugin(fullname.partition('.')[0]):
                # core pluginを直接呼んだ場合
                return importlib.machinery.ModuleSpec(
                    fullname,
                    _ASRCLoader(base_spec, fullname),
                    origin = base_spec.origin if base_spec is not None else None
                )
            elif _plugins.has_user_plugin(fullname.partition('.')[0]) is not None:
                # user pluginを直接呼んだ場合
                return importlib.machinery.ModuleSpec(
                    fullname,
                    _ASRCLoader(base_spec, fullname),
                    origin = base_spec.origin if base_spec is not None else None
                )
            else:
                # 階層の途中でuser pluginのコピーが置かれている可能性がある。
                # その場合は、build identifierを判別して二重ロードとならないように読み込む
                parent = []
                found = None
                basename, rem = None, fullname
                while found is None and rem != '':
                    if basename is not None:
                        parent.append(basename)
                    basename, _, rem = rem.partition('.')
                    found = _plugins.has_user_plugin(basename)
                if found is None:
                    # user pluginを含まない
                    return base_spec
                parent = '.'.join(parent)
                if rem == '':
                    true_name = basename
                else:
                    true_name = basename + '.' + rem
                plugin_name, build_identifier = found
                return importlib.machinery.ModuleSpec(
                    fullname,
                    _ASRCLoader(base_spec, true_name),
                    origin = base_spec.origin if base_spec is not None else None
                )

        def __init__(self,
                     base_spec,
                     true_name,
                     wrap = True
                    ):
            self.true_name = true_name
            self.base_spec = base_spec
            self.wrap = wrap
            self.need_exec_module = False

        def create_module(self, spec):
            fullname = spec.name
            if fullname in sys.modules:
                ret = sys.modules[fullname]
                return ret
            if fullname.startswith('ASRCAISim1.'):
                if fullname.startswith('ASRCAISim1.plugins'):
                    # core plugin
                    if fullname == 'ASRCAISim1.plugins':
                        ret = _plugins.get_as_module()
                        sys.modules[fullname] = ret
                        return ret
                    else:
                        ret = importlib.import_module(self.true_name)
                        ret = _plugins._try_to_register_core_plugin_or_submodule(ret, fullname, self.true_name)
                        sys.modules[fullname] = ret
                        return ret
                elif fullname.startswith('ASRCAISim1.addons'):
                    # For backward compatibility. It will be removed in the future.
                    # core plugin
                    if fullname == 'ASRCAISim1.addons':
                        ret = importlib.import_module('ASRCAISim1.plugins')
                        sys.modules[fullname] = ret
                        return ret
                    else:
                        ret = importlib.import_module('ASRCAISim1.plugins.' + self.true_name)
                        sys.modules[fullname] = ret
                        return ret
                else:
                    # submodules
                    submodule_name = fullname[len('ASRCAISim1.'):].partition('.')[0]
                    parent, _, child = fullname.rpartition('.')
                    if self.base_spec is not None:
                        # ASRCAISim1本体の中に見つかった場合
                        ret = importlib.util.module_from_spec(self.base_spec)
                        ret = _ASRCDynamicModuleWrapper(ret, fullname)
                        self.need_exec_module = True
                    else:
                        # ASRCAISim1本体の中に見つからなかった場合はcore pluginから探す
                        # ASRCAISim1.plugins.<true name>でインポートして↑の分岐を呼ぶ
                        ret = importlib.import_module('ASRCAISim1.plugins.' + self.true_name)
                    sys.modules[fullname] = ret
                    return ret
            elif _plugins.has_core_plugin(fullname.partition('.')[0]):
                # core pluginを直接呼んだ場合
                # 通常のインポートを行った上で、_pluginsに関する操作を追加で行う。
                _ASRCPathFinder._disabled.add(fullname)
                ret = importlib.import_module(fullname) #sys.modules[fullname]はここでセットされるので上書き不要。
                _ASRCPathFinder._disabled.remove(fullname)

                alias_name = 'ASRCAISim1.plugins.' + fullname
                importlib.import_module(alias_name) # ASRCAISim1.pluginsとしてもインポートすることで_pluginsの操作を呼ぶ。
                return ret
            elif (found := _plugins.has_user_plugin(fullname.partition('.')[0])) is not None:
                # user pluginを直接呼んだ場合
                # 通常のインポートを行った上で、_pluginsに関する操作を追加で行う。
                plugin_name, build_identifier = found
                remainder = fullname.partition('.')[2]
                if build_identifier == '':
                    # build idenifierなし
                    _ASRCPathFinder._disabled.add(fullname)
                    ret = importlib.import_module(fullname) #sys.modules[fullname]はここでセットされるので上書き不要。
                    _ASRCPathFinder._disabled.remove(fullname)
                    if remainder == '':
                        # plugin本体
                        _plugins._register_user_plugin(plugin_name, build_identifier, ret)
                    return ret
                else:
                    # build idenifier付きで呼ばれた場合は、既に_pluginsで管理されていることを前提とする。
                    found_plugin = _plugins.get(plugin_name, build_identifier)
                    if remainder == '':
                        # plugin本体の場合はそのまま返す。
                        ret = found_plugin
                    else:
                        cpp_module_name = 'lib'+plugin_name
                        if remainder.startswith(cpp_module_name) and not remainder.startswith(cpp_module_name+build_identifier):
                            # C++モジュールはbuild identifierを付与した方を読み込む
                            remainder = remainder.replace(cpp_module_name,cpp_module_name+build_identifier)
                        true_name = found_plugin.__name__ + '.' + remainder
                        parent = true_name.rpartition('.')[0]
                        _ASRCPathFinder._disabled.add(true_name)
                        ret = importlib.import_module(true_name, parent) #sys.modules[fullname]はここでセットされるので上書き不要。
                        _ASRCPathFinder._disabled.remove(true_name)
                    return ret
            else:
                # 階層の途中でuser pluginのコピーが見つかった場合。
                parent = []
                found = None
                basename, rem = None, fullname
                while found is None and rem != '':
                    if basename is not None:
                        parent.append(basename)
                    basename, _, rem = rem.partition('.')
                    found = _plugins.has_user_plugin(basename)
                parent = '.'.join(parent)
                if rem == '':
                    true_name = basename
                else:
                    true_name = basename + '.' + rem
                # 他のパッケージのサブパッケージとしてuser pluginを含んでいた場合は、
                # 同じbuild identifierのものが読込済ならばそれを使用し、未読込ならば新たにインポートする。
                # このとき、self.base_specは末端のサブモジュールを指すため、plugin本体のspecを取得してbuild identifierの情報を得る。
                # basenameまでで止めて改めてfind_specを行えば、同じ分岐を通して_ASRCLoaderを持つModuleSpecが得られる。
                # そのspecのoriginからBuildIdentifier.txtのディレクトリを特定できる。
                plugin_name, build_identifier = found
                _ = _plugins.get(plugin_name) # build identifier無し版も事前に必ずインポートしておく
                plugin_spec = importlib.util.find_spec(parent + '.' + basename).loader.base_spec
                build_identifier = getBuildIdentifier(plugin_spec)
                if (found_plugin := _plugins.get(plugin_name, build_identifier)) is not None:
                    # 読込済
                    if rem == '':
                        ret = found_plugin
                        getattr(ret, 'factoryHelper').addAliasPackageName(parent)
                    else:
                        cpp_module_name = 'lib'+plugin_name
                        if rem.startswith(cpp_module_name) and not rem.startswith(cpp_module_name+build_identifier):
                            # C++モジュールはbuild identifierを付与した方を読み込む
                            rem = rem.replace(cpp_module_name,cpp_module_name+build_identifier)
                        true_name = found_plugin.__name__ + '.' + rem
                        parent = true_name.rpartition('.')[0]
                        _ASRCPathFinder._disabled.add(true_name)
                        ret = importlib.import_module(true_name, parent)
                        _ASRCPathFinder._disabled.remove(true_name)
                else:
                    # 未読込
                    # suffixとしてbuild identifierを付与した別パッケージとしてインポートする。
                    assert rem == ''
                    self.base_spec.name = plugin_name + build_identifier
                    self.base_spec.loader.name=self.base_spec.name
                    ret=importlib.util.module_from_spec(self.base_spec)
                    sys.modules[plugin_name + build_identifier] = ret
                    _plugins._register_user_plugin(plugin_name, build_identifier, ret)
                    self.base_spec.loader.exec_module(ret)
                    getattr(ret, 'factoryHelper').addAliasPackageName(parent)
                sys.modules[fullname] = ret
                return ret

        def exec_module(self, module):
            if self.need_exec_module:
                self.base_spec.loader.exec_module(module)
                module._validate()

        def get_code(self, fullname):
            if self.need_exec_module:
                return self.base_spec.loader.get_code(fullname)
            
    class _ASRCPathFinder(importlib.machinery.PathFinder):
        """! @brief 透過的アクセス機能を提供するための、独自のモジュール検索クラス。
        """
        _disabled = set()
        @classmethod
        def find_spec(cls, fullname, path, target=None):
            if fullname in cls._disabled:
                return None
            for meta_path in sys.meta_path:
                if meta_path != cls:
                    base_spec = meta_path.find_spec(fullname, path, target)
                    if base_spec is not None:
                        break
            return _ASRCLoader.try_to_create_spec(fullname, base_spec)

    sys.meta_path.insert(0,_ASRCPathFinder)

class PluginManager:
    """! @brief submoduleとpluginを管理するための内部クラス。
    """
    def __init__(self):
        if sys.version_info < (3, 10):
            from importlib_metadata import entry_points
        else:
            from importlib.metadata import entry_points
        self.__submodules = {ep.name: ep for ep in entry_points(group=__name__+'.submodules')}
        ignore = [p for p in os.getenv('ASRCAISim1_ignore_core_plugin','').split(';') if p != '']
        if '__all__' in ignore:
            self.__core_plugins = {}
        else:
            self.__core_plugins = {ep.name: ep for ep in entry_points(group=__name__+'.core_plugins') if not ep.name in ignore}
        self.__user_plugins = {ep.name: ep for ep in entry_points(group=__name__+'.user_plugins')}
        self.__loaded = {} # {"module_name": {"build_identifier": module}}
        self.__failed = {}
        self._need_lazy_load_of_core_plugins = None
        class _ASRCPluginsModuleType(types.ModuleType):
            def __getattr__(_self, key):
                try:
                    return super().__getattr__(key)
                except AttributeError as e:
                    try:
                        if self.has_submodule(key) or self.has_core_plugin(key):
                            ret = self.get(key) #Call PluginManager's get
                            if ret is None:
                                raise e
                            else:
                                setattr(_self, key, ret)
                                return ret
                        else:
                            raise e
                    except ImportError as i:
                        raise i
                    raise e
        self.__module = _ASRCPluginsModuleType('ASRCAISim1.plugins')
        setattr(self.__module, '__package__', 'ASRCAISim1.plugins')
        setattr(self.__module, '__path__', globals()['__path__'])
    def __contains__(self, name):
        return name in self.__core_plugins or name in self.__user_plugins or name in self.__loaded
    def has_core_plugin(self, name):
        """! @brief nameという名を持つcore pluginの存在を判定する
        """
        return name in self.__core_plugins
    def has_user_plugin(self, name):
        """! @brief nameという名を持つuser pluginの存在を判定する

            @details
                user pluginはbuild identifierをsuffixとして付加したものも判定する。
                一致するものがあればplugin nameとbuild identifierに区切ったタプルを返し、
                なければNoneを返す。
        """
        if name in self.__user_plugins:
            return name, ''
        for plugin_name in self.__user_plugins:
            for build_identifier in self.__loaded.get(plugin_name, {}):
                if plugin_name + build_identifier == name:
                    return plugin_name, build_identifier
        return None
    def has_submodule(self, name):
        return name in self.__submodules
    def get_entrypoint(self, name):
        if name in self.__core_plugins:
            return self.__core_plugins[name]
        elif name in self.__user_plugins:
            return self.__user_plugins[name]
        else:
            return None
    def get(self, name, build_identifier=''):
        if name in self.__loaded:
            p = self.__loaded[name]
            if build_identifier in p:
                if p[build_identifier] is None:
                    # EntryPointからインポートされたものと同じbuild identifierの場合はここでLoaderを通さずに別名でインポート
                    p[build_identifier] = import_user_plugin_with_build_identifier_from_file_location(
                        name,
                        os.path.dirname(os.path.dirname(p[''].__spec__.origin)),
                        check=False
                    )
                return p[build_identifier]
            else:
                return None
        self.__check_failure(name)
        if name in self.__core_plugins:
            ret = self.try_load(name)
            self.__check_failure(name)
            return ret
        elif name in self.__user_plugins:
            self.try_load(name)
            self.__check_failure(name)
            return self.get(name, build_identifier)
        else:
            return None
    def __getattr__(self, key):
        if key in dir(self):
            return super().__getattr__(self, key)
        ret = self.get(key)
        if ret is None:
            raise AttributeError(f"There is no loaded plugin named '{key}'")
        return ret
    def __getitem__(self, key):
        ret = self.get(key)
        if ret is None:
            raise AttributeError(f"There is no loaded plugin named '{key}'")
        return ret
    def try_load(self, name):
        if name in self.__loaded:
            return self.__loaded[name]['']
        self.__check_failure(name)
        if name in self.__core_plugins:
            # entry pointをロードすると_ASRCLoader側から_register_core_pluginを呼ばれるので、そちらでself.__loadedに追加する。
            try:
                plugin = self.__core_plugins[name].load()
                if not name in self.__loaded:
                    # ただし、ASRCAISim1より前にpluginを直接インポートしていた場合は、entry pointのloadで_ASRCLoaderを通らない。
                    self._register_core_plugin(name, plugin)
                return self.__loaded[name]['']
            except Exception as ex:
                import traceback
                self.__failed[name] = "The reason for '" + name + "'is an exception below:\n" + "".join(traceback.format_exc())
                return None
        elif name in self.__user_plugins:
            # entry pointをロードすると_ASRCLoader側から_register_user_pluginを呼ばれるので、そちらでself.__loadedに追加する。
            try:
                self.__user_plugins[name].load()
                return self.__loaded[name]['']
            except Exception as ex:
                import traceback
                self.__failed[name] = "The reason for '" + name + "'is an exception below:\n" + "".join(traceback.format_exc())
                return None
        else:
            self.__failed[name] = "The reason for '" + name + "'is: " + "You might have tried to import an uninstalled plugin '" + name + "'"
            return None
    def __check_failure(self, name):
        if name in self.__failed:
            raise ImportError(
                "Importing the plugin '" + name + "' failed.\n" + self.__failed[name]
            )
    def core_plugins(self):
        return [name for name in self.__core_plugins]
    def user_plugins(self):
        return [name for name in self.__user_plugins]
    def loaded(self):
        return [name for name in self.__loaded]
    def _register_core_plugin(self, plugin_name, plugin):
        if plugin_name in self.__loaded:
            # 既に登録済ならば同一性の判定だけ行い、そのまま返す。
            assert plugin is self.__loaded[plugin_name]['']
            return plugin
        fullname = __name__ + '.' + plugin_name
        self.__loaded[plugin_name] = {'': plugin}
        for key in dir(plugin):
            if key.startswith('_') or key in [
                'lib'+plugin_name,
                'factoryHelper',
            ]:
                continue
            attr = getattr(plugin, key)
            if isinstance(attr, types.ModuleType):
                if attr.__name__.startswith(plugin_name):
                    # submoduleは同名のsubmoduleにマージ
                    self._register_dynamic_submodule(key, attr)
                else:
                    # submodule以外はよそのモジュールをインポートしたものなので、マージ不要
                    if not key in globals():
                        globals()[key] = attr
            else:
                globals()[key] = attr
        return plugin 
        setattr(self.__module, plugin_name, plugin)
    def _register_user_plugin(self, plugin_name, build_identifier, plugin):
        # 既に登録済ならばエラーとする。
        if not plugin_name in self.__loaded:
            self.__loaded[plugin_name] = {}
        if self.__loaded[plugin_name].get(build_identifier, None) is not None:
            raise ValueError(
                f"The user plugin '{plugin_name}' with the build identifier '{build_identifier}' is already registered."
            )
        self.__loaded[plugin_name][build_identifier] = plugin
        if build_identifier == '':
            # 読み込んだpluginがbuild identifierを持っていれば、その値も可能な候補として登録しておく(インポートはまだしないでおく)
            plugin_spec = plugin.__spec__
            build_identifier = getBuildIdentifier(plugin_spec)
            if build_identifier != '':
                self.__loaded[plugin_name][build_identifier] = None
    def _register_dynamic_submodule(self, submodule_name, submodule):
        fullname = __name__ + '.' + submodule_name
        if fullname in sys.modules:
            if not (sys.modules[fullname] is submodule):
                sys.modules[fullname].merge(submodule)
        else:
            self.__submodules[submodule_name] = _ASRCDynamicModuleWrapper(submodule, fullname)
            sys.modules[fullname] = self.__submodules[submodule_name]
        if not submodule_name in self.__loaded:
            self.__loaded[submodule_name] = {'': sys.modules[fullname]}
        return sys.modules[fullname]
    def _try_to_register_core_plugin_or_submodule(self, pure_module, fullname, true_name):
        # core pluginは別名(ASRCAISim1.plugins)によるインポートを可能としているため、
        # プラグイン内での自分自身のインポート文の書き方次第では多重に呼ばれ得る。
        # そのような場合にも分岐先の各処理は一度だけ登録/マージが実行される。
        plugin_name, _, remainder = true_name.partition('.')
        submodule_name, _, remainder = remainder.partition('.')
        if submodule_name == '':
            # plugin itself
            # 未登録ならば_pluginsに登録する。
            return _plugins._register_core_plugin(plugin_name, pure_module)
        elif remainder == '':
            # submodule
            # 未登録ならば_pluginsに登録する。
            return _plugins._register_dynamic_submodule(submodule_name, pure_module)
        else:
            # nested submodule
            # ここに到達するまでにparentはインポート済かつ_ASRCDynamicModuleWrapperになっているはず。
            # そのparentの_add_submoduleを呼ぶ。マージ済ならば何も起こらない。
            parent, _, child = fullname.rpartition('.')
            return sys.modules[parent]._add_submodule(child, pure_module)
    def _load_submodules(self):
        self.__loaded['core'] = {'': self.__submodules['core'].load()}
        globals()['core'] = self.__loaded['core']['']
        for name,ep  in self.__submodules.items():
            if name == 'core':
                continue
            self.__loaded[name] = {'': ep.load()}
            sys.modules['ASRCAISim1.' + name] = self.__loaded[name]['']
            globals()[name] = self.__loaded[name]['']
    def _load_all_core_plugins(self):
        for name in self.__core_plugins:
            if name in sys.modules and not name in self.__loaded:
                # _load_all_core_pluginsが呼ばれた時点で既にsys.modulesにpluginが存在している場合、
                # ASRCAISim1より前に直接インポートされていることを意味し、
                # その__init__.pyでASRCAISim1.common_init_for_pluginを呼ぶためASRCAISim1のインポートが行われたという状況である。
                # そのため、circular importを回避するため、_load_all_core_pluginsはcommon_init_for_pluginの中で呼ぶ。
                self._need_lazy_load_of_core_plugins = name
                return
        for name in self.__core_plugins:
            self.try_load(name)
            self.__check_failure(name)
        return None, False
    def get_as_module(self):
        return self.__module

_plugins = PluginManager()
__add_finder(_plugins)

def getBuildIdentifier(plugin_or_spec_or_origin: Union[types.ModuleType, importlib.machinery.ModuleSpec, str]):
    """! @brief 指定したプラグインのBuildIdentifierを返す。
    """
    if isinstance(plugin_or_spec_or_origin, types.ModuleType):
        pluginDir = os.path.dirname(plugin_or_spec_or_origin.__spec__.origin)
    elif isinstance(plugin_or_spec_or_origin, importlib.machinery.ModuleSpec):
        pluginDir = os.path.dirname(plugin_or_spec_or_origin.origin)
    else: # isinstance(plugin_or_spec_or_origin, str):
        pluginDir = os.path.dirname(plugin_or_spec_or_origin)
    infoFilePath=os.path.join(pluginDir,"BuildIdentifier.txt")
    if os.path.exists(infoFilePath):
        return open(infoFilePath,"r").readline().strip()
    else:
        return ""

def import_user_plugin_with_build_identifier_from_file_location(plugin_name, directory, check=True):
    """! @brief directoryにあるplugin_nameという名のuser pluginをBuildIdentifier付きで呼び出す。
    """
    origin = os.path.join(directory, plugin_name, '__init__.py')
    if not os.path.exists(origin):
        raise FileNotFoundError(f"{origin}")
    build_identifier = getBuildIdentifier(origin)
    if check and _plugins.get(plugin_name, build_identifier) is not None:
        raise ImportError(f"A user plugin '{plugin_name}' with the build identifier '{build_identifier}' is already imported.")
    spec = importlib.util.spec_from_file_location(plugin_name, origin)
    spec.name = plugin_name + build_identifier
    spec.loader.name=spec.name
    module = importlib.util.module_from_spec(spec)
    sys.modules[plugin_name + build_identifier] = module
    _plugins._register_user_plugin(plugin_name, build_identifier, module)
    spec.loader.exec_module(module)
    return module

def common_init_for_plugin(plugin: types.ModuleType, globals_dict):
    """!
        @~japanese-en
        @brief ASRCAISim1プラグインの共通初期化処理
            1. 依存するプラグインをインポート(し、通常のインポートと同様に自身のattributeに追加)
            2. C++モジュールがあればインポートしてそのattributeを自身のattributeに追加
            3. FactoryHelperインスタンスを'factoryHelper'としてattributeに追加

        @~english
        @brief Common initialization for ASRCAISim1 plugins
            1. Import plugins required by this plugin (and set as the plugin's attributes as usual.)
            2. If a C++ module exists, import it and set its attributes as the plugin's attributes.
            3. Add a FactoryHelper instance as 'factoryHelper' attribute.
    """
    import glob
    import json
    pluginDir = os.path.dirname(plugin.__spec__.origin)
    fullPackageName = plugin.__name__
    selfPackageName = fullPackageName.rpartition('.')[2] # == <plugin name> + <build identifier>
    selfPluginName = os.path.basename(pluginDir)

    if os.path.exists(os.path.join(pluginDir,"PluginDependencyInfo.json")):
        pluginBuildIdentifiers = json.load(open(os.path.join(pluginDir,"PluginDependencyInfo.json"),"r"))
    else:
        pluginBuildIdentifiers = {}

    # (1) 依存するプラグインをインポート
    # (1) Import plugins required by this plugin
    dependency = {}

    if selfPackageName == selfPluginName:
        # 自身がbuild identifierなしでインポートされた場合は、依存pluginもbuild identifierなしのものを用いる。
        for pluginName, pluginBuildIdentifier in pluginBuildIdentifiers.items():
            dependency[pluginName] = importlib.import_module(pluginName)
    else:
        # 自身がbuild identifier付きでインポートされた場合は、依存pluginもbuild identifier付きのものを用いる。
        for pluginName, pluginBuildIdentifier in pluginBuildIdentifiers.items():
            # まずはbuild identifierなしのものをインポート。
            dependency[pluginName] = importlib.import_module(pluginName)
            try:
                # インポート済のものを探す。
                dependency[pluginName] = importlib.import_module(pluginName+pluginBuildIdentifier)
            except:
                # インポート済でなければ同ディレクトリから探す。
                dependency[pluginName] = import_user_plugin_with_build_identifier_from_file_location(
                    pluginName,
                    os.path.dirname(pluginDir)
                )
                if getBuildIdentifier(dependency[pluginName]) != pluginBuildIdentifier:
                    raise ImportError(
                        f"Build identifier mismatch. required={pluginBuildIdentifier}, found={getBuildIdentifier(dependency[pluginName])}"
                    )


    def isCppPlugin():
        return len(glob.glob(os.path.join(pluginDir,"lib"+selfPackageName+"*.*")))>0

    def loadCppModule():
        libName = "lib" + selfPackageName

        if(os.name=="nt"):
            pyd=os.path.join(pluginDir,libName+".pyd")
            if(not os.path.exists(pyd) or os.path.getsize(pyd)==0):
                print("Info: Maybe the first run after install. A hardlink to a dll will be created.")
                if(os.path.exists(pyd)):
                    os.remove(pyd)
                dll=os.path.join(pluginDir,libName+".dll")
                if(not os.path.exists(dll)):
                    dllName = selfPackageName
                    dll=os.path.join(pluginDir,dllName+".dll")
                if(not os.path.exists(dll)):
                    raise FileNotFoundError("There is no "+libName+".dll or "+dllName+".dll.")
                import subprocess
                subprocess.run([
                    "fsutil",
                    "hardlink",
                    "create",
                    pyd,
                    dll
                ])
        try:
            lib = importlib.import_module(fullPackageName+"."+libName)
            for pluginName, module in dependency.items():
                lib.factoryHelper.addDependency(pluginName, module.factoryHelper)
        except ImportError as e:
            if(os.name=="nt"):
                print("Failed to import the module. If you are using Windows, please make sure that: ")
                print('(1) If you are using conda, CONDA_DLL_SEARCH_MODIFICATION_ENABLE should be set to 1.')
                print('(2) dll dependencies (such as nlopt) are located appropriately.')
            raise e
        return lib

    if isCppPlugin():
        # このディレクトリをパスに追加
        # Add this directory to the path.
        os.environ['PATH'] = pluginDir + os.pathsep + os.environ['PATH']

        # (2) C++モジュールをインポートしてそのattributeを自身のattributeに追加
        # (2) Import C++ module and set its attributes as the plugin's attributes.
        lib = loadCppModule()
        if hasattr(lib, '__all__'):
            all_attr = lib.__all__ 
        else:
            all_attr = [
                key for key in dir(lib)
                if not key.startswith('_')
            ]
        for key in all_attr:
            if key != 'lib':
                globals_dict[key] = getattr(lib, key)
        if 'lib' in all_attr:
            lib = getattr(lib, 'lib')
        factoryHelper=lib.factoryHelper

        # このプラグインのcmakeコンフィグへのパスを返すメソッドを追加
        # add method to return cmake config path for this plugin.
        def get_cmake_dir() -> str:
            return os.path.abspath(os.path.join(pluginDir,'share/cmake')).replace(os.path.sep,'/')
        globals_dict['get_cmake_dir']=get_cmake_dir
    else:
        factoryHelper=core.FactoryHelper(selfPackageName)
    factoryHelper.validate(fullPackageName, os.path.abspath(pluginDir))
    # (1) インポートした依存プラグインをattributeに追加
    # (1) Add imported depending plugins as attributes.
    for name, module in dependency.items():
        globals_dict[name]=module

    # (3) FactoryHelperインスタンスを'factoryHelper'としてattributeに追加
    # (3) Add a FactoryHelper instance as 'factoryHelper' attribute.
    globals_dict['factoryHelper'] = factoryHelper

    # ASRCAISim1より前にpluginを直接インポートしていた場合はここでcore pluginsのインポートを行う。
    if _plugins._need_lazy_load_of_core_plugins == selfPluginName:
        # この場合は_ASRCLoaderを経由していないのでまずはASRCAISim1.pluginsとしての登録をここで完了させる。
        alias_name = 'ASRCAISim1.plugins.' + selfPluginName
        alias = _plugins._try_to_register_core_plugin_or_submodule(plugin, alias_name, selfPluginName)
        sys.modules[alias_name] = alias
        _plugins._need_lazy_load_of_core_plugins = None
        if not os.getenv('ASRCAISim1_disable_automatic_import_of_core_plugins', 'False').lower() in ('true', '1', 't'):
            _plugins._load_all_core_plugins()

# PEP 562 Module __getattr__
def __getattr__(key):
    if key == 'addons': # For backward compatibility. It will be removed in the future.
        return globals()['plugins']
    elif key in globals():
        return globals()[key]
    else:
        import importlib
        try:
            return importlib.import_module(__name__ + '.' + key)
        except:
            raise AttributeError(f"module '{__name__}' has no attribute '{key}'")

_plugins._load_submodules()

if not os.getenv('ASRCAISim1_disable_automatic_import_of_core_plugins', 'False').lower() in ('true', '1', 't'):
    _plugins._load_all_core_plugins()

def get_cmake_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__),'share/cmake')).replace(os.path.sep,'/')

from .core import * # Just for backward compatibility. It will be removed or moved to plugins in the future.
from .common import * # Just for backward compatibility. It will be removed or moved to plugins in the future.
from .GymManager import * # Just for backward compatibility. It will be removed or moved to plugins in the future.
core.factoryHelper.addDefaultModelsFromJsonFile(os.path.join(os.path.dirname(__file__),"./config/Agent.json"))
core.factoryHelper.addDefaultModelsFromJsonFile(os.path.join(os.path.dirname(__file__),"./config/Asset.json"))
core.factoryHelper.addDefaultModelsFromJsonFile(os.path.join(os.path.dirname(__file__),"./config/Controller.json"))
core.factoryHelper.addDefaultModelsFromJsonFile(os.path.join(os.path.dirname(__file__),"./config/CoordinateReferenceSystem.json"))
core.factoryHelper.addDefaultModelsFromJsonFile(os.path.join(os.path.dirname(__file__),"./config/RulerAndReward.json"))
core.factoryHelper.addDefaultModelsFromJsonFile(os.path.join(os.path.dirname(__file__),"./config/Viewer.json"))
