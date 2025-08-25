# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

def find_core_plugins():
    import sys
    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points
    return [get_plugin_directory(ep.name) for ep in entry_points(group='ASRCAISim1.core_plugins')]

def get_plugin_directory(name):
    import os
    import importlib
    try:
        m = importlib.import_module(name)
        return os.path.abspath(os.path.dirname(m.__file__))
    except ImportError as e:
        print(f'Required plugin {name} could not be imported!')
        raise e

if __name__ == '__main__':
    found = find_core_plugins()
    if len(found) > 0:
        print(';'.join(find_core_plugins()), end='')
    else:
        print('')
