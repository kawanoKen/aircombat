# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

def clean_at(base_dir: str = ""):
    import os
    import sys
    from setuptools import setup, find_packages
    from setuptools.build_meta import no_install_setup_requires
    from .setup_command import clean
    sys.argv = [
            *sys.argv[:1],
            "clean"
    ]
    os.chdir(base_dir or os.curdir)
    #clean_cmd = clean
    package_name = find_packages(os.curdir)[0].partition('.')[0]
    os.environ['ASRCAISim1_ignore_core_plugin'] = "package_name"
    with no_install_setup_requires():
        #exec(u"from setuptools import setup; setup(cmdclass={'clean': clean})", locals())
        exec(u"from setuptools import setup; setup()", locals())

def clean_at_standalone(base_dir: str = ""):
    import os
    import sys
    import glob
    import shutil
    from setuptools import find_packages
    package_name = find_packages(base_dir)[0].partition('.')[0]
    build_dir=os.path.join(base_dir, "build")
    dist_dir=os.path.join(base_dir, "dist")
    shutil.rmtree(build_dir,True)
    shutil.rmtree(dist_dir,True)
    files=[
        os.path.join(base_dir,"MANIFEST.in"),
    ]
    for f in files:
        if os.path.exists(f):
            os.remove(f)
    shutil.rmtree(os.path.join(base_dir,package_name+".egg-info"),True)
    for d in glob.glob(os.path.join(base_dir,"**/__pycache__"),recursive=True):
        shutil.rmtree(d,True)
    if (
        len(glob.glob(os.path.join(base_dir,"builder*.*"))) > 0 and
        os.path.exists(os.path.join(base_dir,"CMakeLists.txt"))
    ):
        module_dir=os.path.join(base_dir, package_name)
        files=[
            os.path.join(module_dir, "BuildIdentifier.txt"),
        ]
        for f in files:
            if os.path.exists(f):
                os.remove(f)
        module_include_dir=os.path.join(module_dir, "include")
        shutil.rmtree(module_include_dir,True)
        module_third_party_dir=os.path.join(module_dir, "thirdParty")
        shutil.rmtree(module_third_party_dir,True)
        for f in glob.glob(os.path.join(module_dir,"**/*"+package_name+"*.*"),recursive=True):
            if os.path.splitext(f)[1] in [".so",".lib",".dll",".dll.a",".pyd"]:
                os.remove(f)
    msys = os.getenv('ASRCAISim1_build_with_msys', 'False').lower() in ('true', '1', 't')
    if(os.name=="nt" and not msys):
        cleaner=os.path.join(base_dir, "thirdParty","scripts","cleaner.bat")
        if os.path.exists(cleaner):
            import subprocess
            subprocess.check_call([
                cleaner,
            ])
    else:
        cleaner=os.path.join(base_dir, "thirdParty","scripts","cleaner.sh")
        if os.path.exists(cleaner):
            import subprocess
            subprocess.check_call([
                "bash",
                cleaner,
            ])



if __name__ == '__main__':
    import sys
    for arg in sys.argv[1:]:
        clean_at(arg)
