# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import sys
import os
import shutil
from setuptools.command.sdist import sdist as sdist_orig
from setuptools.command.build import build as build_orig
from setuptools.command.install import install as install_orig
from distutils.command.clean import clean as clean_orig
from setuptools import setup,find_packages
import glob

packageName="ASRCAISim1"

def gather_manifest(base_dir):
        manifest=open("MANIFEST.in","w",encoding="utf-8")
        with open("MANIFEST.in.fragment","r",encoding="utf-8") as fragment:
            manifest.write(fragment.read()+"\n")
        manifest.close()

def clean_dst_directory_for_cpp(base_dir, package_name, cpp_module_name=None):
    # Cleans destination package directory. Build directory is kept as-is.
    # This is for cleaning auto-generated files by CMake install command.
    module_dir=os.path.join(base_dir, package_name)
    module_include_dir=os.path.join(module_dir, "include")
    shutil.rmtree(module_include_dir,True)
    module_third_party_dir=os.path.join(module_dir, "thirdParty")
    shutil.rmtree(module_third_party_dir,True)
    if cpp_module_name is None:
        cpp_module_name = package_name
    for f in glob.glob(os.path.join(module_dir,"**/*"+cpp_module_name+"*.*"),recursive=True):
        if os.path.splitext(f)[1] in [".so",".lib",".dll",".dll.a",".pyd"]:
            os.remove(f)

class sdist(sdist_orig):
    def run(self):
        base_dir = os.path.dirname(__file__)
        gather_manifest(base_dir)
        clean_dst_directory_for_cpp(base_dir, packageName, "Core")
        sdist_orig.run(self)

class build(build_orig):
    def run(self):
        base_dir = os.path.dirname(__file__)
        gather_manifest(base_dir)
        clean_dst_directory_for_cpp(base_dir, packageName, "Core")

        self.debug = os.getenv('ASRCAISim1_enable_debug_build', 'False').lower() in ('true', '1', 't')
        self.build_config = "Debug" if self.debug else "Release"
        self.msys = os.getenv('ASRCAISim1_build_with_msys', 'False').lower() in ('true', '1', 't')

        import sys
        import subprocess
        import numpy
        prefix=sys.base_prefix
        py_ver="%d.%d" % sys.version_info[:2]
        os.environ["PythonLibsNew_FIND_VERSION"]=py_ver
        if(os.name=="nt"):
            python_include_dir=os.path.dirname(glob.glob(os.path.join(prefix,"include","Python.h"))[0])
            python_lib_dir=os.path.join(prefix,"libs")
        else:
            python_include_dir=os.path.dirname(glob.glob(os.path.join(prefix,"include/python"+py_ver+sys.abiflags+"/Python.h"))[0])
            python_lib_dir=os.path.join(prefix,"lib")
        numpy_include_dir=numpy.get_include()
        if(os.name=="nt"):
            if(self.msys):
                base_cmd_args = ["bash","./builder.sh"]
            else:
                base_cmd_args = [".\\builder.bat"]
        else:
            base_cmd_args = ["bash", "./builder.sh"]

        common_cmd_args = [
            self.build_config,
            python_include_dir.replace(os.path.sep,'/'),
            python_lib_dir.replace(os.path.sep,'/'),
            numpy_include_dir.replace(os.path.sep,'/'),
        ]

        cmd_args = base_cmd_args + common_cmd_args
        subprocess.check_call(cmd_args)
        if(os.name=="nt"):
            dummy=os.path.join(os.path.dirname(__file__),packageName,"libCore.pyd")
            if(os.path.exists(dummy)):
                os.remove(dummy)
            f=open(dummy,"w",encoding="utf-8") #Dummy
            f.close()
        build_orig.run(self)

class install(install_orig):
    def run(self):
        install_orig.run(self)
        if(os.name=="nt"):
            dummy=os.path.join(os.path.dirname(__file__),packageName+"/libCore.pyd")
            if(os.path.exists(dummy)):
                os.remove(dummy) #Dummy

class clean(clean_orig):
    def run(self):
        clean_orig.run(self)
        base_dir=os.path.dirname(__file__)
        build_dir=os.path.join(base_dir, "build")
        dist_dir=os.path.join(base_dir, "dist")
        shutil.rmtree(build_dir,True)
        for dist in glob.glob(os.path.join(dist_dir, packageName+"-*")):
            os.remove(dist)
        files=[
            os.path.join(base_dir,"MANIFEST.in"),
        ]
        for f in files:
            if os.path.exists(f):
                os.remove(f)
        shutil.rmtree(os.path.join(base_dir,packageName+".egg-info"),True)
        for d in glob.glob(os.path.join(base_dir,"**/__pycache__"),recursive=True):
            shutil.rmtree(d,True)
        clean_dst_directory_for_cpp(base_dir, packageName, "Core")
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


packages = find_packages()
submodules = [sub[sub.find(".") + 1:] for sub in packages if sub.count(".") == 1]

setup(
    cmdclass={"sdist":sdist,"build":build,"install":install,"clean":clean},
    entry_points = {
        packageName + '.submodules': [
            name + ' = ' + packageName + '.' + name
            for name in submodules
        ],
    },
)
