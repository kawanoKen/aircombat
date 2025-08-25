import sys,os,glob
prefix=sys.base_prefix
py_ver='%d.%d' % sys.version_info[:2]
print(
    os.path.dirname(glob.glob(os.path.join(prefix,'include','Python.h'))[0])
    if os.name=='nt' else
    os.path.dirname(glob.glob(os.path.join(prefix,'include/python'+py_ver+sys.abiflags+'/Python.h'))[0])
)
