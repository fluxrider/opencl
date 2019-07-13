# Windows 10:
#  python-3.7.4-amd64-webinstall.exe
#  vs_buildtools__2042667395.1562888908.exe
#  NVidia:
#    cuda_10.1.168_425.25_win10.exe
#  In a admin cmd ("cmd"+Ctrl+Shift+Enter in Win+R dialog)
#  set INCLUDE=C:\_\apps\nvidia_cuda_v10.1\include
#  set LIB=C:\_\apps\nvidia_cuda_v10.1\lib\x64
#  pip install pyopencl

import pyopencl

for p in pyopencl.get_platforms():
  print("Platform")
  print(f" version: {p.version}")
  print(f" vendor: {p.vendor}")
  print(f" profile: {p.profile}")
  print(f" name: {p.name}")
  print(f" extensions: {p.extensions}")
  for d in p.get_devices():
    print(" Device")
    print(f"  address_bits: {d.address_bits}")
    print(f"  extensions: {d.extensions}")
    print(f"  global_mem_size: {d.global_mem_size}")
    print(f"  max_work_group_size: {d.max_work_group_size}")
    print(f"  name: {d.name}")
    print(f"  vendor: {d.vendor}")