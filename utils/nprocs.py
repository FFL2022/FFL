import os
try:
    nprocs = int(os.popen('nproc').read()) - 1 # leave one core for reserve
except:
    nprocs = int(os.popen('sysctl -n hw.ncpu').read().strip()) - 1 # leave one core for reserve

