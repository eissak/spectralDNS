-- # Cray
load("PrgEnv-cray")
load("craype-x86-milan")
load("cce")
load("craype")

-- # OFI
load("cray-mpich")
load("craype-network-ofi")
load("libfabric")

-- # shared memory support
load("cray-pmi")
load("cray-dsmml")
load("cray-openshmemx")

-- # other tools
load("cray-hdf5-parallel")
load("cray-python")
load("cray-libsci")
load("cray-fftw")
load("perftools-base")

local user = os.getenv("USER")

execute {cmd="source /home/" .. user .. "/.virtualenvs/spectralDNS/bin/activate", modeA={"load"}}
