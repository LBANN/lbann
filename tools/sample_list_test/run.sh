HDF5_DIR=/usr/workspace/wsb/icfsi/conduit-helpers/hdf5-quartz/hdf5
HDF5_LIB=${HDF5_DIR}/lib
CONDUIT_DIR=/usr/workspace/wsb/icfsi/conduit/install-toss3-4.9.3-serial
CONDUIT_LIB=${CONDUIT_DIR}/lib

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDUIT_LIB}:${HDF5_LIB}
list_file=/p/lscratchh/brainusr/datasets/10MJAG/1M_A/index.txt
./test ${list_file} 1

