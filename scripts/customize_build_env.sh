#!/bin/sh

# Function to setup the following fields based on the center that you are running at:
# CENTER
# BUILD_SUFFIX
# COMPILER
# NINJA_NUM_PROCESSES
set_center_specific_fields()
{
    SYS=$(uname -s)

    if [[ ${SYS} = "Darwin" ]]; then
        CENTER="osx"
        COMPILER="clang"
        BUILD_SUFFIX=llnl.gov
    else
        CORI=$([[ $(hostname) =~ (cori|cgpu) ]] && echo 1 || echo 0)
        DOMAINNAME=$(python -c 'import socket; domain = socket.getfqdn().split("."); print(domain[-2] + "." + domain[-1])')
        if [[ ${CORI} -eq 1 ]]; then
            CENTER="nersc"
            # Make sure to purge and setup the modules properly prior to finding the Spack architecture
            source ${SPACK_ENV_DIR}/${CENTER}/setup_modules.sh
            BUILD_SUFFIX=nersc.gov
        elif [[ ${DOMAINNAME} = "ornl.gov" ]]; then
            CENTER="olcf"
            BUILD_SUFFIX=${DOMAINNAME}
            NINJA_NUM_PROCESSES=16 # Don't let OLCF kill build jobs
        elif [[ ${DOMAINNAME} = "llnl.gov" ]]; then
            CENTER="llnl_lc"
            BUILD_SUFFIX=${DOMAINNAME}
        else
            CENTER="llnl_lc"
            BUILD_SUFFIX=llnl.gov
        fi
        COMPILER="gnu"
    fi
}

# Function to setup the following fields based on the center that you are running at:
# GPU_ARCH_VARIANTS
# CMAKE_GPU_ARCH
set_center_specific_gpu_arch()
{
    local center="$1" 
    local spack_arch_target="$2"

    if [[ ${center} = "llnl_lc" ]]; then
        case ${spack_arch_target} in
            "power9le")
                GPU_ARCH_VARIANTS="cuda_arch=70"
                CMAKE_GPU_ARCH="70"
                ;;
            "power8le")
                GPU_ARCH_VARIANTS="cuda_arch=60"
                CMAKE_GPU_ARCH="70"
                ;;
            "broadwell")
                GPU_ARCH_VARIANTS="cuda_arch=60"
                CMAKE_GPU_ARCH="70"
                ;;
            *)
                ;;
        esac
    fi
}
