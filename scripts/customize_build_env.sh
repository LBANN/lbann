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
        DOMAINNAME=$(python3 -c 'import socket; domain = socket.getfqdn().split("."); print(domain[-2] + "." + domain[-1])')
        if [[ ${CORI} -eq 1 ]]; then
            CENTER="nersc"
        elif [[ ${DOMAINNAME} = "ornl.gov" ]]; then
            CENTER="olcf"
            NINJA_NUM_PROCESSES=16 # Don't let OLCF kill build jobs
        elif [[ ${DOMAINNAME} = "llnl.gov" ]]; then
            CENTER="llnl_lc"
        elif [[ ${DOMAINNAME} = "riken.jp" ]]; then
            CENTER="riken"
        else
            CENTER="llnl_lc"
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
                CMAKE_GPU_ARCH="60"
                ;;
            "broadwell")
                GPU_ARCH_VARIANTS="cuda_arch=60"
                CMAKE_GPU_ARCH="60"
                ;;
            *)
                ;;
        esac
    fi
}


set_center_specific_modules()
{
    local center="$1"
    local spack_arch_target="$2"

    if [[ ${center} = "llnl_lc" ]]; then
        # Disable the StdEnv for systems in LC
        case ${spack_arch_target} in
            "power9le" | "power8le")
                MODULE_CMD="module --force unload StdEnv; module load gcc/8.3.1 cuda/11.1.1 spectrum-mpi/rolling-release python/3.7.2"
                ;;
            "broadwell" | "haswell")
                MODULE_CMD="module --force unload StdEnv; module load gcc/8.3.1 cuda/11.1.0 mvapich2/2.3 python/3.7.2"
                ;;
            "ivybridge")
                MODULE_CMD="module --force unload StdEnv; module load gcc/8.3.1 mvapich2/2.3 python/3.7.2"
                ;;
            "zen")
                MODULE_CMD="module --force unload StdEnv; module load gcc/8.3.1 mvapich2/2.3 python/3.7.2"
                ;;
            *)
                echo "No pre-specified modules found for this system. Make sure to setup your own"
                ;;
        esac
    elif [[ ${center} = "olcf" ]]; then
        case ${spack_arch_target} in
            "power9le")
                MODULE_CMD="module load gcc/9.3.0 cuda/11.1.1 spectrum-mpi/10.3.1.2-20200121"
                ;;
            *)
                echo "No pre-specified modules found for this system. Make sure to setup your own"
                ;;
        esac
    elif [[ ${center} = "nersc" ]]; then
        case ${spack_arch_target} in
            "skylake_avx512")
                MODULE_CMD="module purge; module load cgpu modules/3.2.11.4 gcc/8.3.0 cuda/11.1.1 openmpi/4.0.3 cmake/3.18.2"
                ;;
            *)
                echo "No pre-specified modules found for this system. Make sure to setup your own"
                ;;
        esac
    elif [[ ${center} = "riken" ]]; then
        case ${spack_arch_target} in
            "a64fx")
                MODULE_CMD="module load system/fx700 gcc/10.1 gcc/openmpi/4.0.3"
                ;;
            *)
                echo "No pre-specified modules found for this system. Make sure to setup your own"
                ;;
        esac
    else
        echo "No pre-specified modules found for this system. Make sure to setup your own"
    fi
}

set_center_specific_mpi()
{
    local center="$1"
    local spack_arch_target="$2"

    if [[ ${center} = "llnl_lc" ]]; then
        case ${spack_arch_target} in
            "power9le" | "power8le")
                MPI="^spectrum-mpi"
                ;;
            "broadwell" | "haswell")
                # On LC the mvapich2 being used is built against HWLOC v1
                MPI="^mvapich2 ^hwloc@1.11.13"
                ;;
            *)
		echo "No center-specified MPI library."
                ;;
        esac
    elif [[ ${center} = "olcf" ]]; then
        case ${spack_arch_target} in
            "power9le")
                MPI="^spectrum-mpi"
                ;;
            *)
		echo "No center-specified MPI library."
                ;;
        esac
    elif [[ ${center} = "nersc" ]]; then
        case ${spack_arch_target} in
            "skylake_avx512")
                MPI="^openmpi"
                ;;
            *)
		echo "No center-specified MPI library."
                ;;
        esac
    elif [[ ${center} = "riken" ]]; then
        case ${spack_arch_target} in
            "a64fx")
                MPI="^openmpi"
                ;;
            *)
		echo "No center-specified MPI library."
                ;;
        esac
    else
        echo "No center-specified MPI library."
    fi
}

set_center_specific_externals()
{
    local center="$1"
    local spack_arch_target="$2"
    local spack_arch="$3"
    local yaml="$4"

    # Point compilers that don't have a fortran compiler a default one
    sed -i.bak -e 's/\(f[c7]7*:\)$/\1 \/usr\/bin\/gfortran/g' ${yaml}
    echo "Updating Clang compiler's to see the gfortran compiler."

    if [[ ${center} = "llnl_lc" ]]; then
        case ${spack_arch_target} in
            "broadwell" | "haswell" | "zen" | "power9le" | "power8le")
cat <<EOF  >> ${yaml}
    rdma-core:
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
EOF
                ;;
            *)
                echo "No center-specified externals."
                ;;
        esac
    elif [[ ${center} = "olcf" ]]; then
        case ${spack_arch_target} in
            "power9le" | "power8le")
cat <<EOF  >> ${yaml}
    rdma-core:
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
EOF
                ;;
            *)
                echo "No center-specified externals."
                ;;
        esac
    elif [[ ${center} = "nersc" ]]; then
        case ${spack_arch_target} in
            "skylake_avx512")
cat <<EOF  >> ${yaml}
    rdma-core:
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
EOF
                ;;
            *)
                echo "No center-specified externals."
                ;;
        esac
    elif [[ ${center} = "riken" ]]; then
        case ${spack_arch_target} in
            *)
        echo "No center-specified externals."
                ;;
        esac
    else
        echo "No center-specified externals."
    fi

    echo "Setting up a sane definition of how to represent modules."
cat <<EOF >> ${yaml}
  modules:
    enable::
      - tcl
      - lmod
    lmod::
      hash_length: 7
      projections:
        all: '\${PACKAGE}/\${VERSION}'
      all:
        filter:
          # Exclude changes to any of these variables
          environment_blacklist:
          - 'CPATH'
          - 'LIBRARY_PATH'
      ^python:
        autoload:  'direct'
    tcl:
      hash_length: 7
      projections:
        all: '\${PACKAGE}/\${VERSION}'
      all:
        filter:
          # Exclude changes to any of these variables
          environment_blacklist:
          - 'CPATH'
          - 'LIBRARY_PATH'
      ^python:
        autoload:  'direct'
EOF
}
