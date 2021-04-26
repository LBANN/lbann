#!/bin/bash

ORIG_CMD="$0 $@"
SCRIPT=${BASH_SOURCE}

# Grab some helper functions
source $(dirname ${BASH_SOURCE})/utilities.sh

# Detect system parameters
SYS=$(uname -s)
if [[ ${SYS} = "Darwin" ]]; then
SCRIPTS_DIR=$(normpath $(dirname $(osx_realpath ${SCRIPT})))
else
SCRIPTS_DIR=$(realpath $(dirname ${SCRIPT}))
fi

LBANN_HOME=$(dirname ${SCRIPTS_DIR})
#SPACK_ENV_DIR=${LBANN_HOME}/spack_environments

SCRIPT=$(basename ${BASH_SOURCE})
LBANN_ENV=
INSTALL_DEPS=
DRY_RUN=
CLEAN_BUILD=
# Flag for passing subcommands to spack dev-build
DEV_BUILD_FLAGS=
# Flag for passing subcommands to spack install and dev-build
INSTALL_DEV_BUILD_EXTRAS=

LBANN_VARIANTS=
CMD_LINE_VARIANTS=

# Default versions of Hydrogen, DiHydrogen, and Aluminum - use head of repo
HYDROGEN_VER="@develop"
ALUMINUM_VER="@master"
DIHYDROGEN_VER="@develop"

################################################################
# Help message
################################################################

function help_message {
    local SCRIPT=$(basename ${BASH_SOURCE})
    local N=$(tput sgr0)    # Normal text
    local C=$(tput setf 4)  # Colored text
    cat << EOF
##########################################################################################
Build LBANN: has preconfigured module lists for LLNL LC, OLCF, and NERSC systems.
  Will create a build directory with the name spack-build-<hash> in the root of the LBANN project tree.
  primary variants that can be passed into lbann come from spack and can be seen with:
    spack info lbann
  and passed to this script via:
    ${SCRIPT} -- <variants>
##########################################################################################
Usage: ${SCRIPT} [options] -- [list of spack variants]
Options:
  ${C}--help${N}                  Display this help message and exit.
  ${C}--clean-build${N}           Delete the local link to the build directory
  ${C}--clean-deps${N}            Forcibly uninstall Hydrogen, Aluminum, and DiHydrogen dependencies
  ${C}-d | --install-deps${N}     Install the lbann dependencies in addition to building from local source
  ${C}--dry-run${N}               Dry run the commands (no effect)
  ${C}-l | --label${N}            LBANN version label prefix: (default label is local-<SPACK_ARCH_TARGET>,
                          and is built and installed in the spack environment lbann-<label>-<SPACK_ARCH_TARGET>
  ${C}-j | --build-jobs${N}       Number of parallel processes to use for compiling, e.g. -j \$((\$(nproc)+2))
  ${C}-m | --mirror${N}           Specify a Spack mirror (and buildcache)
  ${C}--no-modules${N}            Don't try to load any modules (use the existing users environment)
  ${C}--tmp-build-dir${N}         Put the build directory in tmp space
  ${C}--spec-only${N}             Stop after a spack spec command
  ${C}-s | --stable${N}           Use the latest stable defaults not the head of Hydrogen, DiHydrogen and Aluminum repos
  ${C}--test${N}                  Enable local unit tests
  ${C}--hydrogen-repo PATH${N}    Use a local repository for the Hydrogen library
  ${C}--dihydrogen-repo PATH${N}  Use a local repository for the DiHydrogen library
  ${C}--aluminum-repo PATH${N}    Use a local repository for the Aluminum library
  ${C}--${N}                      Pass all variants to spack after the dash dash (--)
EOF
}

################################################################
# Parse command-line arguments
################################################################

while :; do
    case ${1} in
        -h|--help)
            # Help message
            help_message
            exit 1
            ;;
        --clean-build)
            CLEAN_BUILD="TRUE"
            ;;
        --clean-deps)
            CLEAN_DEPS="TRUE"
            ;;
        -d|--install-deps)
            INSTALL_DEPS="TRUE"
            ;;
        --dry-run)
            DRY_RUN="TRUE"
            ;;
        -l|--label)
            # Change default LBANN version label
            if [ -n "${2}" ]; then
                LBANN_LABEL_PREFIX=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        -j|--build-jobs)
            if [ -n "${2}" ]; then
                BUILD_JOBS="-j${2}"
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        -m|--mirror)
            if [ -n "${2}" ]; then
                USER_MIRROR="${2}"
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --no-modules)
            SKIP_MODULES="TRUE"
            ;;
        --tmp-build-dir)
            CLEAN_BUILD="TRUE"
            TMP_BUILD_DIR="TRUE"
            ;;
        --spec-only)
            SPEC_ONLY="TRUE"
            ;;
        -s|--stable-defaults)
            # Use the latest released version
            HYDROGEN_VER=
            ALUMINUM_VER=
            DIHYDROGEN_VER=
            ;;
        --test)
            INSTALL_DEV_BUILD_EXTRAS="--test root"
            ;;
        --hydrogen-repo)
            if [ -n "${2}" ]; then
                HYDROGEN_PATH=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --dihydrogen-repo)
            if [ -n "${2}" ]; then
                DIHYDROGEN_PATH=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --aluminum-repo)
            if [ -n "${2}" ]; then
                ALUMINUM_PATH=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --update-buildcache)
            UPDATE_BUILDCACHE="TRUE"
            ;;
        --)
            shift
            CMD_LINE_VARIANTS=${*}
            LBANN_VARIANTS=${CMD_LINE_VARIANTS}
            break
            ;;
        -?*)
            # Unknown option
            echo "Unknown option (${1})" >&2
            exit 1
            ;;
        *)
            # Break loop if there are no more options
            break
    esac
    shift
done

function uninstall_specific_versions()
{
    local package="$1"
    local version="$2"

    SPACK_ARCH=$(spack arch)
    # Ensure that only versions for this architecture are found
    FIND_CMD="spack find --format {hash:7} ${package}${version} arch=${SPACK_ARCH}"
    echo ${FIND_CMD} | tee -a ${LOG}
    HASH=$(${FIND_CMD})
    if [[ -n "${HASH}" && ! "${HASH}" =~ "No package matches the query" ]]; then
        HASH_ARRAY=(${HASH})
        for h in ${HASH_ARRAY[@]}
        do
            CMD="spack uninstall -y --force ${package}${version} /${h}"
            echo ${CMD} | tee -a ${LOG}
            [[ -z "${DRY_RUN:-}" ]] && (${CMD} || exit_on_failure "${CMD}")
        done
    fi
}

# "spack" is just a shell function; it may not be exported to this
# scope. Just to be sure, reload the shell integration.
if [ -n "${SPACK_ROOT}" ]; then
    source ${SPACK_ROOT}/share/spack/setup-env.sh
else
    echo "Spack required.  Please set SPACK_ROOT environment variable"
    exit 1
fi

SPACK_VERSION=$(spack --version | sed 's/-.*//g')
MIN_SPACK_VERSION=0.16.0

compare_versions ${SPACK_VERSION} ${MIN_SPACK_VERSION}
VALID_SPACK=$?

if [[ ${VALID_SPACK} -eq 2 ]]; then
    echo "Newer version of Spack required.  Detected version ${SPACK_VERSION} requires at least ${MIN_SPACK_VERSION}"
    exit 1
fi

##########################################################################################
# Make sure that Spack is using Clingo
function ask_permission()
{
    local question="$1"
    echo "${question}"
    read -p "Continue (y/n)? " choice
    case "$choice" in
        y|Y ) RESPONSE=1 ;;
        n|N ) RESPONSE=0 ;;
        * ) RESPONSE=0 ;;
    esac
    return ${RESPONSE}
}

SPACK_SITE_CONFIG="${SPACK_ROOT}/etc/spack/config.yaml"
if [[ ! -f "${SPACK_SITE_CONFIG}" ]]; then
    ask_permission "No site specific ${SPACK_SITE_CONFIG} file found, create one?"
    RESPONSE=$?
    if [[ ${RESPONSE} -eq 1 ]]; then
        echo "Creating a new ${SPACK_SITE_CONFIG} file."
        if [[ -z "${DRY_RUN:-}" ]]; then
            cat <<EOF >> ${SPACK_SITE_CONFIG}
config:
  concretizer: clingo
EOF
        fi
    else
        echo "${SCRIPT} requires use of Spack's clingo optimizer -- please enable it"
        echo "e.g. create a site specific config at ${SPACK_SITE_CONFIG}"
        cat <<EOF
config:
  concretizer: clingo
EOF
        exit 1
    fi
else
    SPACK_CONCRETIZER=$(grep "concretizer:" ${SPACK_SITE_CONFIG} | awk '{print $2}')
    if [[ -z "${SPACK_CONCRETIZER}" ]]; then
        ask_permission "Site specific ${SPACK_SITE_CONFIG} file does not have a explicit concretizer field, add one?"
        RESPONSE=$?
        if [[ ${RESPONSE} -eq 1 ]]; then
            CMD="cp ${SPACK_SITE_CONFIG} ${SPACK_SITE_CONFIG}.pre_lbann"
            echo "Appending 'concretizer: clingo' and saving old config ${CMD}"
            if [[ -z "${DRY_RUN:-}" ]]; then
                ${CMD}
                cat <<EOF >> ${SPACK_SITE_CONFIG}
  concretizer: clingo
EOF
            fi
        else
            echo "${SCRIPT} requires use of Spack's clingo optimizer -- please enable it"
            echo "e.g. add the line to ${SPACK_SITE_CONFIG}"
            cat <<EOF
  concretizer: clingo
EOF
            exit 1
        fi
    else
        if [[ ! "${SPACK_CONCRETIZER}" == "clingo" ]]; then
            echo "${SCRIPT} requires use of Spack's clingo optimizer -- please enable it"
            echo "e.g. edit the line in ${SPACK_SITE_CONFIG}"
            cat <<EOF
  concretizer: original
EOF
            echo "to look like"
            cat <<EOF
  concretizer: clingo
EOF
            exit 1
        fi
    fi
fi
##########################################################################################

# Detect system parameters
CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')

# Identify the center that we are running at
CENTER=
# Customize the build based on the center
source $(dirname ${BASH_SOURCE})/customize_build_env.sh
set_center_specific_fields

SPACK_ARCH=$(spack arch)
SPACK_ARCH_TARGET=$(spack arch -t)
SPACK_ARCH_PLATFORM=$(spack arch -p)
SPACK_ARCH_GENERIC_TARGET=$(spack python -c "import archspec.cpu as cpu; print(str(cpu.host().family))")
# Create a modified spack arch with generic target architecture
SPACK_ARCH_PLATFORM_GENERIC_TARGET="${SPACK_ARCH_PLATFORM}-${SPACK_ARCH_GENERIC_TARGET}"

LBANN_LABEL="${LBANN_LABEL_PREFIX:-local}-${SPACK_ARCH_TARGET}"
LBANN_ENV="${LBANN_ENV:-lbann-${LBANN_LABEL}}"
CORE_BUILD_PATH="${LBANN_HOME}/build/${CLUSTER}.${LBANN_ENV}"

LOG="spack-build-${LBANN_ENV}.log"
if [[ -f ${LOG} ]]; then
    CMD="rm ${LOG}"
    echo ${CMD}
    [[ -z "${DRY_RUN:-}" ]] && ${CMD}
fi

function exit_on_failure()
{
    local cmd="$1"
    echo -e "FAILED: ${cmd}"
    echo "##########################################################################################" | tee -a ${LOG}
    echo "LBANN is being installed in a spack environment named ${LBANN_ENV} but an error occured, access it via:" | tee -a ${LOG}
    echo "  spack env activate -p ${LBANN_ENV}" | tee -a ${LOG}
    echo "To rebuild LBANN from source drop into a shell with the spack build environment setup (requires active environment):" | tee -a ${LOG}
    echo "  spack build-env lbann -- bash" | tee -a ${LOG}
    echo "  cd spack-build-${LBANN_SPEC_HASH}" | tee -a ${LOG}
    echo "  ninja install" | tee -a ${LOG}
    echo "##########################################################################################" | tee -a ${LOG}
    echo "All details of the run are logged to ${LOG}"
    echo "##########################################################################################"
    exit 1
}

if [[ ! "${LBANN_VARIANTS}" =~ .*"^hydrogen".* ]]; then
    # If the user didn't supply a specific version of Hydrogen on the command line add one
    HYDROGEN="^hydrogen${HYDROGEN_VER}"
fi

if [[ (! "${LBANN_VARIANTS}" =~ .*"^aluminum".*) && (! "${LBANN_VARIANTS}" =~ .*"~al".*) ]]; then
    # If the user didn't supply a specific version of Aluminum on the command line add one
    ALUMINUM="^aluminum${ALUMINUM_VER}"
fi

if [[ ! "${LBANN_VARIANTS}" =~ .*"^dihydrogen".* ]]; then
    # If the user didn't supply a specific version of DiHydrogen on the command line add one
    # Due to concretizer errors force the openmp variant for DiHydrogen
    DIHYDROGEN="^dihydrogen${DIHYDROGEN_VER}"
fi

GPU_VARIANTS_ARRAY=('+cuda' '+rocm')
for GPU_VARIANTS in ${GPU_VARIANTS_ARRAY[@]}
do
    if [[ "${LBANN_VARIANTS}" =~ .*"${GPU_VARIANTS}".* ]]; then
        # Define the GPU_ARCH_VARIANTS field
        GPU_ARCH_VARIANTS=
        set_center_specific_gpu_arch ${CENTER} ${SPACK_ARCH_TARGET}
        # Prepend the GPU_ARCH_VARIANTS for the LBANN variants if the +cuda variant is defined
        LBANN_VARIANTS=" ${GPU_ARCH_VARIANTS} ${LBANN_VARIANTS}"
    fi
done

# Record the original command in the log file
echo "${ORIG_CMD}" | tee -a ${LOG}

if [[ ! -n "${SKIP_MODULES:-}" ]]; then
    # Activate modules
    MODULE_CMD=
    set_center_specific_modules ${CENTER} ${SPACK_ARCH_TARGET}
    if [[ -n ${MODULE_CMD} ]]; then
        echo ${MODULE_CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { eval ${MODULE_CMD} || exit_on_failure "${MODULE_CMD}"; }
    fi
fi

# If the dependencies are being installed then you should clean things up
if [[ -n "${INSTALL_DEPS:-}" ]]; then
    # Remove any old environment with the same name
    if [[ $(spack env list | grep -e "${LBANN_ENV}$") ]]; then
        echo "Spack environment ${LBANN_ENV} already exists... overwriting it"
        CMD="spack env rm --yes-to-all ${LBANN_ENV}"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" && -n "${INSTALL_DEPS:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi

    # Create the environment
    CMD="spack env create ${LBANN_ENV}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

#     SPACK_ENV_YAML_FILE="${SPACK_ROOT}/var/spack/environments/${LBANN_ENV}/spack.yaml"
# cat <<EOF  >> ${SPACK_ENV_YAML_FILE}
#   concretization: together
# EOF

fi

##########################################################################################
# Uninstall any existing versions for this architecture with the same label -- note that
# this has to be done outside of an environment
# For finding the lbann version don't use the architecture because sometimes it is "downgraded"
LBANN_FIND_CMD="spack find --format {hash:7} lbann@${LBANN_LABEL}"
echo ${LBANN_FIND_CMD} | tee -a ${LOG}
LBANN_HASH=$(${LBANN_FIND_CMD})
if [[ -n "${LBANN_HASH}" && ! "${LBANN_HASH}" =~ "No package matches the query" ]]; then
    LBANN_HASH_ARRAY=(${LBANN_HASH})
    for h in ${LBANN_HASH_ARRAY[@]}
    do
        CMD="spack uninstall -y --force lbann@${LBANN_LABEL} /${h}"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    done
fi

if [[ -n "${CLEAN_DEPS:-}" ]]; then
    uninstall_specific_versions "hydrogen" "${HYDROGEN_VER}"
    uninstall_specific_versions "aluminum" "${ALUMINUM_VER}"
    uninstall_specific_versions "dihydrogen" "${DIHYDROGEN_VER}"
fi

CMD="spack env activate -p ${LBANN_ENV}"
echo ${CMD} | tee -a ${LOG}
if [[ -z "${DRY_RUN:-}" ]]; then
    if [[ -z $(spack env list | grep -e "${LBANN_ENV}$") ]]; then
        echo "Spack could not activate environment ${LBANN_ENV} -- install dependencies with -d flag"
        exit 1
    fi
    ${CMD} || exit_on_failure "${CMD}"
fi

# Figure out if there is a default MPI library for the center
CENTER_DEPENDENCIES=
CENTER_FLAGS=
set_center_specific_spack_dependencies ${CENTER} ${SPACK_ARCH_TARGET}

##########################################################################################
# See if the is a local spack mirror or buildcache
if [[ -n "${USER_MIRROR:-}" ]]; then
    # Allow the user to overwrite a standard mirror
    MIRROR=${USER_MIRROR}
fi

if [[ -n "${INSTALL_DEPS:-}" && -n "${MIRROR:-}" && -r "${MIRROR:-}" ]]; then
    CMD="spack mirror add lbann ${MIRROR}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    # Tell Spack to trust the keys in the build cache
    CMD="spack buildcache keys --install --trust"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    # Manually force Spack to trust the keys in the build cache - this is a hack until
    # https://github.com/spack/spack/issues/23186 is fixed
    if [[ -e "${MIRROR}/build_cache/_pgp/B180FE4A5ECF4C02D21E6A67F13D1FBB0E55F96F.pub" ]]; then
        CMD="spack gpg trust ${MIRROR}/build_cache/_pgp/B180FE4A5ECF4C02D21E6A67F13D1FBB0E55F96F.pub"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi
fi

##########################################################################################
# Establish the spec for LBANN
LBANN_SPEC="lbann@${LBANN_LABEL} ${CENTER_FLAGS} ${LBANN_VARIANTS} ${HYDROGEN} ${DIHYDROGEN} ${ALUMINUM} ${CENTER_DEPENDENCIES}"
##########################################################################################

##########################################################################################
# Add things to the environment
##########################################################################################
if [[ -n "${INSTALL_DEPS:-}" ]]; then
    # See if there are any center-specific externals
    SPACK_ENV_YAML_FILE="${SPACK_ROOT}/var/spack/environments/${LBANN_ENV}/spack.yaml"
    CMD="set_center_specific_externals ${CENTER} ${SPACK_ARCH_TARGET} ${SPACK_ARCH} ${SPACK_ENV_YAML_FILE}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    CMD="spack compiler find --scope env:${LBANN_ENV}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    CMD="spack external find --scope env:${LBANN_ENV}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    CMD="cleanup_clang_compilers ${CENTER} ${SPACK_ENV_YAML_FILE}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    ##########################################################################################
    # Tell the spack environment to use a local repository for these libraries
    if [[ -n "${HYDROGEN_PATH:-}" ]]; then
        CMD="spack develop --no-clone -p ${HYDROGEN_PATH} hydrogen${HYDROGEN_VER}"
        echo "${CMD}" | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi

    if [[ -n "${DIHYDROGEN_PATH:-}" ]]; then
        CMD="spack develop --no-clone -p ${DIHYDROGEN_PATH} dihydrogen${DIHYDROGEN_VER}"
        echo "${CMD}" | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi

    if [[ -n "${ALUMINUM_PATH:-}" ]]; then
        CMD="spack develop --no-clone -p ${ALUMINUM_PATH} aluminum${ALUMINUM_VER}"
        echo "${CMD}" | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi
    ##########################################################################################

    # Explicitly add the lbann spec to the environment
    CMD="spack add ${LBANN_SPEC}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    # Explicitly mark lbann for development
    CMD="spack develop --no-clone -p ${LBANN_HOME} ${LBANN_SPEC}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    ##########################################################################################
    # If there is a local mirror, pad out the install tree so that it can be relocated
    if [[ -n "${MIRROR:-}" && -r "${MIRROR:-}" && -n "${UPDATE_BUILDCACHE:-}" ]]; then
        spack config add "config:install_tree:padded_length:128"
    fi

    # Add any extra packages that you want to build in conjuction with the LBANN package
    # spack add py-merlin
fi

CMD="spack solve -l ${LBANN_SPEC}"
echo ${CMD} | tee -a ${LOG}
if [[ -z "${DRY_RUN:-}" ]]; then
    eval ${CMD} || exit_on_failure "${CMD}\nIf the error is that boostrapping failed try something like 'module load gcc/8.3.1; spack compiler add' and then rerunning"
fi
# Get the spack hash for LBANN
LBANN_SPEC_HASH=$(spack solve -l ${LBANN_SPEC} | grep lbann | grep arch=${SPACK_ARCH_PLATFORM} | awk '{print $1}')
[[ -z "${DRY_RUN:-}" && "${SPEC_ONLY}" == "TRUE" ]] && exit

LINK_DIR="${LINK_DIR:-${CORE_BUILD_PATH}}"
BUILD_DIR=$(dirname ${LINK_DIR})
if [[ ! -d "${BUILD_DIR}" ]]; then
    CMD="mkdir -p ${BUILD_DIR}"
    echo ${CMD}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi

# Check to see if the link to the build directory exists and is valid
SPACK_BUILD_DIR="spack-build-${LBANN_SPEC_HASH}"
if [[ -L "${SPACK_BUILD_DIR}" ]]; then
  # If the link is not valid or are told to clean it, remove the link
  if [[ ! -d "${SPACK_BUILD_DIR}" || ! -z "${CLEAN_BUILD}" ]]; then
      CMD="rm ${SPACK_BUILD_DIR}"
      echo ${CMD}
      [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
  fi
fi

# If the spack build directory does not exist, create a tmp directory and link it
if [[ ! -e "${SPACK_BUILD_DIR}" && -n "${TMP_BUILD_DIR:-}" && -z "${DRY_RUN:-}" ]]; then
    tmp_dir=$(mktemp -d -t lbann-spack-build-${LBANN_SPEC_HASH}-$(date +%Y-%m-%d-%H%M%S)-XXXXXXXXXX)
    echo ${tmp_dir}
    CMD="ln -s ${tmp_dir} spack-build-${LBANN_SPEC_HASH}"
    echo ${CMD}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi

##########################################################################################
# Actually install LBANN from local source
CMD="spack install ${BUILD_JOBS}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

if [[ -n "${MIRROR:-}" && -r "${MIRROR:-}"  && -n "${UPDATE_BUILDCACHE:-}" ]]; then
    # Make sure that all of the packages in the environment are in the mirror
    CMD="spack mirror create -d ${MIRROR} --all"
    echo ${CMD} | tee -a ${LOG}
    # Don't check the return code of the mirror create command, it will fail to install some packages
    [[ -z "${DRY_RUN:-}" ]] && ${CMD}

    if [[ ! -e "${MIRROR}/pubring.gpg" ]]; then
        CMD="cp ${SPACK_ROOT}/opt/spack/gpg/pubring.gpg ${MIRROR}/pubring.gpg"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi

    SPACK_INSTALL_ROOT=$(grep root $SPACK_ROOT/etc/spack/config.yaml | awk '{ print $2 }')
    for ii in $(spack find --format "{prefix} {version} {name},/{hash}" |
        grep -v -E "^(develop^master)" |
        grep -e "${SPACK_ROOT}" -e "${SPACK_INSTALL_ROOT}" |
        cut -f3 -d" ")
    do
        NAME=${ii%,*};
        HASH=${ii#*,};
        case ${NAME} in
            "cuda" | "cudnn" | "ncurses" | "openssl" | "lbann")
                echo "Skipping $ii"
                continue
                ;;
        esac
        CMD="spack buildcache check --rebuild-on-error --mirror-url file://${MIRROR} -s ${HASH}"
        echo -e "${NAME}:\t ${CMD}" | tee -a ${LOG}
        if [[ -z "${DRY_RUN:-}" ]]; then
            if ${CMD};
            then
                true
            else
                CMD="spack buildcache create -af -d ${MIRROR} --only=package ${HASH}"
                echo ${CMD} | tee -a ${LOG}
                [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
            fi
        fi
    done
    CMD="spack buildcache update-index -d ${MIRROR}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi

# Don't use the output of this file since it will not exist if the compilation is not successful
# LBANN_BUILD_DIR=$(grep "PROJECT_BINARY_DIR:" ${LBANN_HOME}/spack-build-out.txt | awk '{print $2}')

if [[ -L "${LINK_DIR}" ]]; then
    CMD="rm ${LINK_DIR}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi

CMD="ln -s ${LBANN_HOME}/spack-build-${LBANN_SPEC_HASH} ${LINK_DIR}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

##########################################################################################
# Once LBANN is installed deactivate the environment and try to find the package to get the
# installed path
# This is no longer necessary since we don't need to find the compiler version right now
# spack env deactivate
# LBANN_FIND_CMD="spack find --path lbann@${LBANN_LABEL} arch=${SPACK_ARCH} /${LBANN_SPEC_HASH}"
# echo ${LBANN_FIND_CMD} | tee -a ${LOG}
# COMPILER_VER="<FIND ONE>"
# if [[ -z "${DRY_RUN:-}" ]]; then
#     LBANN_PATH=$(${LBANN_FIND_CMD})
#     LBANN_INSTALL_DIR=${LBANN_PATH##* }
#     COMPILER_VER=$(basename $(dirname $LBANN_INSTALL_DIR))
# fi
echo "##########################################################################################" | tee -a ${LOG}
echo "LBANN is installed in a spack environment named ${LBANN_ENV}, access it via:" | tee -a ${LOG}
echo "  spack env activate -p ${LBANN_ENV}" | tee -a ${LOG}
echo "To rebuild LBANN from source drop into a shell with the spack build environment setup (requires active environment):" | tee -a ${LOG}
echo "  spack build-env lbann -- bash" | tee -a ${LOG}
echo "  cd spack-build-${LBANN_SPEC_HASH}" | tee -a ${LOG}
echo "  ninja install" | tee -a ${LOG}
echo "To use this version of LBANN use the module system without the need for activating the environment (does not require being in an environment)" | tee -a ${LOG}
echo "  module load lbann/${LBANN_LABEL}-${LBANN_SPEC_HASH}" | tee -a ${LOG}
echo "or have spack load the module auto-magically. It is installed in a spack environment named ${LBANN_ENV}, access it via: (has to be executed from the environment)"  | tee -a ${LOG}
echo "  spack load lbann@${LBANN_LABEL} arch=${SPACK_ARCH}" | tee -a ${LOG}
echo "##########################################################################################" | tee -a ${LOG}
echo "All details of the run are logged to ${LOG}"
echo "##########################################################################################"

# Lastly, Save the log file in the build directory
CMD="cp ${LOG} ${LBANN_HOME}/spack-build-${LBANN_SPEC_HASH}/${LOG}"
echo ${CMD}
[[ -z "${DRY_RUN:-}" ]] && ${CMD}
