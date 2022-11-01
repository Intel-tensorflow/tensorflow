#!/bin/bash
# This script is a CI script for invoking 'bazel test ... ...'
# It assumes the standard setup on tensorflow Jenkins windows machines.
# Update the flags/variables below to make it work on your local system.
#
# REQUIREMENTS:
# * All installed in standard locations:
#   - JDK8, and JAVA_HOME set.
#   - Microsoft Visual Studio 2015 Community Edition
#   - Msys2
#   - Python 3.x (with pip, setuptools, venv)
# * Bazel windows executable copied as "bazel.exe" and included in PATH.


# All commands shall pass (-e), and all should be visible (-x).
set -x
#set -e

POSITIONAL_ARGS=()
XBF_ARGS=""
XTF_ARGS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --extra_build_flags)
      XBF_ARGS="$2"
      shift # past argument
      shift # past value
      ;;
    --extra_test_flags)
      XTF_ARGS="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

#SCRIPT_ARGS=${POSITIONAL_ARGS[@]}

# bazelisk (renamed as bazel) is kept in C:\Tools
export PATH=/c/Tools/bazel:/c/Program\ Files/Git:/c/Program\ Files/Git/cmd:/c/msys64:/c/msys64/usr/bin:/c/Windows/system32:/c/Windows:/c/Windows/System32/Wbem

# Environment variables to be set by Jenkins before calling this script
# 

export PYTHON_VERSION=${PYTHON_VERSION:-"310"}  #We expect Python installation as C:\Python39
MYTFWS_ROOT=${WORKSPACE:-"C:/Users/mlp_admin"} # keep the tensorflow git repo clone under here as tensorflow subdir
MYTFWS_ROOT=`cygpath -m $MYTFWS_ROOT`
export MYTFWS_ROOT="$MYTFWS_ROOT"
export MYTFWS_NAME="tensorflow"
export MYTFWS="${MYTFWS_ROOT}/${MYTFWS_NAME}"
export MYTFWS_ARTIFACT="${MYTFWS_ROOT}/artifact"


export TF_LOCATION=%MYTFWS%

# Environment variables specific to the system where this job is running, to
# be set by a script for the specific system. This needs to be set here by
# sourcing a file.

export TMP=${TMP:-"${MYTFWS_ROOT}/tmp"}
export TEMP="$TMP"
export TMPDIR=${TMPDIR:-"${MYTFWS}-build"} # used internally by TF build
export TEST_TARGET=${TEST_TARGET:-"//tensorflow/..."}
export MSYS_LOCATION='C:/msys64'
export GIT_LOCATION='C:/Program Files/Git'
export JAVA_LOCATION='C:/Program Files/Eclipse Adoptium/jdk-11.0.14.101-hotspot'
export VS_LOCATION='C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools'
export NATIVE_PYTHON_LOCATION="C:/Python${PYTHON_VERSION}"

echo "*** *** hostname is $(hostname) *** ***"
which bazel
which git
[[ -e "$NATIVE_PYTHON_LOCATION/python.exe" ]] || { echo "Specified Python path is incorrect: $NATIVE_PYTHON_LOCATION"; exit 1;}
[[ -e "$NATIVE_PYTHON_LOCATION/Scripts/pip.exe" ]] || { echo "Specified Python path has no pip: $NATIVE_PYTHON_LOCATION"; exit 1;}
[[ -e "$NATIVE_PYTHON_LOCATION/Lib/venv" ]] || { echo "Specified Python path has no venv: $NATIVE_PYTHON_LOCATION"; exit 1;}

$NATIVE_PYTHON_LOCATION/python.exe -m pip list

# =========================== Start of actual script =========================
# This script set necessary environment variables and run TF-Windows build & unit tests
# We also assume a few Software components are also installed in the machine: MS VC++,
# MINGW SYS64, Python 3.x, JAVA, Git, Bazelisk etc.


# Asuumptions
# 1) TF repo cloned into to %WORKSPACE%\tensorflow (aka %TF_LOCATION%)
# 2) Bazelisk is installed in "C:\Tools\Bazel"
# 3) The following jobs specific env vars will be export  by the caller
#       WORKSPACE (ex. C:\Jenkins\workspace\tensorflow-eigen-test-win)
#       PYTHON_VERSION  (ex. 38)
#       PIP_MODULES (if set will conatain any additional pip packages)
# 4) System specific env variables for location of different software
#    components needed for building.


# create python virtual env.
cd ${MYTFWS_ROOT}
export PYTHON_DIRECTORY="${MYTFWS_ROOT}"/venv_py${PYTHON_VERSION}
"${NATIVE_PYTHON_LOCATION}"/python.exe -mvenv --clear  "${PYTHON_DIRECTORY}"

#activate virtual env
source "${PYTHON_DIRECTORY}"/Scripts/activate

which python
python --version


# Install pip modules as per specs in tensorflow/tools/ci_build/release/requirements_common.txt
python -m pip install -r $MYTFWS/tensorflow/tools/ci_build/release/requirements_common.txt

# set up other Variables required by bazel.
export PYTHON_BIN_PATH="${PYTHON_DIRECTORY}"/Scripts/python.exe
export PYTHON_LIB_PATH="${PYTHON_DIRECTORY}"/Lib/site-packages
export BAZEL_VS=${VS_LOCATION}
export BAZEL_VC=${VS_LOCATION}/VC
export JAVA_HOME=${JAVA_LOCATION}
export BAZEL_SH="${MSYS_LOCATION}"/usr/bin/bash.exe

cd ${MYTFWS_ROOT}
mkdir -p "$TMP"
# remove old logs
#rm -f summary.log test_failures.log test_run.log
mv summary.log summary.log.bak
mv test_failures.log test_failures.log.bak
mv test_run.log test_run.log.bak
rm -rf ${MYTFWS_ARTIFACT}
mkdir -p ${MYTFWS_ARTIFACT}

cd $MYTFWS


# All commands shall pass
set -e

# Setting up the environment variables Bazel and ./configure needs
source "tensorflow/tools/ci_build/windows/bazel/common_env.sh" \
  || { echo "Failed to source common_env.sh" >&2; exit 1; }

# load bazel_test_lib.sh
source "tensorflow/tools/ci_build/windows/bazel/bazel_test_lib.sh" \
  || { echo "Failed to source bazel_test_lib.sh" >&2; exit 1; }

# Recreate an empty bazelrc file under source root
export TMP_BAZELRC=.tmp.bazelrc
rm -f "${TMP_BAZELRC}"
touch "${TMP_BAZELRC}"

function cleanup {
  # Remove all options in .tmp.bazelrc
  echo "" > "${TMP_BAZELRC}"
}
trap cleanup EXIT

# Enable short object file path to avoid long path issue on Windows.
echo "startup --output_user_root=${TMPDIR}" >> "${TMP_BAZELRC}"

if ! grep -q "import %workspace%/${TMP_BAZELRC}" .bazelrc; then
  echo "import %workspace%/${TMP_BAZELRC}" >> .bazelrc
fi

run_configure_for_cpu_build

set +e   # Unset so script continues even if commands fail, this is needed to correctly process the logs

# NUMBER_OF_PROCESSORS is predefined on Windows
N_JOBS="${NUMBER_OF_PROCESSORS}"

# --config=release_cpu_windows 
bazel test \
  --action_env=TEMP=${TMP} --action_env=TMP=${TMP} ${XTF_ARGS} \
  --experimental_cc_shared_library --enable_runfiles --nodistinct_host_configuration \
  --dynamic_mode=off --config=xla --config=short_logs --announce_rc \
  --build_tag_filters=-no_windows,-no_oss --build_tests_only --config=monolithic \
  --config=opt \
  -k --keep_going --test_output=errors \
  --test_tag_filters=-no_windows,-no_oss,-gpu,-tpu,-v1only \
  --test_size_filters=small,medium --jobs="${N_JOBS}" --test_timeout=300,450,1200,3600 --verbose_failures \
  --flaky_test_attempts=3 \
  ${POSITIONAL_ARGS[@]} \
  -- ${TEST_TARGET} \
  > run.log 2>&1

build_ret_val=$?   # Store the ret value

# process results
cd $MYTFWS_ROOT

# Check to make sure log was created.
[ ! -f "${MYTFWS}"/run.log  ] && exit 1

# handle logs for unit test
cd ${MYTFWS_ARTIFACT}
cp "${MYTFWS}"/run.log ./test_run.log

fgrep "FAILED: Build did NOT complete" test_run.log > summary.log
fgrep "Executed" test_run.log >> summary.log

[ $build_ret_val -eq 0 ] && exit 0

echo "FAILED TESTS:" > test_failures.log
fgrep "FAILED" test_run.log | grep " ms)" | sed -e 's/^.*\] //' -e 's/ .*$//' | sort | uniq >> test_failures.log
echo >> test_failures.log
echo "SKIPPED TESTS:" >> test_failures.log
fgrep "SKIPPED" test_run.log | grep -v "listed below:" | sed -e 's/^.*\] //' | sort | uniq >> test_failures.log
# fgrep "TIMEOUT" test_run.log | grep "out of" | sed -e 's/[ ][ ]*.*//' -e 's/$/ TIMEOUT/' >> test_failures.log
# count=$(wc -l < test_failures.log)

exit 1
