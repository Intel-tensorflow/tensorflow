# This script is a wrapper CI script for invoking 
# <repo_root>\tensorflow\tools\ci_build\windows\cpu\pip\build_tf_windows.sh
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


# All commands shall pass, and all should be visible.
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

export TMP="${MYTFWS_ROOT}/tmp"
export TEMP="$TMP"
export TMPDIR="${MYTFWS}-build" # used internally by TF build
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

set +e   # Unset so script continues even if commands fail, this is needed to correctly process the logs

cd $MYTFWS

bash "${MYTFWS}"/tensorflow/tools/ci_build/windows/cpu/pip/build_tf_windows.sh \
   --extra_build_flags "--action_env=TEMP=${TMP} --action_env=TMP=${TMP} ${XBF_ARGS}" \
   --extra_test_flags "--action_env=TEMP=${TMP} --action_env=TMP=${TMP} ${XTF_ARGS}" \
   ${POSITIONAL_ARGS[@]}  > run.log 2>&1

build_ret_val=$?   # Store the ret value

# Retry once more with "bazel clean" for failed builds to get rid of any stale cache
if [[ $build_ret_val -ne 0 ]]; then
  cd ${MYTFWS}
  bazel --output_user_root=${TMPDIR} clean --expunge

  bash "${MYTFWS}"/tensorflow/tools/ci_build/windows/cpu/pip/build_tf_windows.sh \
     --extra_build_flags "--action_env=TEMP=${TMP} --action_env=TMP=${TMP} ${XBF_ARGS}" \
     --extra_test_flags "--action_env=TEMP=${TMP} --action_env=TMP=${TMP} ${XTF_ARGS}" \
     ${POSITIONAL_ARGS[@]}, --jobs=24  > run.log 2>&1

  build_ret_val=$?   # Store the ret value
fi

# process results
cd $MYTFWS_ROOT

# Check to make sure log was created.
[ ! -f "${MYTFWS}"/run.log  ] && exit 1

# Handle the case when only whl are built
if [[ "$TF_NIGHTLY" = 1 ]] || [[ "$RELEASE_BUILD" = 1 ]]; then
  if [[ $build_ret_val -eq 0 ]]; then
    cp ${MYTFWS}/py_test_dir/*.whl ${MYTFWS_ARTIFACT}
    cp "${MYTFWS}"/run.log ${MYTFWS_ARTIFACT}/test_run_${PYTHON_VERSION}.log
  else
    # build failed just copy the log, mark log with py version.
    cp "${MYTFWS}"/run.log ${MYTFWS_ARTIFACT}/test_run_${PYTHON_VERSION}.log
  fi
  exit $build_ret_val
fi


# handle logs for unit test
cd ${MYTFWS_ARTIFACT}
cp "${MYTFWS}"/run.log ./test_run.log

ret=0
fgrep "FAILED: Build did NOT complete" test_run.log > summary.log
[ $? -eq 0 ] && ret=1
fgrep "Executed" test_run.log >> summary.log
fgrep "FAILED" test_run.log | grep "out of" | sed -e 's/[ ][ ]*.*//' -e 's/$/ FAILED/' > test_failures.log
fgrep "TIMEOUT" test_run.log | grep "out of" | sed -e 's/[ ][ ]*.*//' -e 's/$/ TIMEOUT/' >> test_failures.log
count=$(wc -l < test_failures.log)
[ $count -gt 0 ] && ret=1

exit $ret
