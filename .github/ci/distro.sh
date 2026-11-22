#!/bin/bash
# Run inside a distribution container (rockylinux:8/9, opensuse/leap:15.6) as root, with the repository mounted at
# /src: install the distribution's compiler, Open MPI, GSL and Python, build FFTW3 from
# /src/ci_cache/fftw-$FFTW_VERSION.tar.gz into /src/ci_cache/fftw-<distro> as in the installation docs (kept between
# runs by the workflow's cache), then pip install Aarambam and run every test on the installed package as an
# unprivileged user (Open MPI refuses root). Fails if a test fails or is skipped.
set -euo pipefail
. /etc/os-release
: "${FFTW_VERSION:=3.3.10}"

case "$ID" in
  rocky|rhel|almalinux)
    dnf -y install dnf-plugins-core      #gsl-devel is in PowerTools (8) / CRB (9)
    if [ "${VERSION_ID%%.*}" = 8 ]; then dnf config-manager --set-enabled powertools; else dnf config-manager --set-enabled crb; fi
    dnf -y install gcc make which procps-ng tar gzip openmpi-devel gsl-devel python3.11 python3.11-devel python3.11-pip
    mpi_bin=/usr/lib64/openmpi/bin ;;
  opensuse-leap|sles)
    zypper -n install gcc make which tar gzip openmpi4-devel gsl-devel python311 python311-devel python311-pip
    mpi_bin=/usr/lib64/mpi/gcc/openmpi4/bin ;;
  *)
    echo "distro.sh: unsupported distribution $ID"; exit 1 ;;
esac

uid=$(stat -c %u /src)
id ci >/dev/null 2>&1 || useradd -m -u "$uid" ci

cat > /tmp/as_ci.sh <<EOF
set -euo pipefail
export PATH=$mpi_bin:\$PATH OMP_NUM_THREADS=1
mpicc --version | head -1; mpirun --version 2>&1 | head -1
prefix=/src/ci_cache/fftw-$ID${VERSION_ID%%.*}      #major version: the minor releases of an image share the cache
if [ ! -e \$prefix/lib/libfftw3_mpi.so ] || [ ! -e \$prefix/lib/libfftw3f.so ]; then
  rm -rf /tmp/fftw && mkdir -p /tmp/fftw && cd /tmp/fftw
  tar -xzf /src/ci_cache/fftw-$FFTW_VERSION.tar.gz && cd fftw-$FFTW_VERSION
  ./configure --prefix=\$prefix --enable-mpi --enable-shared --enable-sse2 --enable-avx --enable-avx2 MPICC=mpicc
  make -j4 && make install && cd .. && rm -rf fftw-$FFTW_VERSION
  tar -xzf /src/ci_cache/fftw-$FFTW_VERSION.tar.gz && cd fftw-$FFTW_VERSION
  ./configure --prefix=\$prefix --enable-float --enable-shared --enable-sse --enable-sse2 --enable-avx --enable-avx2
  make -j4 && make install
fi
export Aarambam_FFTW3_ROOT=\$prefix
python3.11 -m venv ~/venv && . ~/venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -v /src pytest
cd /src
AARAMBAM_TEST_INSTALLED=1 python -m pytest tests -rs -p no:cacheprovider | tee /tmp/pytest.log
if grep -q SKIPPED /tmp/pytest.log; then echo "tests were skipped: a compiled part is missing"; exit 1; fi
EOF
runuser -u ci -- bash /tmp/as_ci.sh
