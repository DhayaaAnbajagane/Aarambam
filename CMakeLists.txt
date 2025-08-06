cmake_minimum_required(VERSION 3.16)
project(Aarambam C)

# --- Compiler: use mpicc like the Makefile ---
# You can also pass -DCMAKE_C_COMPILER=... on the cmake command line.
if(NOT CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER "/software/openmpi-4.1.0-el8-x86_64/bin/mpicc")
endif()

# --- Match Makefile flags: -O3 -Wall and GNU extensions (like drand48, M_PI) ---
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)           # => gnu99, like GCC default used by the Makefile
add_compile_options(-O3 -Wall -g)

# --- Sources (OBJS without .o) ---
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Aarambam/LPT/Basis/)
add_executable(2LPTBasis
  ${SRC_DIR}/main.c
  ${SRC_DIR}/power.c
  ${SRC_DIR}/checkchoose.c
  ${SRC_DIR}/allvars.c
  ${SRC_DIR}/save.c
  ${SRC_DIR}/read_param.c
  ${SRC_DIR}/read_glass.c
  ${SRC_DIR}/nrsrc/nrutil.c
  ${SRC_DIR}/nrsrc/qromb.c
  ${SRC_DIR}/nrsrc/polint.c
  ${SRC_DIR}/nrsrc/trapzd.c
)

# --- Include paths (CFLAGS part) ---
# GSL
target_include_directories(2LPTBasis PRIVATE
  /software/gsl-2.7-el8-x86_64/include
)
# FFTW2 (use THEPATH env var exactly like your Makefile)
if(DEFINED ENV{THEPATH})
  target_include_directories(2LPTBasis PRIVATE
    $ENV{THEPATH}/FFTW/fftw-2.1.5/build/include
  )
endif()
# Local headers
target_include_directories(2LPTBasis PRIVATE
  ${SRC_DIR}
  ${SRC_DIR}/nrsrc
)

# --- Preprocessor options (OPT / OPTIONS in the Makefile) ---
# Default Makefile switches:
target_compile_definitions(2LPTBasis PRIVATE
  NO64BITID=1
  CORRECT_CIC=1
  # PRODUCEGAS=1   # uncomment if you want this default ON
  # ONLY_ZA=1      # uncomment if you want this default ON
)
# Optional “MODE” passthrough (you can pass -DMODE="-DSOMETHING" to cmake)  
if(DEFINED MODE AND NOT MODE STREQUAL "")
  # MODE is expected to be something like "-DSOMEFLAG"
  target_compile_options(2LPTBasis PRIVATE ${MODE})
endif()

# --- Link directories (LIBS part: -L...) ---
target_link_directories(2LPTBasis PRIVATE
  /software/openmpi-4.1.0-el8-x86_64/lib
  /software/gsl-2.7-el8-x86_64/lib
)
if(DEFINED ENV{THEPATH})
  target_link_directories(2LPTBasis PRIVATE
    $ENV{THEPATH}/FFTW/fftw-2.1.5/build/lib
  )
endif()

# --- Link libraries (order matches your Makefile) ---
#   LIBS = -lm  $(MPICHLIB)  $(FFTW_LIB)  $(GSL_LIBS)  -lgsl -lgslcblas
#   $(FFTW_LIB) expands to: -L... -ldrfftw_mpi -ldfftw_mpi -ldrfftw -ldfftw
target_link_libraries(2LPTBasis PRIVATE
  m
  mpi
  drfftw_mpi dfftw_mpi drfftw dfftw
  gsl gslcblas
)
install(TARGETS 2LPTBasis RUNTIME DESTINATION Aarambam/bin)

# You can `cmake --build . --target clean` for cleaning; no manual 'clean' needed.


# ─── CMB-BEST Python Extension ──────────────────────────────────────────────
# requires: CMake >= 3.18 (for Python3_add_library), Cython & numpy in your build env

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(OpenMP)  # for -fopenmp

# paths
set(CMB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Aarambam/basis)
set(CMB_PYX    ${CMB_SRC_DIR}/cmbbest.pyx)
set(CMB_C      ${CMAKE_CURRENT_BINARY_DIR}/Aarambam/basis/cmbbest.c)

# 1 generate cmbbest.c
add_custom_command(
  OUTPUT ${CMB_C}
  COMMAND ${Python3_EXECUTABLE} -m cython -3 --fast-fail -o ${CMB_C} ${CMB_PYX}
  DEPENDS ${CMB_PYX} ${CMB_SRC_DIR}/tetrapyd.c ${CMB_SRC_DIR}/arrays.c
  COMMENT "Cythonizing ${CMB_PYX}"
  VERBATIM
)

# 2 build the extension
if(COMMAND Python3_add_library)
  Python3_add_library(cmbbest MODULE WITH_SOABI
    ${CMB_C}
    ${CMB_SRC_DIR}/tetrapyd.c
    ${CMB_SRC_DIR}/arrays.c
  )
else()
  add_library(cmbbest MODULE
    ${CMB_C}
    ${CMB_SRC_DIR}/tetrapyd.c
    ${CMB_SRC_DIR}/arrays.c
  )
  # fallback suffix
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
      "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so', end='')"
    OUTPUT_VARIABLE _ext
  )
  set_target_properties(cmbbest PROPERTIES SUFFIX "${_ext}")
endif()

# 3 include & link
target_include_directories(cmbbest PRIVATE
  ${Python3_NumPy_INCLUDE_DIRS}
  ${CMB_SRC_DIR}
)
if(OpenMP_C_FOUND)
  target_link_libraries(cmbbest PRIVATE OpenMP::OpenMP_C)
  target_compile_options(cmbbest PRIVATE -fopenmp)
  target_link_options(cmbbest PRIVATE -fopenmp)
endif()

# 4 install into your package
install(TARGETS cmbbest
  RUNTIME DESTINATION Aarambam/basis      # no-op on MODULE, but fine
  LIBRARY DESTINATION Aarambam/basis      # .so goes here under site-packages/Aarambam/basis
)
