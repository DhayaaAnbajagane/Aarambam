EXEC   = 2LPTnonlocal

OBJS   = main.o power.o checkchoose.o allvars.o save.o read_param.o  read_glass.o  \
         nrsrc/nrutil.o nrsrc/qromb.o nrsrc/polint.o nrsrc/trapzd.o

INCL   = allvars.h proto.h  nrsrc/nrutil.h  Makefile



#OPT   +=  -DPRODUCEGAS   # Set this to automatically produce gas particles 
                          # for a single DM species in the input file by interleaved by a half a grid spacing
                          # only for Gaussian and ZA


OPT   +=  -DNO64BITID     # switch this on if you want normal 32-bit IDs

OPT  +=  -DCORRECT_CIC  # only switch this on if particles start from a glass (as opposed to grid)
                         # only for Gaussian and ZA

#OPT += -DONLY_ZA    # swith this on if you want ZA initial conditions (2LPT otherwise)

EXEC:=2LPTNGBasis
OPT += $(MODE)
OPTIONS =  $(OPT)

GSL_INCL= -I/software/gsl-2.7-el8-x86_64/include
GSL_LIBS= -L/software/gsl-2.7-el8-x86_64/lib
FFTW_INCL=-I$(THEPATH)/FFTW/fftw-2.1.5/build/include
FFTW_LIBS=-L$(THEPATH)/FFTW/fftw-2.1.5/build/lib -ldrfftw_mpi -ldfftw_mpi -ldrfftw -ldfftw

MPICHLIB = -L/software/openmpi-4.1.0-el8-x86_64/lib -lmpi
CC       =  /software/openmpi-4.1.0-el8-x86_64/bin/mpicc #-g -Wall -fbounds-check    # sets the C-compiler (default)

OPTIMIZE =   -O3 -Wall    # optimization and warning flags (default)
FFTW_LIB =  $(FFTW_LIBS)

LIBS   =   -lm  $(MPICHLIB)  $(FFTW_LIB)  $(GSL_LIBS)  -lgsl -lgslcblas
CFLAGS =   $(OPTIONS)  $(OPTIMIZE)  $(FFTW_INCL) $(GSL_INCL)

$(EXEC): $(OBJS) 
	$(CC) -g $(OPTIMIZE) $(OBJS) $(LIBS)   -o  $(EXEC) 
	rm -f $(OBJS)

$(OBJS): $(INCL) 


.PHONY : clean
clean:
	rm -f $(OBJS) 2LPTNGBasis

