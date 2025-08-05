#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "allvars.h"
#include "proto.h"


void read_parameterfile(char *fname)
{
#define FLOAT 1
#define STRING 2
#define INT 3
#define MAXTAGS 300

  FILE *fd;
  char buf[200], buf1[200], buf2[200], buf3[200];
  int i, j, nt;
  int id[MAXTAGS];
  void *addr[MAXTAGS];
  char tag[MAXTAGS][50];
  int errorFlag = 0;

  /* read parameter file on all processes for simplicty */

  nt = 0;

  strcpy(tag[nt], "Omega");
  addr[nt] = &Omega;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "OmegaLambda");
  addr[nt] = &OmegaLambda;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "OmegaBaryon");
  addr[nt] = &OmegaBaryon;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "OmegaDM_2ndSpecies");
  addr[nt] = &OmegaDM_2ndSpecies;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "HubbleParam");
  addr[nt] = &HubbleParam;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "ShapeGamma");
  addr[nt] = &ShapeGamma;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "Sigma8");
  addr[nt] = &Sigma8;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "PrimordialIndex");
  addr[nt] = &PrimordialIndex;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "Box");
  addr[nt] = &Box;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "Redshift");
  addr[nt] = &Redshift;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "Fnl");
  addr[nt] = &Fnl;
  id[nt++] = FLOAT;
    
// ********** DHAYAA *************
    
  strcpy(tag[nt], "N_modes");
  addr[nt] = &N_modes;
  id[nt++] = INT;
    
  strcpy(tag[nt], "FileWithAlphaCoeff");
  addr[nt] = FileWithAlphaCoeff;
  id[nt++] = STRING;
    
  strcpy(tag[nt], "FileWithX0Coeff");
  addr[nt] = FileWithX0Coeff;
  id[nt++] = STRING;
    
// ********** DHAYAA *************
    

// ********** FAVN/DSJ  ************
  strcpy(tag[nt], "FixedAmplitude");
  addr[nt] = &FixedAmplitude;
  id[nt++] = INT;

  strcpy(tag[nt], "PhaseFlip");
  addr[nt] = &PhaseFlip;
  id[nt++] = INT;
// ********** FAVN/DSJ  ************

  strcpy(tag[nt], "Nmesh");
  addr[nt] = &Nmesh;
  id[nt++] = INT;

  strcpy(tag[nt], "Nsample");
  addr[nt] = &Nsample;
  id[nt++] = INT;

  strcpy(tag[nt], "Nmax");
  addr[nt] = &Nmax;
  id[nt++] = INT;

  strcpy(tag[nt], "GlassFile");
  addr[nt] = GlassFile;
  id[nt++] = STRING;

  strcpy(tag[nt], "FileWithInputSpectrum");
  addr[nt] = FileWithInputSpectrum;
  id[nt++] = STRING;

  strcpy(tag[nt], "FileWithInputTransfer");
  addr[nt] = FileWithInputTransfer;
  id[nt++] = STRING;

  strcpy(tag[nt], "GlassTileFac");
  addr[nt] = &GlassTileFac;
  id[nt++] = INT;

  strcpy(tag[nt], "Seed");
  addr[nt] = &Seed;
  id[nt++] = INT;

  strcpy(tag[nt], "SphereMode");
  addr[nt] = &SphereMode;
  id[nt++] = INT;

  strcpy(tag[nt], "NumFilesWrittenInParallel");
  addr[nt] = &NumFilesWrittenInParallel;
  id[nt++] = INT;

  strcpy(tag[nt], "OutputDir");
  addr[nt] = OutputDir;
  id[nt++] = STRING;

  strcpy(tag[nt], "FileBase");
  addr[nt] = FileBase;
  id[nt++] = STRING;

  strcpy(tag[nt], "WhichSpectrum");
  addr[nt] = &WhichSpectrum;
  id[nt++] = INT;

  strcpy(tag[nt], "WhichTransfer");
  addr[nt] = &WhichTransfer;
  id[nt++] = INT;

  strcpy(tag[nt], "UnitVelocity_in_cm_per_s");
  addr[nt] = &UnitVelocity_in_cm_per_s;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "UnitLength_in_cm");
  addr[nt] = &UnitLength_in_cm;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "UnitMass_in_g");
  addr[nt] = &UnitMass_in_g;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "InputSpectrum_UnitLength_in_cm");
  addr[nt] = &InputSpectrum_UnitLength_in_cm;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "WDM_On");
  addr[nt] = &WDM_On;
  id[nt++] = INT;

  strcpy(tag[nt], "WDM_Vtherm_On");
  addr[nt] = &WDM_Vtherm_On;
  id[nt++] = INT;

  strcpy(tag[nt], "WDM_PartMass_in_kev");
  addr[nt] = &WDM_PartMass_in_kev;
  id[nt++] = FLOAT;

  if((fd = fopen(fname, "r")))
    {
      while(!feof(fd))
	{
	  buf[0] = 0;
	  fgets(buf, 200, fd);

	  if(sscanf(buf, "%s%s%s", buf1, buf2, buf3) < 2)
	    continue;

	  if(buf1[0] == '%')
	    continue;

	  for(i = 0, j = -1; i < nt; i++)
	    if(strcmp(buf1, tag[i]) == 0)
	      {
		j = i;
		tag[i][0] = 0;
		break;
	      }

	  if(j >= 0)
	    {
	      switch (id[j])
		{
		case FLOAT:
		  *((double *) addr[j]) = atof(buf2);
		  break;
		case STRING:
		  strcpy(addr[j], buf2);
		  break;
		case INT:
		  *((int *) addr[j]) = atoi(buf2);
		  break;
		}
	    }
	  else
	    {
	      if(ThisTask == 0)
		fprintf(stdout, "Error in file %s:   Tag '%s' not allowed or multiple defined.\n", fname,
			buf1);
	      errorFlag = 1;
	    }
	}
      fclose(fd);

    }
  else
    {
      if(ThisTask == 0)
	fprintf(stdout, "Parameter file %s not found.\n", fname);
      errorFlag = 1;
    }


  for(i = 0; i < nt; i++)
    {
      if(*tag[i])
	{
	  if(ThisTask == 0)
	    fprintf(stdout, "Error. I miss a value for tag '%s' in parameter file '%s'.\n", tag[i], fname);
	  errorFlag = 1;
	}
    }

  if(errorFlag)
    {
      MPI_Finalize();
      exit(0);
    }


#undef FLOAT
#undef STRING
#undef INT
#undef MAXTAGS
}


#include <mpi.h>
void broadcast_parameters()
{
    // Broadcast all FLOAT parameters
    MPI_Bcast(&Omega, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&OmegaLambda, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&OmegaBaryon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&OmegaDM_2ndSpecies, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&HubbleParam, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ShapeGamma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Sigma8, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&PrimordialIndex, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Box, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Redshift, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Fnl, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&UnitVelocity_in_cm_per_s, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&UnitLength_in_cm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&UnitMass_in_g, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&InputSpectrum_UnitLength_in_cm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&WDM_PartMass_in_kev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast all INT parameters
    MPI_Bcast(&N_modes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&FixedAmplitude, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&PhaseFlip, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Nmesh, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Nsample, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Nmax, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&GlassTileFac, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&SphereMode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NumFilesWrittenInParallel, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&WhichSpectrum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&WhichTransfer, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&WDM_On, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&WDM_Vtherm_On, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast all STRING parameters (use buffer size large enough)
    MPI_Bcast(FileWithAlphaCoeff, 500, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(FileWithX0Coeff, 500, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(GlassFile, 500, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(FileWithInputSpectrum, 500, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(FileWithInputTransfer, 500, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(OutputDir, 500, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(FileBase, 500, MPI_CHAR, 0, MPI_COMM_WORLD);
}