#include <math.h>
#include <stdlib.h>

#include "allvars.h"
#include "proto.h"


//wrc
#include <stdio.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

void write_phi(fftw_real *pot, int isNonGaus)
{
    char buf_will[400];
    FILE *write_ptr;
    int total_size = TotalSizePlusAdditional;
    int i, j, k;

    // Local buffer size for the potential array
    int local_pot_size = Local_nx * Nmesh * (2 * (Nmesh / 2 + 1));

    if (ThisTask != 0)
    {
        // Send metadata and local buffer to Task 0
        MPI_Send(&Local_x_start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&Local_nx,      1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&local_pot_size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(pot, local_pot_size, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD); // fftw_real assumed to be double
        return;
    }

    // === Task 0 writes for each task ===
    for (int task = 0; task < NTask; task++)
    {
        int start_x, nx, recv_size;
        fftw_real *buffer = NULL;

        if (task == 0)
        {
            start_x = Local_x_start;
            nx = Local_nx;
            recv_size = local_pot_size;
            buffer = pot;
        }
        else
        {
            MPI_Recv(&start_x,   1, MPI_INT, task, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&nx,        1, MPI_INT, task, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&recv_size, 1, MPI_INT, task, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            buffer = malloc(recv_size * sizeof(fftw_real));
            if (!buffer)
            {
                printf("Task 0: failed to allocate buffer from task %d\n", task);
                FatalError(12345);
            }

            MPI_Recv(buffer, recv_size, MPI_DOUBLE, task, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Construct filename based on original logic
        if (isNonGaus == 0)
        {
            if (NTaskWithN > 1)
                sprintf(buf_will, "%s/Gauss_potential_%d_StartX_%04d_NumX_%d", OutputDir, Seed, start_x, nx);
            else
                sprintf(buf_will, "%s/Gauss_potential_%d", OutputDir, Seed);
        }
        else
        {
            if (NTaskWithN > 1)
                sprintf(buf_will, "%s/Nongauss_potential_%d_StartX_%04d_NumX_%d", OutputDir, Seed, start_x, nx);
            else
                sprintf(buf_will, "%s/Nongauss_potential_%d", OutputDir, Seed);
        }

        if (!(write_ptr = fopen(buf_will, "w")))
        {
            printf("Error. Can't write in file '%s'\n", buf_will);
            FatalError(10);
        }

        // Write header
        my_fwrite(&total_size, sizeof(total_size), 1, write_ptr);
        my_fwrite(&Nmesh, sizeof(Nmesh), 1, write_ptr);
        my_fwrite(&Box, sizeof(Box), 1, write_ptr);

        // Write data
        for (i = 0; i < nx; i++)
          for (j = 0; j < Nmesh; j++)
            for (k = 0; k < Nmesh; k++)
            {
                int coord = (i * Nmesh + j) * (2 * (Nmesh / 2 + 1)) + k;
                my_fwrite(&coord, sizeof(int), 1, write_ptr);
                my_fwrite(&buffer[coord], sizeof(fftw_real), 1, write_ptr);
            }

        fclose(write_ptr);

        if (task != 0)
            free(buffer);
    }
}


void write_particle_data(void)
{
  int nprocgroup, groupTask, masterTask;

  if((NTask < NumFilesWrittenInParallel))
    {
      printf
	("Fatal error.\nNumber of processors must be a smaller or equal than `NumFilesWrittenInParallel'.\n");
      FatalError(24131);
    }

  for (int task = 0; task < NTask; task++) {
  if (ThisTask == 0) {
    if (task == 0) {
      // Task 0 writes its own data
      save_local_data(0);
    } else {
      // Receive from other task and write on their behalf
      MPI_Status status;
      int numpart;
      MPI_Recv(&numpart, 1, MPI_INT, task, 0, MPI_COMM_WORLD, &status);

      struct part_data *recvbuf = malloc(numpart * sizeof(struct part_data));
      MPI_Recv(recvbuf, numpart * sizeof(struct part_data), MPI_BYTE, task, 1, MPI_COMM_WORLD, &status);

      // Temporarily replace P and NumPart
      struct part_data *P_old = P;
      int NumPart_old = NumPart;

      P = recvbuf;
      NumPart = numpart;

      save_local_data(task);  // Write to FileBase.task

      free(P);
      P = P_old;
      NumPart = NumPart_old;
    }
  } else {
    if (ThisTask == task) {
      MPI_Send(&NumPart, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      MPI_Send(P, NumPart * sizeof(struct part_data), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);  // Synchronize
}


}




void save_local_data(int task)
{
#define BUFFER 10
  size_t bytes;
  float *block;
  int *blockid;
#ifndef NO64BITID
  long long *blocklongid;
#endif
  int blockmaxlen, maxlongidlen;
  int4byte dummy;
  FILE *fd;
  char buf[300];
  int i, k, pc;
#ifdef  PRODUCEGAS
  double meanspacing, shift_gas, shift_dm;
#endif


  if(NumPart == 0)
    return;

  if(NTaskWithN > 1)
    sprintf(buf, "%s/%s.%d", OutputDir, FileBase, task);
  else
    sprintf(buf, "%s/%s", OutputDir, FileBase);

  if(!(fd = fopen(buf, "w")))
    {
      printf("Error. Can't write in file '%s'\n", buf);
      FatalError(10);
    }

  for(i = 0; i < 6; i++)
    {
      header.npart[i] = 0;
      header.npartTotal[i] = 0;
      header.mass[i] = 0;
    }


#ifdef MULTICOMPONENTGLASSFILE
  qsort(P, NumPart, sizeof(struct part_data), compare_type);  /* sort particles by type, because that's how they should be stored in a gadget binary file */

  for(i = 0; i < 3; i++)
    header.npartTotal[i] = header1.npartTotal[i + 1] * GlassTileFac * GlassTileFac * GlassTileFac;

  for(i = 0; i < NumPart; i++)
    header.npart[P[i].Type]++;

  if(header.npartTotal[0])
    header.mass[0] =
      (OmegaBaryon) * 3 * Hubble * Hubble / (8 * PI * G) * pow(Box, 3) / (header.npartTotal[0]);

  if(header.npartTotal[1])
    header.mass[1] =
      (Omega - OmegaBaryon - OmegaDM_2ndSpecies) * 3 * Hubble * Hubble / (8 * PI * G) * pow(Box,
											    3) /
      (header.npartTotal[1]);

  if(header.npartTotal[2])
    header.mass[2] =
      (OmegaDM_2ndSpecies) * 3 * Hubble * Hubble / (8 * PI * G) * pow(Box, 3) / (header.npartTotal[2]);


#else

  header.npart[1] = NumPart;
  header.npartTotal[1] = TotNumPart;
  header.npartTotal[2] = (TotNumPart >> 32);
  header.mass[1] = (Omega) * 3 * Hubble * Hubble / (8 * PI * G) * pow(Box, 3) / TotNumPart;

#ifdef  PRODUCEGAS
  header.npart[0] = NumPart;
  header.npartTotal[0] = TotNumPart;
  header.mass[0] = (OmegaBaryon) * 3 * Hubble * Hubble / (8 * PI * G) * pow(Box, 3) / TotNumPart;
  header.mass[1] = (Omega - OmegaBaryon) * 3 * Hubble * Hubble / (8 * PI * G) * pow(Box, 3) / TotNumPart;
#endif
#endif


  header.time = InitTime;
  header.redshift = 1.0 / InitTime - 1;

  header.flag_sfr = 0;
  header.flag_feedback = 0;
  header.flag_cooling = 0;
  header.flag_stellarage = 0;
  header.flag_metals = 0;

  header.num_files = NTaskWithN;

  header.BoxSize = Box;
  header.Omega0 = Omega;
  header.OmegaLambda = OmegaLambda;
  header.HubbleParam = HubbleParam;

  header.flag_stellarage = 0;
  header.flag_metals = 0;
  header.hashtabsize = 0;

  dummy = sizeof(header);
  my_fwrite(&dummy, sizeof(dummy), 1, fd);
  my_fwrite(&header, sizeof(header), 1, fd);
  my_fwrite(&dummy, sizeof(dummy), 1, fd);

#ifdef  PRODUCEGAS
  meanspacing = Box / pow(TotNumPart, 1.0 / 3);
  shift_gas = -0.5 * (Omega - OmegaBaryon) / (Omega) * meanspacing;
  shift_dm = +0.5 * OmegaBaryon / (Omega) * meanspacing;
#endif


  if(!(block = malloc(bytes = BUFFER * 1024 * 1024)))
    {
      printf("failed to allocate memory for `block' (%g bytes).\n", (double)bytes);
      FatalError(24);
    }

  blockmaxlen = bytes / (3 * sizeof(float));

  blockid = (int *) block;
#ifndef NO64BITID
  blocklongid = (long long *) block;
#endif
  maxlongidlen = bytes / (sizeof(long long));

  /* write coordinates */
  dummy = sizeof(float) * 3 * NumPart;
#ifdef  PRODUCEGAS
  dummy *= 2;
#endif
  my_fwrite(&dummy, sizeof(dummy), 1, fd);
  for(i = 0, pc = 0; i < NumPart; i++)
    {
      for(k = 0; k < 3; k++)
	{
	  block[3 * pc + k] = P[i].Pos[k];
#ifdef  PRODUCEGAS
	  block[3 * pc + k] = periodic_wrap(P[i].Pos[k] + shift_gas);
#endif
	}

      pc++;

      if(pc == blockmaxlen)
	{
	  my_fwrite(block, sizeof(float), 3 * pc, fd);
	  pc = 0;
	}
    }
  if(pc > 0)
    my_fwrite(block, sizeof(float), 3 * pc, fd);
#ifdef  PRODUCEGAS
  for(i = 0, pc = 0; i < NumPart; i++)
    {
      for(k = 0; k < 3; k++)
	{
	  block[3 * pc + k] = periodic_wrap(P[i].Pos[k] + shift_dm);
	}

      pc++;

      if(pc == blockmaxlen)
	{
	  my_fwrite(block, sizeof(float), 3 * pc, fd);
	  pc = 0;
	}
    }
  if(pc > 0)
    my_fwrite(block, sizeof(float), 3 * pc, fd);
#endif
  my_fwrite(&dummy, sizeof(dummy), 1, fd);



  /* write velocities */
  dummy = sizeof(float) * 3 * NumPart;
#ifdef  PRODUCEGAS
  dummy *= 2;
#endif
  my_fwrite(&dummy, sizeof(dummy), 1, fd);
  for(i = 0, pc = 0; i < NumPart; i++)
    {
      for(k = 0; k < 3; k++)
	block[3 * pc + k] = P[i].Vel[k];

#ifdef MULTICOMPONENTGLASSFILE
      if(WDM_On == 1 && WDM_Vtherm_On == 1 && P[i].Type == 1)
	add_WDM_thermal_speeds(&block[3 * pc]);
#else
#ifndef PRODUCEGAS
      if(WDM_On == 1 && WDM_Vtherm_On == 1)
	add_WDM_thermal_speeds(&block[3 * pc]);
#endif
#endif

      pc++;

      if(pc == blockmaxlen)
	{
	  my_fwrite(block, sizeof(float), 3 * pc, fd);
	  pc = 0;
	}
    }
  if(pc > 0)
    my_fwrite(block, sizeof(float), 3 * pc, fd);
#ifdef PRODUCEGAS
  for(i = 0, pc = 0; i < NumPart; i++)
    {
      for(k = 0; k < 3; k++)
	block[3 * pc + k] = P[i].Vel[k];

      if(WDM_On == 1 && WDM_Vtherm_On == 1)
	add_WDM_thermal_speeds(&block[3 * pc]);

      pc++;

      if(pc == blockmaxlen)
	{
	  my_fwrite(block, sizeof(float), 3 * pc, fd);
	  pc = 0;
	}
    }
  if(pc > 0)
    my_fwrite(block, sizeof(float), 3 * pc, fd);
#endif
  my_fwrite(&dummy, sizeof(dummy), 1, fd);


  /* write particle ID */
#ifdef NO64BITID
  dummy = sizeof(int) * NumPart;
#else
  dummy = sizeof(long long) * NumPart;
#endif
#ifdef  PRODUCEGAS
  dummy *= 2;
#endif
  my_fwrite(&dummy, sizeof(dummy), 1, fd);
  for(i = 0, pc = 0; i < NumPart; i++)
    {
#ifdef NO64BITID
      blockid[pc] = P[i].ID;
#else
      blocklongid[pc] = P[i].ID;
#endif

      pc++;

      if(pc == maxlongidlen)
	{
#ifdef NO64BITID
	  my_fwrite(blockid, sizeof(int), pc, fd);
#else
	  my_fwrite(blocklongid, sizeof(long long), pc, fd);
#endif
	  pc = 0;
	}
    }
  if(pc > 0)
    {
#ifdef NO64BITID
      my_fwrite(blockid, sizeof(int), pc, fd);
#else
      my_fwrite(blocklongid, sizeof(long long), pc, fd);
#endif
    }

#ifdef PRODUCEGAS
  for(i = 0, pc = 0; i < NumPart; i++)
    {
#ifdef NO64BITID
      blockid[pc] = P[i].ID + TotNumPart;
#else
      blocklongid[pc] = P[i].ID + TotNumPart;
#endif

      pc++;

      if(pc == maxlongidlen)
	{
#ifdef NO64BITID
	  my_fwrite(blockid, sizeof(int), pc, fd);
#else
	  my_fwrite(blocklongid, sizeof(long long), pc, fd);
#endif
	  pc = 0;
	}
    }
  if(pc > 0)
    {
#ifdef NO64BITID
      my_fwrite(blockid, sizeof(int), pc, fd);
#else
      my_fwrite(blocklongid, sizeof(long long), pc, fd);
#endif
    }
#endif

  my_fwrite(&dummy, sizeof(dummy), 1, fd);





  /* write zero temperatures if needed */
#ifdef  PRODUCEGAS
  dummy = sizeof(float) * NumPart;
  my_fwrite(&dummy, sizeof(dummy), 1, fd);
  for(i = 0, pc = 0; i < NumPart; i++)
    {
      block[pc] = 0;

      pc++;

      if(pc == blockmaxlen)
	{
	  my_fwrite(block, sizeof(float), pc, fd);
	  pc = 0;
	}
    }
  if(pc > 0)
    my_fwrite(block, sizeof(float), pc, fd);
  my_fwrite(&dummy, sizeof(dummy), 1, fd);
#endif


  /* write zero temperatures if needed */
#ifdef  MULTICOMPONENTGLASSFILE
  if(header.npart[0])
    {
      dummy = sizeof(float) * header.npart[0];
      my_fwrite(&dummy, sizeof(dummy), 1, fd);

      for(i = 0, pc = 0; i < header.npart[0]; i++)
	{
	  block[pc] = 0;

	  pc++;

	  if(pc == blockmaxlen)
	    {
	      my_fwrite(block, sizeof(float), pc, fd);
	      pc = 0;
	    }
	}
      if(pc > 0)
	my_fwrite(block, sizeof(float), pc, fd);
      my_fwrite(&dummy, sizeof(dummy), 1, fd);
    }
#endif



  free(block);

  fclose(fd);
}


/* This catches I/O errors occuring for my_fwrite(). In this case we better stop.
 */
size_t my_fwrite(void *ptr, size_t size, size_t nmemb, FILE * stream)
{
  size_t nwritten;

  if((nwritten = fwrite(ptr, size, nmemb, stream)) != nmemb)
    {
      printf("I/O error (fwrite) on task=%d has occured.\n", ThisTask);
      fflush(stdout);
      perror("fwrite");
      FatalError(777);
    }
  return nwritten;
}


/* This catches I/O errors occuring for fread(). In this case we better stop.
 */
size_t my_fread(void *ptr, size_t size, size_t nmemb, FILE * stream)
{
  size_t nread;

  if((nread = fread(ptr, size, nmemb, stream)) != nmemb)
    {
      printf("I/O error (fread) on task=%d has occured.\n", ThisTask);
      fflush(stdout);
      FatalError(778);
    }
  return nread;
}


#ifdef MULTICOMPONENTGLASSFILE
int compare_type(const void *a, const void *b)
{
  if(((struct part_data *) a)->Type < (((struct part_data *) b)->Type))
    return -1;

  if(((struct part_data *) a)->Type > (((struct part_data *) b)->Type))
    return +1;

  return 0;
}
#endif
