/*
 * This software is a copyright (c) of Carnegie Mellon University, 2002.
 *           
 * Permission to reproduce, use and prepare derivative works of this
 * software for use is granted provided the copyright and "No Warranty"
 * statements are included with all reproductions and derivative works.
 * This software may also be redistributed provided that the copyright
 * and "No Warranty" statements are included in all redistributions.
 *           
 * NO WARRANTY. THIS SOFTWARE IS FURNISHED ON AN "AS IS" BASIS.  CARNEGIE
 * MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR
 * IMPLIED AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF
 * FITNESS FOR PURPOSE OR MERCHANABILITY, EXCLUSIVITY OF RESULTS OR
 * RESULTS OBTAINED FROM USE OF THIS SOFTWARE. CARNEGIE MELLON UNIVERSITY
 * DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM
 * PATENT, TRADEMARK OR COPYRIGHT INFRINGEMENT.
 *
 * Carnegie Mellon encourages (but does not require) users of this
 * software to return any improvements or extensions that they make, and
 * to grant Carnegie Mellon the rights to redistribute these changes
 * without encumbrance.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "gnp.h"
#include "simplex_downhill.h"
#include "mylib.h"

/*#define VERSION 6*/  /* need to change this every time we update the
			  input file format so we don't accidentally reuse
			  and old input file */

/*#define VERSION 7*/  /* added dump_fine_data switch */
/*#define VERSION 8*/  /* added skewed normalized error */
/*#define VERSION 9*/  /* added gd switch, powell method */
/*#define VERSION 10*/ /* added landmark subset selection */
#define VERSION 11     /* added dump_computed matrix/combined dump */

#define MAX_ERR 1e20


#define PI 3.1415926

int gd; /* 0 gradient decent, 1 downhill simplex, 2 powell */
/* since we have discovered that lambda makes a large difference,
   we need to dynamically explore it */
/* int lambda = 1000; */      /* initial simplex lambda */
/* #define TRY 200*/        /* number of repetitions when fitting */
int range;
int lambda;
float ftol; /* tolerance of convergence for simplex downhill */
int try;  /* for models */
int try2; /* for targets */
int restarts;
int really_random;
int subset; /* 0 don't do it; 1 nearest; 2 random */
int num_subset; /* number of Landmarks to use in the subset > dimension */
int dump_computed; /* 1 output computed matrix/combined, number of models
		      assumed to be one */


/* basic data */
float **matrix;     /* measured distances between probes */
float **dt2p;       /* measured distances between probes and targets */
unsigned long *tip; /* ip addresses of targets */


// my basic data
float **myMatrix;
double criArray[150];
double criK;
double criMax = 0;


/* command line arguments */
int np;       /* number of probes altogether */
int nt;       /* number of targets altogether */
char *set;    /* name of the model set */


/* settings */
int ver;
embeddings_t embedding; /* see enum embeddings */
int param1, param2, param3;     /* parameters for the embedding */
int norm;               /* 1 (Manhattan), 2 (Euclidean), etc */
int normalized;         /* 0 or 1 or 2,
			   two ways to normalize,
			   2 means use log
			*/
int no_square;          /* 0 or 1, do not square */
mg_t mg;                /* 0 means read from file,
			   others means more intelligent
			   methods: use internal consistency to decide,
			   use some test data to train...
			*/

int full_mesh_data;     /* this is if we have full mesh of distances,
			   in that case there is no longer a distinction
			   between a target or a test node
			*/

int dump_fine_data;     /* dump individual files for each test probe */

/* configurations */
int var_per_loc;
float (*modelfit)(float *, int, void*);
float (*targetfit)(float *, int, void*);
float (*dist_func)(float *, float *);


/* run time */
int mp;       /* number of probes used in the model */
int *mpi;     /* array of indices of the model probes
		 int mpi[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	      */
int mps;      /* subset of model probes to use for final host fitting */
int *mpsi;    /* index of the subset of model probes */
int mid;      /* to index different models we generate in each run */
float *model; /* array of fitted coordinates of mp */
float *txy;   /* array for the model location of each target */
float *mxy;   /* array for the probes' coordinates, dump_computed */







int main(int argc, char** argv) {
  int i, j;
  FILE *f;
  char buf[100];
  struct in_addr a;
  float dis;

  if (argc != 4) {
    printf("Usage: %s num_probes num_targets model_file\n", argv[0]);
  }

  srand(0); /* always generate the same set of random number
	       when rand is used */

  np = atoi(argv[1]);
  nt = atoi(argv[2]);

  read_basic_data();

  set = argv[3];
  f = fopen(set, "r");
  fscanf(f, "%u", &ver);
  fscanf(f, "%[^\n]", buf);
  fgetc(f);
  if (ver != VERSION) {
    printf("Input file version does not match that of executable\n");
    exit(1);
  }

  fscanf(f, "%u", &embedding);
  fscanf(f, "%u", &param1);
  fscanf(f, "%u", &param2);
  fscanf(f, "%u", &param3);
  fscanf(f, "%u", &norm);
  fscanf(f, "%u", &normalized);
  fscanf(f, "%u", &no_square);
  fscanf(f, "%u", &range);
  fscanf(f, "%u", &lambda);
  fscanf(f, "%f", &ftol);
  fscanf(f, "%u", &try);
  fscanf(f, "%u", &try2);
  fscanf(f, "%u", &restarts);
  fscanf(f, "%u", &really_random);
  fscanf(f, "%u", &mg);
  fscanf(f, "%u", &full_mesh_data);
  fscanf(f, "%u", &dump_fine_data);
  fscanf(f, "%u", &gd);
  fscanf(f, "%u", &subset);
  fscanf(f, "%u", &num_subset);
  fscanf(f, "%u", &dump_computed);
  fscanf(f, "%[^\n]", buf);
  fgetc(f);
  switch (embedding) {
  case ND_SPACE:
    /* param1 is the number of dimensions */
    var_per_loc = param1;
    modelfit = fit_p1;
    printf("modelfit fit_p1 \n");
    targetfit = fit_p2_subset;
    dist_func = linear_dist;
    break;
  case CYLINDER:
    /* cylinder */
    var_per_loc = 2;
    /* param1 circumference, param2 height */
    modelfit = fit_p1;
    targetfit = fit_p2_subset;
    dist_func = cylindrical_dist;
    break;
  case SPHERE:
    /* sphere */
    var_per_loc = 2;
    /* param1 radius */
    modelfit = fit_p1;
    targetfit = fit_p2_subset;
    dist_func = spherical_dist;
    break;
  case D1_PLUS_D2_SPACE:
    var_per_loc = param1+param2;
    /* param1 is d1, param2 is the number of d2 dimensions,
       norm of d1 is specified in param3, norm of d2 is specified
       in norm */
    modelfit = fit_p1;
    targetfit = fit_p2_subset;
    dist_func = d1_plus_d2_dist;
    break;
  case LAND_MASS:
    /* land-mass */
    printf("Land mass not defined\n");
    exit(1);
    break;
  default:
    printf("Model ID not defined\n");
    exit(1);
  }

  if (subset && var_per_loc >= num_subset) {
    printf("num_subset is too small\n");
    exit(1);
  }
  if (dump_computed && full_mesh_data) {
    printf("dump_computed for full_mesh_data not yet implemented\n");
    exit(1);
  }

  txy = (float *)malloc(sizeof(float)*var_per_loc*nt);
  if (dump_computed) {
    mxy = (float *)malloc(sizeof(float)*var_per_loc*np);
  }
  
  switch (mg) {
  case SFILE:
    fscanf(f, "%u", &j); /* number of models to follow */
    if (dump_computed && j != 1) {
      printf("more than one model when dump_computed is requested\n");
      exit(1);
    }
    fscanf(f, "%[^\n]", buf);
    fgetc(f); /* new line */
    //get mymatrix
    
    
    /* set up mp and mpi */
    for (mid=0; mid<j; mid++) {
      fscanf(f, "%u", &mp);
      mpi = (int *)malloc(sizeof(int)*mp);
      mpsi = (int *)malloc(sizeof(int)*mp);
      for (i=0; i<mp; i++) {
        fscanf(f, "%u", &mpi[i]);
      }
      fscanf(f, "%[^\n]", buf);
      fgetc(f); /* new line */
      model = (float *)malloc(sizeof(float)*mp*var_per_loc);
      fitModel(0);
      int tmpi;
      for(tmpi = 1;tmpi < 10;tmpi ++){
        
        criK = ((PI / criMax) / 10) * tmpi;
        printf("myself function %d %f\n",tmpi,criK);
        convertMatrix();   
        fitModel(tmpi);
      }
     
    }
    
    break;
  default:
    printf("Model generation method not yet implemented\n");
    exit(1);
  }

  fclose(f);

  if (dump_computed) {
    f = fopen("genmatrix", "w");
    for (i=0; i<np; i++) {
      for (j=0; j<np; j++) {
	dis = (*dist_func)(&mxy[var_per_loc*i], &mxy[var_per_loc*j]);
	fprintf(f, "%f ", dis);
      }
      fprintf(f, "\n");
    }
    close(f);
    
    f = fopen("gencombined", "w");
    for (i=0; i<nt; i++) {
      a.s_addr = tip[i];
      fprintf(f, "%s ", inet_ntoa(a));
      for (j=0; j<np; j++) {
	dis = (*dist_func)(&txy[var_per_loc*i], &mxy[var_per_loc*j]);
	fprintf(f, "%f ", dis);      
      }
      fprintf(f,"\n");
    }
    fclose(f);

    free(mxy);
  }

  free(txy);

  return 0;
}

void saveConfig(FILE* f) {
  fprintf(f, "%u %u %u %u %u %u %u %u %u %f %u %u %u %u %u %u %u %u %u %u %u\n",
	  embedding, param1,
	  param2, param3, norm, normalized, no_square,
	  range, lambda, ftol, try, try2, restarts, really_random, mg,
	  full_mesh_data, dump_fine_data, gd, subset, num_subset, 
	  dump_computed);
}

/* use mp and mpi to construct the model key 0 origin 1 modified*/
void fitModel(int key) {
  printf("fitModel %d \n",key);
  int i, j;
  char buf[100];
  FILE *f;
  float error, dis1, dis2, sum;
  
  if (key == 0)
    error = solve(NULL, model, mp*var_per_loc,
		  (float (*)(float *, int, void*))(modelfit), try);
  else
    error = solve_myself(NULL, model,mp*var_per_loc,
      (float (*)(float *,int, void*))(fit_p1_myself),try);

  /* record useful info about the model we just created */
  sprintf(buf, "model.%s.%u.%d", set, mid,key);
  f = fopen(buf, "w");

  /* settings */
  saveConfig(f);

  /* model probes id list */
  fprintf(f, "%u ", mp);
  for (i=0; i<mp; i++) {
    fprintf(f, "%u ", mpi[i]);
  }
  fprintf(f, "\n");

  /* overall and average fitness */
  if (error == MAX_ERR) {
    fprintf(f, "Fatal ");
  } else {
    fprintf(f, "Normal ");
  }
  fprintf(f, "%f %f\n", error, error/((mp*(mp-1))/2));
    /* model/measured distances and errors between probe pairs */
  /*
  sum = 0;
  for (i=0; i<mp; i++) {
    for (j=0; j<i; j++) {
      dis1 = (*dist_func)(&model[var_per_loc*i], &model[var_per_loc*j]);
      dis2 = matrix[mpi[i]][mpi[j]];
      fprintf(f, "%u %u %f %f %f\n", mpi[i], mpi[j], dis1, dis2,
	      fabs(dis1 - dis2)/dis2);
      sum += fabs(dis1 - dis2)/dis2;
    }
  }
  fprintf(f, "%f\n", sum/(mp*(mp-1)/2));
*/
  /* model coordinates */
  /*
  fprintf(f, "======\n");
  for (i=0; i<mp; i++) {
    fprintf(f, "%u ", mpi[i]);
    for (j=0; j<var_per_loc; j++) {
      fprintf(f, "%f ", model[var_per_loc*i + j]);
      if (dump_computed) {
	     mxy[var_per_loc*mpi[i] + j] = model[var_per_loc*i + j];
      }
    }
    fprintf(f, "\n");
  }
  */
  fclose(f);

}

void fitTargetData() {
  float *xy, error, dis1, dis2, sum;
  int i,j,k;
  struct in_addr a;
  FILE *f;
  char buf[100];

  xy = (float *)malloc(sizeof(float)*var_per_loc);

  sprintf(buf, "target.%s.%u", set, mid);
  f = fopen(buf, "w");

  saveConfig(f);

  /* model probes id list */
  fprintf(f, "%u ", mp);
  for (i=0; i<mp; i++) {
    fprintf(f, "%u ", mpi[i]);
  }
  fprintf(f, "\n");

  for (i=0; i<nt; i++) {
    mps = mp;
    for (k=0; k<mp; k++) {
      mpsi[k] = k;
    }
    error = solve(dt2p[i], xy, var_per_loc,
		  (float (*)(float *, int, void*))(targetfit), try2);
    if (subset) {
      mps = num_subset; /* one bigger than dimensionality */
      choose_mpsi(xy);
      error = solve(dt2p[i], xy, var_per_loc,
		    (float (*)(float *, int, void*))(targetfit), try2);
    }

    for (j=0; j<var_per_loc; j++) {
      txy[var_per_loc*i+j] = xy[j];
    }

    a.s_addr = tip[i];
    fprintf(f, "%s\n", inet_ntoa(a));

    if (error == MAX_ERR) {
      fprintf(f, "Fatal ");
    } else {
      fprintf(f, "Normal ");
    }
    fprintf(f, "%f %f\n", error, error/mp);
    sum = 0;
    for (j=0; j<mp; j++) {
      /* percent error for each t2p distance */
      dis1 = (*dist_func)(&txy[var_per_loc*i], &model[var_per_loc*j]);
      dis2 = dt2p[i][mpi[j]];
      fprintf(f, "%u %f %f %f\n", mpi[j], dis1, dis2, fabs(dis1-dis2)/dis2);
      sum += fabs(dis1-dis2)/dis2;
    }
    fprintf(f, "%f\n", sum/mp);
  }

  fprintf(f, "======\n");
  for (i=0; i<nt; i++) {
    a.s_addr = tip[i];
    fprintf(f, "%s ", inet_ntoa(a));
    for (j=0; j<var_per_loc; j++) {
      fprintf(f, "%f ", txy[var_per_loc*i + j]);
    }
    fprintf(f, "\n");
  }

  fclose(f);
  free(xy);
}

/* use data in model and matrix to fit test data sets */
void fitTestData() {
  float *dp2p;
  int i, j, k, m, count;
  float *xy, dist, error, dis1, dis2, sum;
  FILE *f, *f2;
  struct in_addr a;
  char buf[100];

  if (!dump_fine_data) {
    sprintf(buf, "gnp.%s.%u", set, mid);
    f2 = fopen(buf, "w");
    count = 0;
  }

  sprintf(buf, "test.%s.%u", set, mid);
  f = fopen(buf, "w");
  
  /* settings */
  saveConfig(f);
  
  /* model probes id list */
  fprintf(f, "%u ", mp);
  for (k=0; k<mp; k++) {
    fprintf(f, "%u ", mpi[k]);
  }
  fprintf(f, "\n");

  dp2p = (float *)malloc(sizeof(float)*np);
  xy = (float *)malloc(sizeof(float)*var_per_loc);

  m = 0;
  for (i=0; i<np; i++) {
    /* does this probe belong to the test set? */
    if (m < mp && mpi[m] == i) {
      m++;
      continue;
    }
    /* okay, i is in the test set */

    /* extract distances from model probes from matrix */
    for(k=0; k<np; k++) {
      for (j=0; j<k; j++) {
	if (k == i) {
	  dp2p[j] = matrix[k][j];
	}
	if (j == i) {
	  dp2p[k] = matrix[k][j];
	}
      }
    }

    mps = mp;
    for (k=0; k<mp; k++) {
      mpsi[k] = k;
    }
    error = solve(dp2p, xy, var_per_loc,
		  (float (*)(float *, int, void*))(targetfit), try2);

    if (subset) {
      /* determine which subset do we use, then recompute xy */
      mps = num_subset; /* one bigger than dimensionality */
      choose_mpsi(xy);
      error = solve(dp2p, xy, var_per_loc,
		    (float (*)(float *, int, void*))(targetfit), try2);
    }

    if (error == MAX_ERR) {
      fprintf(f, "Fatal ");
    } else {
      fprintf(f, "Normal ");
    }
    fprintf(f, "%f %f\n", error, error/mp);
    sum = 0;
    for (k=0; k<mp; k++) {
      /* percent error for each t2p distance */
      dis1 = (*dist_func)(xy, &model[var_per_loc*k]);
      dis2 = dp2p[mpi[k]];
      fprintf(f, "%u %f %f %f\n", mpi[k], dis1, dis2, fabs(dis1-dis2)/dis2);
      sum += fabs(dis1-dis2)/dis2;
    }
    fprintf(f, "%f\n", sum/mp);

    fprintf(f, "======\n");
    fprintf(f, "%u ", i);
    for (j=0; j<var_per_loc; j++) {
      fprintf(f, "%f ", xy[j]);
      if (dump_computed) {
	mxy[var_per_loc*i + j] = xy[j];
      }
    }
    fprintf(f, "\n");

    if (dump_fine_data) {
      sprintf(buf, "gnp.%s.%u.%u", set, mid, i);
      f2 = fopen(buf, "w");
      for (k=0; k<nt; k++) {
	dist = (*dist_func)(&txy[var_per_loc*k], xy);
	a.s_addr = tip[k];
	fprintf(f2, "%s %.15f\n", inet_ntoa(a), dist);
      }
      fclose(f2);
    } else {
      for (k=0; k<nt; k++) {
	dist = (*dist_func)(&txy[var_per_loc*k], xy);
	fprintf(f2, "%u %.15f\n", count, dist);
	count++;
      }
    }
  }

  fclose(f);
  free(dp2p);
  free(xy);
}


/* use data in model and matrix to fit all remaining non-Landmark data */
void fitRemainData() {
  float *dp2p;
  int i, j, k, m, count;
  int *id;
  float *xy, dist, error, dis1, dis2, sum;
  float **rdata; /* to store all coordinates */
  FILE *f, *f2;
  char buf[100];

  /* array of coordinates vector */
  rdata = (float **)malloc(sizeof(float*)*(np-mp));
  for (i=0;i<(np-mp);i++) {
    rdata[i] = (float *)malloc(sizeof(float)*var_per_loc);
  }
  /* array of node ids corresponding to the test data */
  id = (int *)malloc(sizeof(int)*(np-mp));

  /* save our settings */
  sprintf(buf, "remain.%s.%u", set, mid);
  f = fopen(buf, "w");

  /* settings */
  saveConfig(f);

  /* model probes id list */
  fprintf(f, "%u ", mp);
  for (k=0; k<mp; k++) {
    fprintf(f, "%u ", mpi[k]);
  }
  fprintf(f, "\n");

  dp2p = (float *)malloc(sizeof(float)*np);
  xy = (float *)malloc(sizeof(float)*var_per_loc);

  m = 0;
  count = 0;
  for (i=0; i<np; i++) {
    /* does this probe belong to the test set? */
    if (m < mp && mpi[m] == i) {
      m++;
      continue;
    }
    /* okay, i is in the test set */

    /* extract distances from model probes from matrix */
    for(k=0; k<np; k++) {
      for (j=0; j<k; j++) {
	if (k == i) {
	  dp2p[j] = matrix[k][j];
	}
	if (j == i) {
	  dp2p[k] = matrix[k][j];
	}
      }
    }

    mps = mp;
    for (k=0; k<mp; k++) {
      mpsi[k] = k;
    }
    error = solve(dp2p, xy, var_per_loc,
		  (float (*)(float *, int, void*))(targetfit), try2);

    if (subset) {
      /* determine which subset do we use, then recompute xy */
      mps = num_subset; /* one bigger than dimensionality */
      choose_mpsi(xy);
      error = solve(dp2p, xy, var_per_loc,
		    (float (*)(float *, int, void*))(targetfit), try2);
    }


    if (error == MAX_ERR) {
      fprintf(f, "Fatal ");
    } else {
      fprintf(f, "Normal ");
    }
    fprintf(f, "%f %f\n", error, error/mp);
    sum = 0;
    for (k=0; k<mp; k++) {
      /* percent error for each t2p distance */
      dis1 = (*dist_func)(xy, &model[var_per_loc*k]);
      dis2 = dp2p[mpi[k]];
      fprintf(f, "%u %f %f %f\n", mpi[k], dis1, dis2, fabs(dis1-dis2)/dis2);
      sum += fabs(dis1-dis2)/dis2;
    }
    fprintf(f, "%f\n", sum/mp);
    
    fprintf(f, "======\n");
    fprintf(f, "%u ", i);
    for (j=0; j<var_per_loc; j++) {
      fprintf(f, "%f ", xy[j]);
    }
    fprintf(f, "\n");

    /* store away the coordinates */
    memcpy(rdata[count], xy, sizeof(float)*var_per_loc);
    id[count] = i;
    count++;
  }
  
  free(dp2p);
  free(xy);
  fclose(f);

  /* now we have all the coordinates of the test data
     we will generate the files */

  if (!dump_fine_data) {
    sprintf(buf, "gnpf.%s.%u", set, mid);
    f = fopen(buf, "w");
    sprintf(buf, "measuredf.%s.%u", set, mid);
    f2 = fopen(buf, "w");
    count = 0;
    for (i=0; i<(np-mp); i++) {
      for (j=0; j<i; j++) {
	dist = (*dist_func)(rdata[i], rdata[j]);
	dist += 0.000000000000001; /* to avoid division by zero later */
	fprintf(f, "%u %.15f\n", count, dist);
	fprintf(f2, "%u %f\n", count, matrix[id[i]][id[j]]);
	count++;
      }
    }
    fclose(f);
    fclose(f2);
  } else {
    /* this loop will double count each path, but that's okay for now */
    for (i=0; i<(np-mp); i++) {
      sprintf(buf, "gnpfi.%s.%u.%u", set, mid, id[i]);
      f = fopen(buf, "w");
      sprintf(buf, "measuredfi.%s.%u.%u", set, mid, id[i]);
      f2 = fopen(buf, "w");
      for (j=0; j<(np-mp); j++) {
	if (i != j) {
	  dist = (*dist_func)(rdata[i], rdata[j]);
	  dist += 0.000000000000001; /* to avoid division by zero later */
	  fprintf(f, "%u %.15f\n", id[j], dist);
	  if (id[i]>id[j]) {
	    fprintf(f2, "%u %f\n", id[j], matrix[id[i]][id[j]]);
	  } else if (id[i]<id[j]) {
	    fprintf(f2, "%u %f\n", id[j], matrix[id[j]][id[i]]);
	  } else {
	    printf("Something is wrong!!!\n");
	  }
	}
      }
      fclose(f);
      fclose(f2);
    }
  }

  for (i=0;i<(np-mp);i++) {
    free(rdata[i]);
  }
  free(rdata);
  free(id);
}



/* compute cylindrical distance between a and b */
/* a[0] is on the circumference, a[1] is on the height */
/* remember, param1 is circumference, param2 is height */
float cylindrical_dist(float *a, float *b) {
  float p1[2]; /* transformed points */
  float p2[2];
  float temp;

  p1[0] = fmod(a[0], param1);
  if (p1[0] < 0) {
    p1[0] = param1 + p1[0];
  }

  p2[0] = fmod(b[0], param1);
  if (p2[0] < 0) {
    p2[0] = param1 + p2[0];
  }

  p1[1] = fmod(a[1], param2);
  if (p1[1] < 0) {
    p1[1] = param2 + p1[1];
  }

  p2[1] = fmod(b[1], param2);
  if (p2[1] < 0) {
    p2[1] = param2 + p2[1];
  }

  /* distance along the circumference wraps so at most 0.5*param1 */
  temp = fabs(p1[0] - p2[0]);
  if (temp > 0.5*param1) {
    /* artificially make the transformation */
    p1[0] = 0;
    p2[0] = param1 - temp;
  }

  /* all set, just treat the rest like a linear dist */
  return linear_dist(p1, p2);

}


/* a and b are longitude and latitude in radian */
/* a[0] is longitude (-pi, pi) , a[1] is latitude (-.5pi, .5pi)*/
/* param1 is the radius */
float spherical_dist(float *a, float *b) {
  float p1[2], p2[2];
  float dlon, dlat, i, c, temp;

  /* bring the coordinates back into the finite space */
  /* need to make sure the transformation is continuous */

  /* longitude */
  p1[0] = fmod(a[0], 2*PI);
  if (fabs(p1[0]) > PI) {
    if (p1[0] > 0) {
      p1[0] = - (2*PI - p1[0]);
    } else {
      p1[0] = (2*PI + p1[0]);
    }
  }

  p2[0] = fmod(b[0], 2*PI);
  if (fabs(p2[0]) > PI) {
    if (p2[0] > 0) {
      p2[0] = - (2*PI - p2[0]);
    } else {
      p2[0] = (2*PI + p2[0]);
    }
  }

  /* latitude */
  p1[1] = fmod(a[1], 2*PI);
  if (p1[1] < 0) {
    p1[1] = (2*PI + p1[1]);
  }
  temp = p1[1]/(0.5*PI);
  if (temp > 3) {
    p1[1] = - (2*PI - p1[1]);
  } else if (temp > 2) {
    p1[1] = - (p1[1] - PI);
  } else if (temp > 1) {
    p1[1] = PI - p1[1];
  }

  p2[1] = fmod(b[1], 2*PI);
  if (p2[1] < 0) {
    p2[1] = (2*PI + p2[1]);
  }
  temp = p2[1]/(0.5*PI);
  if (temp > 3) {
    p2[1] = - (2*PI - p2[1]);
  } else if (temp > 2) {
    p2[1] = - (p2[1] - PI);
  } else if (temp > 1) {
    p2[1] = PI - p2[1];
  }
  
  /*
    Algorithm from http://www.census.gov/cgi-bin/geo/gisfaq?Q5.1

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = (sin(dlat/2))^2 + cos(lat1) * cos(lat2) * (sin(dlon/2))^2 
    c = 2 * atan2( sqrt(a), sqrt(1-a) ) 
    d = R * c 
  */

  dlon = p2[0] - p1[0];
  dlat = p2[1] - p1[1];

  i = pow(sin(dlat/2), 2) + cos(p1[1]) * cos(p2[1]) * pow(sin(dlon/2), 2);
  c = 2 * atan2(sqrt(i), sqrt(1-i));

  return  param1*c;
}


/* a and b are arrays of coordinates */
/*
float linear_dist(float *a, float *b) {
  int i;
  float dist;

  dist = 0;

  for (i=0; i<var_per_loc; i++) {
    switch (norm) {
    case MANHATTAN:
      dist += fabs(a[i] - b[i]);
      break;
    case EUCLIDEAN:
      dist += pow(a[i] - b[i], 2);
      break;
    default:
      printf("Unknown norm measure\n");
      exit(1);
    }
  }

  switch (norm) {
  case MANHATTAN:
    return dist;
  case EUCLIDEAN:
    return sqrt(dist);
  default:
    printf("Unknown norm measure\n");
    exit(1);
  }
}
*/

float linear_dist(float *a, float *b) {
  int i;
  float dist;

  dist = 0;

  for (i=0; i<var_per_loc; i++) {
    if (norm == 1) {
      dist += fabs(a[i] - b[i]);
    } else {
      dist += pow(fabs(a[i] - b[i]), norm);
    }
  }

  if (norm == 1) {
    return dist;
  } else {
    return pow(dist, 1.0/norm);
  }
}


/* a and b are arrays of coordinates */
/* param1 is d1 dimensions param2 is d2 dimensions
   param3 is the norm of d1, norm is the norm of d2 */
float d1_plus_d2_dist(float *a, float *b) {
  int i;
  float dist, dist2;

  dist = 0;
  dist2 = 0;

  /* for d1 */
  for (i=0; i<param1; i++) {
    if (param3 == 1) {
      dist += fabs(a[i] - b[i]);
    } else {
      dist += pow(fabs(a[i] - b[i]), param3);
    }
  }
  if (param3 != 1) {
    dist = pow(dist, 1.0/param3);
  }

  for (i=param1; i<param1+param2; i++) {
    if (norm == 1) {
      dist2 += fabs(a[i] - b[i]);
    } else {
      dist2 += pow(fabs(a[i] - b[i]), norm);
    }
  }
  if (norm != 1) {
    dist2 = pow(dist2, 1.0/norm);
  }

  return dist + dist2;
}

unsigned int myrand() {
  static FILE *f;
  static int here = 0;
  unsigned int ret;

  if (really_random) {
    if (!here) {
      f = fopen("/dev/urandom", "r");
      if (!f) {
	printf("Cannot open /dev/urandom device, change the config file.\n");
	exit(1);
      }
      here = 1;
    }
    
    ((char *)&ret)[0] = fgetc(f);
    ((char *)&ret)[1] = fgetc(f);
    ((char *)&ret)[2] = fgetc(f);
    ((char *)&ret)[3] = fgetc(f);
  } else {
    ret = rand();
  }

  return ret;
}


float solve_myself(void *d, float *solution, int num, 
      float (*fit_func)(float *, int, void*), int mytry) {
  float min, fit, localftol;
  float *myxy;
  float **p;
  float *y;
  int i, j, restarted;

  localftol = ftol;
  myxy = (float *)malloc(sizeof(float)*(num+1));

  if (gd == 1) { /* downhill simplex */
    p = (float **)malloc(sizeof(float *)*(num+1+1));
    p[1] = myxy;
    for (i=2;i<=num+1;i++) {
      p[i]=(float *)malloc(sizeof(float)*(num+1));
    }
    y = (float *)malloc(sizeof(float)*(num+1+1));
  } else if (gd == 2) { /* powell */
    p = (float **)malloc(sizeof(float*)*(num+1));
    for (i=1;i<=num;i++) {
      p[i] = (float *)malloc(sizeof(float)*(num+1));
    }
  }

again:
  min = MAX_ERR;

  for (i=0; i<mytry; i++) {
    restarted = 0;

    for (j=1; j<=num; j++) {
      myxy[j] = (float) (myrand()%range) - (range/2);
    }
      
    if (gd == 0) {
      fit = gradient_decent(&myxy[1], num, d, fit_func);
    } else if (gd == 1) {
      /* use simplex downhill */
    restart:
      y[1] = fit_func(&myxy[1],num,d);
      for (j=2; j<=num+1; j++) {
        memcpy(p[j],myxy,sizeof(float)*(num+1));
        p[j][j-1] += lambda;
        y[j] = fit_func(&p[j][1],num,d);
      }
      simplex_downhill(p,y,num,localftol,fit_func,&j,d);
      if (j < 0) {
        printf("No answer\n");
        continue;
      }

      if (restarted < restarts) {
        restarted++;
        goto restart;
      }

      fit = fit_p1_error(&myxy[1],num,d);
      printf("Myself Fit = %.15f\n", fit);
    } else if (gd == 2) {
      /* use powell */
      for (j=1; j<=num; j++) {
  /* initialize the direction set to unit vectors */
        bzero(p[j], sizeof(float)*(num+1));
        p[j][j] = 1;
      }
      printf("Powell's method is not available in this code release\n");
      exit(1);
      /*
      powell(myxy, p, num, localftol, &j, &fit, fit_func, d);
      */
      if (j < 0) {
        printf("No answer\n");
        continue;
      }

      printf("Fit = %.15f\n", fit);
    }

    if (fit < min) {
      min = fit;
      for (j=0; j<num; j++) {
        solution[j] = myxy[j+1];
      }
    }
  }

  printf("\n\n");

  if (min == MAX_ERR) {
    /* we did not converge, relax ftol and try again */
    localftol = localftol*10;
    goto again;
  }

  if (gd == 1) {
    free(y);
    for (i=2;i<=num+1;i++) {
      free(p[i]);
    }
    free(p);
  } else if (gd == 2) {
    for (i=1; i<=num; i++) {
      free(p[i]);
    }
    free(p);
  }
  free(myxy);

  return min;
}








float solve(void *d, float *solution, int num, 
	    float (*fit_func)(float *, int, void*), int mytry) {
  float min, fit, localftol;
  float *myxy;
  float **p;
  float *y;
  int i, j, restarted;

  localftol = ftol;
  myxy = (float *)malloc(sizeof(float)*(num+1));

  if (gd == 1) { /* downhill simplex */
    p = (float **)malloc(sizeof(float *)*(num+1+1));
    p[1] = myxy;
    for (i=2;i<=num+1;i++) {
      p[i]=(float *)malloc(sizeof(float)*(num+1));
    }
    y = (float *)malloc(sizeof(float)*(num+1+1));
  } else if (gd == 2) { /* powell */
    p = (float **)malloc(sizeof(float*)*(num+1));
    for (i=1;i<=num;i++) {
      p[i] = (float *)malloc(sizeof(float)*(num+1));
    }
  }

again:
  min = MAX_ERR;

  for (i=0; i<mytry; i++) {
    restarted = 0;

    for (j=1; j<=num; j++) {
      myxy[j] = (float) (myrand()%range) - (range/2);
    }
      
    if (gd == 0) {
      fit = gradient_decent(&myxy[1], num, d, fit_func);
    } else if (gd == 1) {
      /* use simplex downhill */
    restart:
      y[1] = fit_func(&myxy[1],num,d);
      for (j=2; j<=num+1; j++) {
	       memcpy(p[j],myxy,sizeof(float)*(num+1));
	       p[j][j-1] += lambda;
	       y[j] = fit_func(&p[j][1],num,d);
      }
      simplex_downhill(p,y,num,localftol,fit_func,&j,d);
      if (j < 0) {
	       printf("No answer\n");
	       continue;
      }

      if (restarted < restarts) {
	       restarted++;
	       goto restart;
      }

      fit = fit_func(&myxy[1],num,d);
      printf("Origin Fit = %.15f\n", fit);
    } else if (gd == 2) {
      /* use powell */
      for (j=1; j<=num; j++) {
	/* initialize the direction set to unit vectors */
	bzero(p[j], sizeof(float)*(num+1));
	p[j][j] = 1;
      }
      printf("Powell's method is not available in this code release\n");
      exit(1);
      /*
      powell(myxy, p, num, localftol, &j, &fit, fit_func, d);
      */
      if (j < 0) {
	printf("No answer\n");
	continue;
      }

      printf("Fit = %.15f\n", fit);
    }

    if (fit < min) {
      min = fit;
      for (j=0; j<num; j++) {
	solution[j] = myxy[j+1];
      }
    }
  }

  printf("\n\n");

  if (min == MAX_ERR) {
    /* we did not converge, relax ftol and try again */
    localftol = localftol*10;
    goto again;
  }

  if (gd == 1) {
    free(y);
    for (i=2;i<=num+1;i++) {
      free(p[i]);
    }
    free(p);
  } else if (gd == 2) {
    for (i=1; i<=num; i++) {
      free(p[i]);
    }
    free(p);
  }
  free(myxy);

  return min;
}

void read_basic_data() {
  FILE *f;
  char *buf;
  int i, j;

  buf = (char*)malloc(30*np);

  matrix = (float **)malloc(sizeof(void *)*np);
  myMatrix = (float **)malloc(sizeof(void *)*np);
  for (i=0; i<np; i++) {
    matrix[i] = (float *)malloc(sizeof(float)*np);
    myMatrix[i] = (float *)malloc(sizeof(float)*np);
  }
  
  f = fopen("matrix", "r");
  
  for (i=0; i<np; i++) {
    for (j=0; j<i; j++) {
      fscanf(f, "%f", &matrix[i][j]);
      if (matrix[i][j] > criMax)
        criMax = matrix[i][j];
   
    }
    /* get to the end of line */
    fscanf(f, "%[^\n]", buf);
    fgetc(f); /* new line */
  }
  criMax *= 1.5;
  fclose(f);
  free(buf);

}

void convertMatrix(){
        printf("converMatrix \n");
        initCalc(criK,criMax,7,criArray);
        printf("After Init \n");
        int i,j;        
        for(i = 0;i < np;i ++)
          for(j = 0;j < i;j ++){
                //printf("%d %d \n",i,j);
                myMatrix[i][j] = calcRe(criK,criMax,7,criArray,matrix[i][j]);
          }
}


/* this is to fit mp points for the model */
float fit_p1(float *array, int num, void *stuff) {
  float dist, sum, error;
  int i, j;
  
  sum = 0;
  
  for (i=0; i<mp; i++) {
    for (j=0; j<i; j++) {
      dist = dist_func(&array[var_per_loc*i], &array[var_per_loc*j]);
      if (normalized == 1) {
	     /* normalized error */
	       error = (matrix[mpi[i]][mpi[j]] - dist)/matrix[mpi[i]][mpi[j]];
      } else if (normalized == 2) {
	/* log transform error */
	       error = log(fabs(matrix[mpi[i]][mpi[j]] - dist)+1.0);
      } else if (normalized == 3) {
	/* transform distances by log */
	       error = log(matrix[mpi[i]][mpi[j]]/dist);
      } else if (normalized == 4) {
	/* relative error */
	       error = max(matrix[mpi[i]][mpi[j]],dist)/
	         min(matrix[mpi[i]][mpi[j]],dist);
      } else if (normalized == 5) {
	/* heavily skewed normalized error */
	     error = (matrix[mpi[i]][mpi[j]] - dist)/
	       pow(matrix[mpi[i]][mpi[j]],2);
      } else {
	/* absolute error */
	       error = matrix[mpi[i]][mpi[j]] - dist;
      }

      if (no_square) {
	       error = fabs(error);
      } else {
	       error = pow(error, 2);
      }

      sum += error;
    }
  }
  return sum;
}

float fit_p1_error(float *array, int num, void *stuff) {
  float dist, sum, error;
  int i, j;
  
  sum = 0;
  
  for (i=0; i<mp; i++) {
    for (j=0; j<i; j++) {
      dist = calcDist(criK,dist_func(&array[var_per_loc*i], &array[var_per_loc*j]));
      if (normalized == 1) {
       /* normalized error */
         error = (matrix[mpi[i]][mpi[j]] - dist)/matrix[mpi[i]][mpi[j]];
      } else if (normalized == 2) {
  /* log transform error */
         error = log(fabs(matrix[mpi[i]][mpi[j]] - dist)+1.0);
      } else if (normalized == 3) {
  /* transform distances by log */
         error = log(matrix[mpi[i]][mpi[j]]/dist);
      } else if (normalized == 4) {
  /* relative error */
         error = max(matrix[mpi[i]][mpi[j]],dist)/
           min(matrix[mpi[i]][mpi[j]],dist);
      } else if (normalized == 5) {
  /* heavily skewed normalized error */
       error = (matrix[mpi[i]][mpi[j]] - dist)/
         pow(matrix[mpi[i]][mpi[j]],2);
      } else {
  /* absolute error */
         error = matrix[mpi[i]][mpi[j]] - dist;
      }

      if (no_square) {
         error = fabs(error);
      } else {
         error = pow(error, 2);
      }

      sum += error;
    }
  }
  return sum;
}




float fit_p1_myself(float *array, int num, void *stuff) {
  float dist, sum, error;
  int i, j;
  
  sum = 0;
  
  for (i=0; i<mp; i++) {
    for (j=0; j<i; j++) {
      dist = dist_func(&array[var_per_loc*i], &array[var_per_loc*j]);
      if (normalized == 1) {
       /* normalized error */
         error = (myMatrix[mpi[i]][mpi[j]] - dist)/myMatrix[mpi[i]][mpi[j]];
      } else if (normalized == 2) {
  /* log transform error */
         error = log(fabs(myMatrix[mpi[i]][mpi[j]] - dist)+1.0);
      } else if (normalized == 3) {
  /* transform distances by log */
         error = log(myMatrix[mpi[i]][mpi[j]]/dist);
      } else if (normalized == 4) {
  /* relative error */
         error = max(myMatrix[mpi[i]][mpi[j]],dist)/
           min(myMatrix[mpi[i]][mpi[j]],dist);
      } else if (normalized == 5) {
  /* heavily skewed normalized error */
       error = (myMatrix[mpi[i]][mpi[j]] - dist)/
         pow(myMatrix[mpi[i]][mpi[j]],2);
      } else {
  /* absolute error */
         error = myMatrix[mpi[i]][mpi[j]] - dist;
      }

      if (no_square) {
         error = fabs(error);
      } else {
         error = pow(error, 2);
      }

      sum += error;
    }
  }
  return sum;
}

/* to fit a target in the model of mp model probes
   stuff: array of distaces between target and probes
 */
float fit_p2(float *array, int num, void *stuff) {
  float dist, sum, error;
  int i;

  sum = 0;

  for (i=0; i<mp; i++) {
    dist = dist_func(&array[0], &model[var_per_loc*i]);
    if (normalized == 1) {
      /* normalized error */
      error = (((float *)stuff)[mpi[i]] - dist)/((float *)stuff)[mpi[i]];
    } else if (normalized == 2) {
      /* transform error by log */
      error = log(fabs(((float *)stuff)[mpi[i]] - dist)+1.0);
    } else if (normalized == 3) {
      /* transform distances by log */
      error = log(((float *)stuff)[mpi[i]]/dist);
    } else if (normalized == 4) {
      /* relative error */
      error = max(((float *)stuff)[mpi[i]],dist)/
	min(((float *)stuff)[mpi[i]],dist);
    } else if (normalized == 5) {
      /* heavily skewed normalized error */
      error = (((float *)stuff)[mpi[i]] - dist)/
	pow(((float *)stuff)[mpi[i]],2);
    } else {
      /* absolute error */
      error = ((float *)stuff)[mpi[i]] - dist;
    }

    if (no_square) {
      error = fabs(error);
    } else {
      error = pow(error, 2);
    }

    sum += error;
  }

  return sum;
}


/* to fit a target in the model of mp model probes
   stuff: array of distaces between target and probes
 */
float fit_p2_subset(float *array, int num, void *stuff) {
  float dist, sum, error;
  int i;

  sum = 0;

  for (i=0; i<mps; i++) {
    dist = dist_func(&array[0], &model[var_per_loc*mpsi[i]]);
    if (normalized == 1) {
      /* normalized error */
      error = (((float *)stuff)[mpi[mpsi[i]]] - dist)/((float *)stuff)[mpi[mpsi[i]]];
    } else if (normalized == 2) {
      /* transform error by log */
      error = log(fabs(((float *)stuff)[mpi[mpsi[i]]] - dist)+1.0);
    } else if (normalized == 3) {
      /* transform distances by log */
      error = log(((float *)stuff)[mpi[mpsi[i]]]/dist);
    } else if (normalized == 4) {
      /* relative error */
      error = max(((float *)stuff)[mpi[mpsi[i]]],dist)/
	min(((float *)stuff)[mpi[mpsi[i]]],dist);
    } else if (normalized == 5) {
      /* heavily skewed normalized error */
      error = (((float *)stuff)[mpi[mpsi[i]]] - dist)/
	pow(((float *)stuff)[mpi[mpsi[i]]],2);
    } else {
      /* absolute error */
      error = ((float *)stuff)[mpi[mpsi[i]]] - dist;
    }

    if (no_square) {
      error = fabs(error);
    } else {
      error = pow(error, 2);
    }

    sum += error;
  }

  return sum;
}

void choose_mpsi(float *xy) {
  float *dist, temp;
  int t, i, j;

  switch (subset) {
  case 1:
    /* pick the mps nearest model probes to xy and fill mpsi
       with those IDs
    */
    
    dist = (float *)malloc(sizeof(float)*mp);
    
    for (i=0; i<mp; i++) {
      dist[i] = dist_func(xy, &model[var_per_loc*i]);
      mpsi[i] = i;
    }
    
    /* now do a sort of dist */
    for (i=0; i<mps; i++) {
      for (j=i+1; j<mp; j++) {
	if (dist[i] > dist[j]) {
	  /* swap */
	  temp = dist[i];
	  dist[i] = dist[j];
	  dist[j] = temp;
	  t = mpsi[i];
	  mpsi[i] = mpsi[j];
	  mpsi[j] = t;
	}
      }
    }
    
    free(dist);
    break;
  case 2:
    /* fill up mpsi randomly */
    for (i=0; i<mp; i++) {
      mpsi[i] = -1;
    }
    for (i=0; i<mp; i++) {
      do {
	t = myrand()%mp;
      }	while (mpsi[t] != -1);
      mpsi[t] = i;
    }
    break;
  default:
    printf("Undefined subset selection option\n");
    exit(1);
  }
}


inline float max(float a, float b) {
  if (a > b) {
    return a;
  }
  return b;
}

inline float min(float a, float b) {
  if (a > b) {
    return b;
  }
  return a;
}


/* array: the list of numbers to nudge
   num: number of elements in the array
   stuff: opaque stuff to be passed to the fit_func
   fit_func: the function to apply for computing fitness
*/
float gradient_decent(float *array, int num, void *stuff, 
		      float (*fit_func)(float *, int, void*)) {
  float fit, ofit, nfit, ox, dfdx;
  int i;

  fit = (*fit_func)(array, num, stuff);
  ofit = fit + 1;

  while (ofit > fit) {
    ofit = fit;
    for (i=0; i<num; i++) {
      ox = array[i];
      array[i] = array[i] + 1;
      /*      array[i] = array[i] + 0.1;*/
      nfit = (*fit_func)(array, num, stuff);
      dfdx = nfit - fit;
      array[i] = ox - 0.05*dfdx;
      /*      array[i] = ox - dfdx;*/
      /*      if (dfdx > 0) {
	array[i] = ox - 0.1;
      } else {
	array[i] = ox + 0.1;
      }*/
      fit = (*fit_func)(array, num, stuff);
    }
  }
  return fit;
}
