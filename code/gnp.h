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

#define PI 3.141592654

typedef enum embeddings {
  ND_SPACE,
  CYLINDER,
  SPHERE,
  D1_PLUS_D2_SPACE,
  LAND_MASS,
  LAST_EM
} embeddings_t;

/*
typedef enum norm_measures {
  MANHATTAN,
  EUCLIDEAN,
  LAST_NM
} norm_measures_t;
*/

typedef enum model_generators {
  SFILE,
  LAST_MG
} mg_t;

void saveConfig(FILE* f);
void fitTargetData();
void fitTestData();
void fitRemainData();
unsigned int myrand();
float solve(void *d, float *solution, int num, 
	    float (*fit_func)(float *, int, void*), int mytry);

float solve_myself(void *d, float *solution, int num, 
      float (*fit_func)(float *, int, void*), int mytry);



void fitModel();
float linear_dist(float *a, float *b);
float d1_plus_d2_dist(float *a, float *b);
float cylindrical_dist(float *a, float *b);
float spherical_dist(float *a, float *b);
void read_basic_data();

float fit_p1(float *array, int num, void *stuff);

float fit_p1_myself(float *array, int num, void *stuff);

float fit_p1_error(float *array, int num, void *stuff);





float fit_p2(float *array, int num, void *stuff);
float fit_p2_subset(float *array, int num, void *stuff);
void choose_mpsi(float *xy);
float gradient_decent(float *array, int num, void *stuff, 
		      float (*fit_func)(float *, int, void*));
inline float max(float a, float b);
inline float min(float a, float b);
