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

void simplex_downhill(float **simplex, float *values, int d, float ftol,
	 	      float (*obj)(float *, int, void *), int *num_eval,
		      void *stuff);
