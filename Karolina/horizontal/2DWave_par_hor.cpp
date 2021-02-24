#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <mpi.h>

using namespace std;

int main(int argc, char* argv[]){

constexpr int M = 2305;  // M length intervals.
double T = 1;            // final time.
constexpr int J = 770;   // rows per process
int nproc = 3;           // number of processes
double dt = 0.2/M;
int N = T/dt;
int t1=0.333/dt, t2=0.666/dt, t3=N; // points at which we want to print our results.
double dy = 2./M;
double dy2 = dy*dy;
double dtdy2 = dt*dt/dy2;
int rank, size, x_shift, s_rank;
double t_start, t_end, run_time;
MPI_Comm comm;
comm  = MPI_COMM_WORLD;

MPI_Init(NULL,NULL);
MPI_Comm_rank(comm, &rank);            
MPI_Comm_size(comm, &size);

// Error message to check whether we have a correct choice of M, J and number of processes
if(rank==0) {
	if(size != nproc) { 
		cout<<"Abort reason: Chosen "<< size <<" processes instead of " << nproc << endl;
		MPI_Abort(comm, 911);
		MPI_Finalize();
	return 0;
	}
}

MPI_Barrier(comm);
t_start = MPI_Wtime();
// -------------------- HORIZONTAL STRIPS --------------------------------
// -----------------------------------------------------------------------
// -------------------- array init ---------------------------------------
// -----------------------------------------------------------------------

// global empty 3D variable for storing the final sol
double*** U = new double**[3]; // m=3 n=M+1 o=M+1
double**  U2 = new double*[3*(M+1)];
double*   U3 = new double[3*(M+1)*(M+1)];
for (int i=0; i<3; i++) {
    for (int j=0; j<M+1; j++) {
        U2[(M+1)*i+j] = U3 + ((M+1)*i+j)*(M+1);
    }
    U[i] = U2 + (M+1)*i;
}

double*** Ur = new double**[3]; // m=3 n=J o=M+1
double**  Ur2 = new double*[3*J];
double*   Ur3 = new double[3*J*(M+1)];
for (int i=0; i<3; i++) {
    for (int j=0; j<J; j++) {
        Ur2[J*i+j] = Ur3 + (J*i+j)*(M+1);
    }
    Ur[i] = Ur2 + J*i;
}

double** Utemp = new double*[J-2];
double*  Utemp1 = new double[(J-2)*(M+1)];
for (int i=0; i<J-2; i++) {
    Utemp[i] = Utemp1 + (M+1)*i;
}

// ---- x (=i) from 0 to J-1 ------
// ---- y (=j) from 0 to M   ------

// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------

// initial conditions - at each rank, the whole array
for (int i=0; i<J; ++i){       // x-dim
	for (int j=0; j<M+1; ++j){     // y-dim
		x_shift = rank*(J-2)+i;
		Ur[0][i][j] = Ur[1][i][j] = exp( -40 * ( (x_shift*dy-1-0.4)*(x_shift*dy-1-0.4) + (j*dy-1)*(j*dy-1) ) );
	}
}

// boundary conditions in the y-dimension (LHS and RHS of the domain (here first and last row), all times)
// fix y=0 and y=M, loop over x
for (int i=0; i<J; ++i){ 	
	Ur[0][i][0] = Ur[0][i][M] = Ur[1][i][0] = Ur[1][i][M] = Ur[2][i][0] = Ur[2][i][M] = 0; 	
}

// boundary conditions in the x-dimension - only in 1st and last process
// fix x=0 (rank 0) and x=J-1 (rank nproc-1), loop over cols (y)
if (rank == 0) {
	for (int j=0; j<M+1; ++j){ 	
		Ur[0][0][j] = Ur[1][0][j] = Ur[0][0][j] = 0;
	}	
}

if (rank == nproc-1) {
	for (int j=0; j<M+1; ++j){ 	
		Ur[0][J-1][j] = Ur[1][J-1][j] = Ur[2][J-1][j] = 0;
	}
}
// ----------------------------------------------------------------------------------------
// --------------------------- SEND THE INITIAL CONDITIONS TO RANK 0 ----------------------
// ----------------------------------------------------------------------------------------

// // send the first j-2 rows from each process (to avoid overlapping)
// if (rank != 0) {
//   MPI_Ssend(&(Ur[0][0][0]),(M+1)*(J-2),MPI_DOUBLE,0,0,comm);
// }
// // send the last 2 rows from the last process
// if (rank == nproc-1) {
// 	  MPI_Ssend(&(Ur[0][J-2][0]),(M+1)*2,MPI_DOUBLE,0,1,comm);
// }

// if (rank == 0) {
// 	// the first J-2 rows of U taken from Ur in rank 0 
// 	for (int i=0; i<J-2; i++) {
//       for (int j=0; j<M+1; j++) {
//           U[0][i][j] = Ur[0][i][j];
//       }
//     }
//     // receive other segments from the other ranks and save them to a temporary array
//     for (int p=1; p<nproc; p++) {
//       MPI_Status status;
//       MPI_Recv(&(Utemp[0][0]),(M+1)*(J-2),MPI_DOUBLE,MPI_ANY_SOURCE,0,comm,&status);
//       s_rank = status.MPI_SOURCE; // get the rank of the sender
//       // assign the values from the temporary array into the correct place in the final array U
//       for (int i=0; i<J-2; i++) {
//         for (int j=0; j<M+1; j++) {
//             U[0][s_rank*(J-2)+i][j] = Utemp[i][j];

//         }
//       }
//     }
//     // receive the last two rows from the last process
//     MPI_Recv(&(U[0][M-1][0]),(M+1)*2,MPI_DOUBLE,nproc-1,1,comm,MPI_STATUS_IGNORE);
	
//  	// print initial U values to file, row by row.
// 	ofstream out {"U_t0_rank0.csv"};
// 	out<<fixed<<setprecision(4);
// 		for(int i=0; i<=M; ++i){
// 			// cout << "i=" << i << "j: ";
// 			for(int j=0; j<=M; ++j){					
// 				out<<U[0][i][j]<<" ";
// 			}
// 			out<<endl;
// 		}	
// 	out.close();

// 	cout << "Rank 0 finished writing U values at t0." << endl;
// }

// -----------------------------------------------------------------------------------------
// ------------------------------ NUMERICAL SCHEME -----------------------------------------
// -----------------------------------------------------------------------------------------

// use numerical scheme to obtain the future values of U.
for (int t=1; t<=N; ++t){
	
	for (int i=1; i<J-1; ++i){        // x from 1 to J-2
		for (int j=1; j<M; ++j){      // y from 1 to M-1		
			Ur[2][i][j] = 2*Ur[1][i][j] - Ur[0][i][j]	 
						 	 + dtdy2*( Ur[1][i+1][j] + Ur[1][i-1][j] + Ur[1][i][j+1] + Ur[1][i][j-1] - 4*Ur[1][i][j] ); 	
		}		
	}
	
	MPI_Barrier(comm);

	// ---------------- HALO SWAPPING -------------------------------------------
	if (rank > 0) {
		MPI_Ssend(&(Ur[2][1][0]), M+1, MPI_DOUBLE, rank-1, 0, comm);
		MPI_Recv(&(Ur[2][0][0]), M+1, MPI_DOUBLE, rank-1, 0, comm, MPI_STATUS_IGNORE);
	}

	if (rank < nproc-1) {
	    MPI_Recv(&(Ur[2][J-1][0]), M+1, MPI_DOUBLE, rank+1, 0, comm, MPI_STATUS_IGNORE);
		MPI_Ssend(&(Ur[2][J-2][0]), M+1, MPI_DOUBLE, rank+1, 0, comm);
	}

	

	MPI_Barrier(comm);

	
 //   // print out files for fixed times	
	// if(t==t1 || t==t2 || t==t3){	
	// 	// send the first j-2 rows from each process (to avoid overlapping)
	// 	if (rank != 0) {
	// 	  MPI_Ssend(&(Ur[2][0][0]),(M+1)*(J-2),MPI_DOUBLE,0,0,comm);
	// 	}
	// 	// send the last 2 rows from the last process
	// 	if (rank == nproc-1) {
	// 		  MPI_Ssend(&(Ur[2][J-2][0]),(M+1)*2,MPI_DOUBLE,0,1,comm);
	// 	}

	// 	if (rank == 0) {
	// 		// the first J-2 rows of U taken from Ur in rank 0 
	// 		for (int i=0; i<J-2; i++) {
	// 	      for (int j=0; j<M+1; j++) {
	// 	          U[2][i][j] = Ur[2][i][j];
	// 	      }
	// 	    }
	// 	    // receive other segments from the other ranks and save them to a temporary array
	// 	    for (int p=1; p<nproc; p++) {
	// 	      MPI_Status status;
	// 	      MPI_Recv(&(Utemp[0][0]),(M+1)*(J-2),MPI_DOUBLE,MPI_ANY_SOURCE,0,comm,&status);
	// 	      s_rank = status.MPI_SOURCE; // get the rank of the sender
	// 	      // assign the values from the temporary array into the correct place in the final array U
	// 	      for (int i=0; i<J-2; i++) {
	// 	        for (int j=0; j<M+1; j++) {
	// 	            U[2][s_rank*(J-2)+i][j] = Utemp[i][j];

	// 	        }
	// 	      }
	// 	    }
	// 	    // receive the last two rows from the last process
	// 	    MPI_Recv(&(U[2][M-1][0]),(M+1)*2,MPI_DOUBLE,nproc-1,1,comm,MPI_STATUS_IGNORE);

	// 		stringstream ss;
	// 		ss << fixed << setprecision(2) << t*dt; // this ensures that the double value gets converted
	// 		string time = ss.str();						 // to string with only 2 trailing digits.
			
	// 		ofstream out {"U_3p_t"+ss.str()+".csv"};
	// 		out<<fixed<<setprecision(4);			
	// 			for(int i=0; i<=M; ++i){
	// 				for(int j=0; j<=M; ++j){		
	// 					out<<U[2][i][j]<<" ";
	// 				}
	// 				out<<endl;
	// 			}
	// 		out.close();
	// 		cout << "Rank 0 finished writing U values at " << t << endl;
	//     }
	// }		
	

	// update the previous times.
	for (int i=0; i<J; ++i){
		for (int j=0; j<M+1; ++j){		
			Ur[0][i][j] = Ur[1][i][j];
			Ur[1][i][j] = Ur[2][i][j];
			}		
	}
	
	if (rank == 0) {
		cout<<"iteration "<<t<<" done"<<endl;	
	}

}

MPI_Barrier(comm);
t_end = MPI_Wtime();
run_time = t_end - t_start;

// print out file for final time	
// send the first j-2 rows from each process (to avoid overlapping)
if (rank != 0) {
  MPI_Ssend(&(Ur[2][0][0]),(M+1)*(J-2),MPI_DOUBLE,0,0,comm);
}
// send the last 2 rows from the last process
if (rank == nproc-1) {
	  MPI_Ssend(&(Ur[2][J-2][0]),(M+1)*2,MPI_DOUBLE,0,1,comm);
}

if (rank == 0) {
	// the first J-2 rows of U taken from Ur in rank 0 
	for (int i=0; i<J-2; i++) {
      for (int j=0; j<M+1; j++) {
          U[2][i][j] = Ur[2][i][j];
      }
    }
    // receive other segments from the other ranks and save them to a temporary array
    for (int p=1; p<nproc; p++) {
      MPI_Status status;
      MPI_Recv(&(Utemp[0][0]),(M+1)*(J-2),MPI_DOUBLE,MPI_ANY_SOURCE,0,comm,&status);
      s_rank = status.MPI_SOURCE; // get the rank of the sender
      // assign the values from the temporary array into the correct place in the final array U
      for (int i=0; i<J-2; i++) {
        for (int j=0; j<M+1; j++) {
            U[2][s_rank*(J-2)+i][j] = Utemp[i][j];

        }
      }
    }
    // receive the last two rows from the last process
    MPI_Recv(&(U[2][M-1][0]),(M+1)*2,MPI_DOUBLE,nproc-1,1,comm,MPI_STATUS_IGNORE);
	
	ofstream out {"horiz_final.csv"};
	out<<fixed<<setprecision(4);			
		for(int i=0; i<=M; ++i){
			for(int j=0; j<=M; ++j){		
				out<<U[2][i][j]<<" ";
			}
			out<<endl;
		}
	out.close();
	cout << "Rank 0 finished writing U values into a file" << endl;

	cout << "Runtime: " << run_time << endl;
}
	
	




delete[] U;
delete[] U2;
delete[] U3;
delete[] Ur;
delete[] Ur2;
delete[] Ur3;
delete[] Utemp;
delete[] Utemp1;

MPI_Finalize();


}