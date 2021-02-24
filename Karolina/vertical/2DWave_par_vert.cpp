#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <mpi.h>

using namespace std;

int main(int argc, char* argv[]){

constexpr int M = 2305;      // M length intervals.
double T = 1;                // final time.
constexpr int J = 770;       // columns per process
int nproc = 3;               // number of processes
double dt = 0.2/M;
int N = T/dt;
int t1=0.333/dt, t2=0.666/dt, t3=N; // points at which we want to print our results.
double dy = 2./M;
double dy2 = dy*dy;
double dtdy2 = dt*dt/dy2;
int rank, size, y_shift, s_rank;
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
// -------------------- VERTICAL STRIPS ----------------------------------
// -----------------------------------------------------------------------
// -------------------- array init ---------------------------------------
// -----------------------------------------------------------------------

// global empty 3D variable for storing the final sol
// stays the same as in horizontal case
double*** U = new double**[3]; 
double**  U2 = new double*[3*(M+1)];
double*   U3 = new double[3*(M+1)*(M+1)];
for (int i=0; i<3; i++) {
    for (int j=0; j<M+1; j++) {
        U2[(M+1)*i+j] = U3 + ((M+1)*i+j)*(M+1);
    }
    U[i] = U2 + (M+1)*i;
}

// swapped # of rows and # of cols compared to the horizontal case
double*** Ur = new double**[3]; 
double**  Ur2 = new double*[3*(M+1)];
double*   Ur3 = new double[3*(M+1)*J];
for (int i=0; i<3; i++) {
    for (int j=0; j<M+1; j++) {
        Ur2[(M+1)*i+j] = Ur3 + ((M+1)*i+j)*J;
    }
    Ur[i] = Ur2 + (M+1)*i;
}

// for transposing the arrays when sending (2D is sufficient)
double** UrtS = new double*[J];
double*  UrtS1 = new double[J*(M+1)];
for (int i=0; i<J; i++) {
    UrtS[i] = UrtS1 + (M+1)*i;
}
// for transposing the arrays when receiving 
double** UrtR = new double*[J];
double*  UrtR1 = new double[J*(M+1)];
for (int i=0; i<J; i++) {
    UrtR[i] = UrtR1 + (M+1)*i;
}

// stays the same as in horizontal case
double** Utemp = new double*[J-2];
double*  Utemp1 = new double[(J-2)*(M+1)];
for (int i=0; i<J-2; i++) {
    Utemp[i] = Utemp1 + (M+1)*i;
}

// for sending the last two rows (columns) 
double** U2r = new double*[2];
double*  U2r1 = new double[2*(M+1)];
for (int i=0; i<2; i++) {
    U2r[i] = U2r1 + (M+1)*i;
}


// ---- x (=i) from 0 to M   ------
// ---- y (=j) from 0 to J-1 ------

// -----------------------------------------------------------------------
// -------------- Set up initial and boundary conditions -----------------
// -----------------------------------------------------------------------

// initial conditions 
for (int i=0; i<M+1; ++i){       // x-dim
	for (int j=0; j<J; ++j){     // y-dim
		y_shift = rank*(J-2)+j;
		Ur[0][i][j] = Ur[1][i][j] = exp( -40 * ( (i*dy-1-0.4)*(i*dy-1-0.4) + (y_shift*dy-1)*(y_shift*dy-1) ) );
	}
}

// boundary conditions
// fix x=0 and x=M, loop over y (top and bottom)
for (int i=0; i<J; ++i){ 	
	Ur[0][0][i] = Ur[0][M][i] = Ur[1][0][i] = Ur[1][M][i] = Ur[2][0][i] = Ur[2][M][i] = 0; 	
}

// fix y=0 (rank 0) and y=J-1 (rank nproc-1), loop over rows (x) (left anr right)
if (rank == 0) {
	for (int j=0; j<M+1; ++j){ 	
		Ur[0][j][0] = Ur[1][j][0] = Ur[0][j][0] = 0;
	}	
}
if (rank == nproc-1) {
	for (int j=0; j<M+1; ++j){ 	
		Ur[0][j][J-1] = Ur[1][j][J-1] = Ur[2][j][J-1] = 0;
	}
}

// ----------------------------------------------------------------------------------------
// --------------------------- SEND THE INITIAL CONDITIONS TO RANK 0 ----------------------
// ----------------------------------------------------------------------------------------

// // since receiving is row-major, transpose Ur[0]
// for (int i=0; i<J; i++) {
// 	for (int j=0; j<M+1; j++) {
// 		UrtS[i][j] = Ur[0][j][i];
// 	}
// }

// // send the first J-2 rows from each process (to avoid overlapping)
// if (rank != 0) {
//     MPI_Ssend(&(UrtS[0][0]),(M+1)*(J-2),MPI_DOUBLE,0,0,comm);
// }
// // send the last 2 rows from the last process (= last two columns in Ur)
// if (rank == nproc-1) {
// 	  MPI_Ssend(&(UrtS[J-2][0]),(M+1)*2,MPI_DOUBLE,0,1,comm);
// }

// if (rank == 0) {
// 	// the first J-2 columns of U taken from Ur in rank 0 
// 	for (int i=0; i<M+1; i++) {
//       for (int j=0; j<J-2; j++) {
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
//             U[0][j][s_rank*(J-2)+i] = Utemp[i][j];

//         }
//       }
//     }


//     // receive the last two rows from the last process
//     // MPI_Recv(&(U[0][M-1][0]),(M+1)*2,MPI_DOUBLE,nproc-1,1,comm,MPI_STATUS_IGNORE);
//     MPI_Recv(&(U2r[0][0]),(M+1)*2,MPI_DOUBLE,nproc-1,1,comm,MPI_STATUS_IGNORE);

//     for (int i=0; i<2; i++) {
//     	for (int j=0; j<M+1; j++) {
//     		U[0][j][nproc*(J-2)+i] = U2r[i][j];
//     	}
//     }
//  	// print initial U values to file, row by row.
// 	ofstream out {"vert_t0.csv"};
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
	
	for (int i=1; i<M; ++i){         // x from 1 to M-1
		for (int j=1; j<J-1; ++j){   // y from 1 to J-2		
		Ur[2][i][j] = 2*Ur[1][i][j] - Ur[0][i][j]	 
					 	 + dtdy2*( Ur[1][i+1][j] + Ur[1][i-1][j] + Ur[1][i][j+1] + Ur[1][i][j-1] - 4*Ur[1][i][j] ); 	
		}		
	}

	MPI_Barrier(comm);

	// ---------------- HALO SWAPPING -------------------------------------------

	// since receiving is row-major, transpose Ur[2] to both send and receive transpose of Ur (UrtS and UrtR)
	for (int i=0; i<J; i++) {
		for (int j=0; j<M+1; j++) {
			UrtS[i][j] = Ur[2][j][i]; // send copy of transpose of Ur
			UrtR[i][j] = Ur[2][j][i]; // receive copy of transpose of Ur with swapped halo after Recv
		}
	}

	if (rank > 0) {
		MPI_Ssend(&(UrtS[1][0]), M+1, MPI_DOUBLE, rank-1, 0, comm);
		MPI_Recv(&(UrtR[0][0]), M+1, MPI_DOUBLE, rank-1, 0, comm, MPI_STATUS_IGNORE);
	}

	if (rank < nproc-1) {
	    MPI_Recv(&(UrtR[J-1][0]), M+1, MPI_DOUBLE, rank+1, 0, comm, MPI_STATUS_IGNORE);
		MPI_Ssend(&(UrtS[J-2][0]), M+1, MPI_DOUBLE, rank+1, 0, comm);
	}

	// transpose UrtR back to put it into Ur[2]
	for (int i=0; i<M+1; i++) {
		for (int j=0; j<J; j++) {
			Ur[2][i][j] = UrtR[j][i];
		}
	}

	MPI_Barrier(comm);

 //   // print out files for fixed times	
	// if(t==t1 || t==t2 || t==t3){
	// 	// transpose	
	// 	for (int i=0; i<J; i++) {
	// 		for (int j=0; j<M+1; j++) {
	// 			UrtS[i][j] = Ur[2][j][i];
	// 		}
	// 	}

	// 	// send the first j-2 rows from each process (to avoid overlapping)
	// 	if (rank != 0) {
	// 	  MPI_Ssend(&(UrtS[0][0]),(M+1)*(J-2),MPI_DOUBLE,0,0,comm);
	// 	}

	// 	// send the last 2 rows from the last process
	// 	if (rank == nproc-1) {
	// 		  MPI_Ssend(&(UrtS[J-2][0]),(M+1)*2,MPI_DOUBLE,0,1,comm);
	// 	}

	// 	if (rank == 0) {
	// 		// the first J-2 columns of U taken from Ur in rank 0 
	// 	for (int i=0; i<M+1; i++) {
	//       for (int j=0; j<J-2; j++) {
	//           U[2][i][j] = Ur[2][i][j];
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
	//             U[2][j][s_rank*(J-2)+i] = Utemp[i][j];

	//         }
	//       }
	//     }
	//     // receive the last two rows from the last process
	//     MPI_Recv(&(U2r[0][0]),(M+1)*2,MPI_DOUBLE,nproc-1,1,comm,MPI_STATUS_IGNORE);

	//     for (int i=0; i<2; i++) {
	//     	for (int j=0; j<M+1; j++) {
	//     		U[2][j][nproc*(J-2)+i] = U2r[i][j];
	//     	}
	//     }
	// 		stringstream ss;
	// 		ss << fixed << setprecision(2) << t*dt; // this ensures that the double value gets converted
	// 		string time = ss.str();						 // to string with only 2 trailing digits.
			
	// 		ofstream out {"vert_t"+ss.str()+".csv"};
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
	for (int i=0; i<M+1; ++i){
		for (int j=0; j<J; ++j){		
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


// Write into a file
// transpose	
for (int i=0; i<J; i++) {
	for (int j=0; j<M+1; j++) {
		UrtS[i][j] = Ur[2][j][i];
	}
}

// send the first j-2 rows from each process (to avoid overlapping)
if (rank != 0) {
  MPI_Ssend(&(UrtS[0][0]),(M+1)*(J-2),MPI_DOUBLE,0,0,comm);
}

// send the last 2 rows from the last process
if (rank == nproc-1) {
	  MPI_Ssend(&(UrtS[J-2][0]),(M+1)*2,MPI_DOUBLE,0,1,comm);
}

if (rank == 0) {
	// the first J-2 columns of U taken from Ur in rank 0 
for (int i=0; i<M+1; i++) {
  for (int j=0; j<J-2; j++) {
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
        U[2][j][s_rank*(J-2)+i] = Utemp[i][j];

    }
  }
}
// receive the last two rows from the last process
MPI_Recv(&(U2r[0][0]),(M+1)*2,MPI_DOUBLE,nproc-1,1,comm,MPI_STATUS_IGNORE);

for (int i=0; i<2; i++) {
	for (int j=0; j<M+1; j++) {
		U[2][j][nproc*(J-2)+i] = U2r[i][j];
	}
}
	
	ofstream out {"vert_final.csv"};
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
delete[] UrtS;
delete[] UrtS1;
delete[] UrtR;
delete[] UrtR1;
delete[] Utemp;
delete[] Utemp1;
delete[] U2r;
delete[] U2r1;

MPI_Finalize();


}