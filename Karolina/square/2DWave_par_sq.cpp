#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <tgmath.h>

using namespace std;

int main(int argc, char* argv[]){

constexpr int M = 2305;       // M length intervals.
double T = 1;                 // final time.
constexpr int J = 1154;       // length of side of the square
constexpr int nproc = 4;      // number of processes
int sqnproc = sqrt(nproc);
double dt = 0.2/M;
int N = T/dt;
int t1=0.333/dt, t2=0.666/dt, t3=N; // points at which we want to print our results.
double dy = 2./M;
double dy2 = dy*dy;
double dtdy2 = dt*dt/dy2;
int rank, size, x_shift, y_shift, s_rank;
int irow, icol;
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
// ------------------------- SQUARES -------------------------------------
// -----------------------------------------------------------------------
// -------------------- array init ---------------------------------------
// -----------------------------------------------------------------------

// global empty 3D array for storing the final sol (M+1)x(M+1)
double*** U = new double**[3]; // m=3 n=M+1 o=M+1
double**  U2 = new double*[3*(M+1)];
double*   U3 = new double[3*(M+1)*(M+1)];
for (int i=0; i<3; i++) {
    for (int j=0; j<M+1; j++) {
        U2[(M+1)*i+j] = U3 + ((M+1)*i+j)*(M+1);
    }
    U[i] = U2 + (M+1)*i;
}

// JxJ array for each process
double*** Ur = new double**[3]; 
double**  Ur2 = new double*[3*J];
double*   Ur3 = new double[3*J*J];
for (int i=0; i<3; i++) {
    for (int j=0; j<J; j++) {
        Ur2[J*i+j] = Ur3 + (J*i+j)*J;
    }
    Ur[i] = Ur2 + J*i;
}

// (J-2)x(J-2) dimensional array for sending values from each process
double** Usend = new double*[J-2];
double*  Usend1 = new double[(J-2)*(J-2)];
for (int i=0; i<J-2; i++) {
    Usend[i] = Usend1 + (J-2)*i;
}

// vector with nproc rows and (J-2)*(J-2) cols to store the gathered data
double** Utemp = new double*[nproc];
double*  Utemp1 = new double[nproc*(J-2)*(J-2)];
for (int i=0; i<nproc; i++) {
    Utemp[i] = Utemp1 + (J-2)*(J-2)*i;
}

// ---- x (=i) from 0 to J-1 ------
// ---- y (=j) from 0 to J-1 ------

// -----------------------------------------------------------------------
// -------------- Set up initial and boundary conditions -----------------
// -----------------------------------------------------------------------

// Initial conditions: 
for (int i=0; i<J; ++i){         // x-dim
	for (int j=0; j<J; ++j){     // y-dim
		x_shift = floor(rank/sqnproc)*(J-2)+i;
		y_shift = (rank % sqnproc)*(J-2)+j;
		Ur[0][i][j] = Ur[1][i][j] = exp( -40 * ( (x_shift*dy-1-0.4)*(x_shift*dy-1-0.4) + (y_shift*dy-1)*(y_shift*dy-1) ) );
	}
}
// Boundary conditions:
// first row of processes (upper boundary)
if (floor (rank / sqnproc) == 0) {
	for(int i=0; i<J; i++) {
		Ur[0][0][i] = Ur[1][0][i] = Ur[2][0][i] = 0; // top
	}
}

// last row of processes (bottom boundary)
if (floor (rank / sqnproc) == sqnproc-1) {
	for(int i=0; i<J; i++) {
		Ur[0][J-1][i] = Ur[1][J-1][i] = Ur[2][J-1][i] = 0; // bottom
	}
}

// first column of processes (left boundary)
if (rank % sqnproc == 0) {
	for(int i=0; i<J; i++) {
		Ur[0][i][0] = Ur[1][i][0] = Ur[2][i][0] = 0; // left
	}
}
// last column of processes (right boundary)
if (rank % sqnproc == sqnproc-1) {
	for(int i=0; i<J; i++) {
		Ur[0][i][J-1] = Ur[1][i][J-1] = Ur[2][i][J-1] = 0; // right
	}
}

// ----------------------------------------------------------------------------------------
// --------------------------- SEND THE INITIAL CONDITIONS TO RANK 0 ----------------------
// ----------------------------------------------------------------------------------------

// // make shorter J-2 dim squares for sending blocks of initial conditions
// for (int i=0; i<J-2; i++) {
// 	for (int j=0; j<J-2; j++) {
// 		Usend[i][j] = Ur[0][i][j];
// 	}
// }

// MPI_Gather(Usend[0], (J-2)*(J-2), MPI_DOUBLE, Utemp[0], (J-2)*(J-2), MPI_DOUBLE, 0, comm);
// //---------------------
// // intention: save (J-2)*(J-2) elements sent by each process into one row of Utemp (which is of the exact same dimension)
// // saving row-major therefore as many rows as processes
// // -----------------------

// MPI_Barrier(comm);

// // the ranks in the last column sends their last two columns (which have J-2 rows )
// if (rank % sqnproc == sqnproc-1) {
// 	for (int i=0; i<J-2; i++) { // loop over rows of square
// 		MPI_Ssend(&(Ur[0][i][J-2]), 2, MPI_DOUBLE, 0, (J-2)*(floor(rank/sqnproc))+i, comm);
// 	}
// }

// // the ranks in the last row send their last two rows (which have J-2 columns)
// if (floor(rank/sqnproc) == sqnproc-1) {
// 	MPI_Ssend(&(Ur[0][J-2][0]), J-2, MPI_DOUBLE, 0, 0, comm);
// 	MPI_Ssend(&(Ur[0][J-1][0]), J-2, MPI_DOUBLE, 0, 1, comm);
// }

// // the last rank sends the 2 rows (2 cols each) (the bottom right corner)
// if (rank == nproc-1) {
// 	MPI_Ssend(&(Ur[0][J-2][J-2]), 2, MPI_DOUBLE, 0, 2, comm);
// 	MPI_Ssend(&(Ur[0][J-1][J-2]), 2, MPI_DOUBLE, 0, 3, comm);
// }


// // assign values from temp array to U
// if (rank == 0) {
// 	for (int p=0; p<nproc; p++) {
// 		for (int j=0; j<(J-2)*(J-2); j++){
// 			irow = floor(j/(J-2)) + (J-2)*floor(p/sqnproc);
// 			icol = j%(J-2) + (J-2)*(p%sqnproc);
// 			U[0][irow][icol] = Utemp[p][j];
// 		}
// 	}
// }

// // receive last two columns
// if (rank == 0) {
// 	for (int p=0; p<nproc; p++) {
// 		if ( p % sqnproc == sqnproc-1) {
// 			for (int i=0; i<J-2; i++) {
// 				irow = (J-2)*(floor(p/sqnproc))+i;
// 				MPI_Recv(&(U[0][irow][M-1]), 2, MPI_DOUBLE, p, irow, comm, MPI_STATUS_IGNORE);
// 			}
// 		}
// 		// receive the last two rows
// 		if (floor(p/sqnproc) == sqnproc-1) {
// 			icol = (J-2)* (p % sqnproc);  
// 			MPI_Recv(&(U[0][M-1][icol]), J-2, MPI_DOUBLE, p, 0, comm, MPI_STATUS_IGNORE);
// 			MPI_Recv(&(U[0][M][icol]), J-2, MPI_DOUBLE, p, 1, comm, MPI_STATUS_IGNORE);
// 		}
// 		if (p == nproc-1) {
// 			MPI_Recv(&(U[0][M-1][M-1]), 2, MPI_DOUBLE, p, 2, comm, MPI_STATUS_IGNORE);
// 			MPI_Recv(&(U[0][M][M-1]), 2, MPI_DOUBLE, p, 3, comm, MPI_STATUS_IGNORE);
// 		}
// 	}

// 	// print initial U values to file, row by row.
// 	ofstream out {"sq_t0.csv"};

// 	out<<fixed<<setprecision(4);
// 		for(int i=0; i<=M; ++i){
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
	
	for (int i=1; i<J-1; ++i){          // x from 1 to J-2
		for (int j=1; j<J-1; ++j){      // y from 1 to M-1		
			Ur[2][i][j] = 2*Ur[1][i][j] - Ur[0][i][j]	 
						 	 + dtdy2*( Ur[1][i+1][j] + Ur[1][i-1][j] + Ur[1][i][j+1] + Ur[1][i][j-1] - 4*Ur[1][i][j] ); 	
		}		
	}
	MPI_Barrier(comm);
	// ---------------- HALO SWAPPING up-down -----------------------------------------

	// all rows except the first one send one row up (row-1)
	if (floor(rank/sqnproc) > 0) {
		MPI_Ssend(&(Ur[2][1][0]), J-1, MPI_DOUBLE, sqnproc*(floor(rank/sqnproc)-1) + (rank % sqnproc), 0, comm); //floor(rank/sqnproc)-1
		MPI_Recv(&(Ur[2][0][0]), J-1, MPI_DOUBLE, sqnproc*(floor(rank/sqnproc)-1) + (rank % sqnproc), 0, comm, MPI_STATUS_IGNORE);
	}

	// all rows except the last one send one row down (row+1)
	if (floor(rank/sqnproc) < sqnproc-1) {
	    MPI_Recv(&(Ur[2][J-1][0]), J-1, MPI_DOUBLE, sqnproc*(floor(rank/sqnproc)+1) + (rank % sqnproc) , 0, comm, MPI_STATUS_IGNORE); 
		MPI_Ssend(&(Ur[2][J-2][0]), J-1, MPI_DOUBLE, sqnproc*(floor(rank/sqnproc)+1) + (rank % sqnproc), 0, comm);
	}

	MPI_Barrier(comm); // to order the swaps - first vertical swapping, then horizontal swapping

	// ---------------- HALO SWAPPING LEFT-RIGHT -----------------------------------------

	// all columns except the first one send left (to rank-1)
	if (rank % sqnproc > 0) {
		for (int row=0; row<J; row++) { 
			MPI_Ssend(&(Ur[2][row][1]), 1, MPI_DOUBLE, rank-1, row, comm);
			MPI_Recv(&(Ur[2][row][0]), 1, MPI_DOUBLE, rank-1, row, comm, MPI_STATUS_IGNORE);
		}
	}

	// all columns except the last one send right (to rank+1)
	if (rank % sqnproc < sqnproc-1) {

		for (int row=0; row<J; row++) { 
			MPI_Recv(&(Ur[2][row][J-1]), 1, MPI_DOUBLE, rank+1, row, comm, MPI_STATUS_IGNORE);
			MPI_Ssend(&(Ur[2][row][J-2]), 1, MPI_DOUBLE, rank+1, row, comm);
		}
	}

 //   // print out files for fixed times	
	// if (t==t1 || t==t2 || t==t3) {
	// 	// --------Assign Ur back into U (same as for the initial condition) -------------------------------------

	// 	// make shorter J-2 dim squares for sending blocks of initial conditions
	// 	for (int i=0; i<J-2; i++) {
	// 		for (int j=0; j<J-2; j++) {
	// 			Usend[i][j] = Ur[2][i][j];
	// 		}
	// 	}

	// 	MPI_Gather(Usend[0], (J-2)*(J-2), MPI_DOUBLE, Utemp[0], (J-2)*(J-2), MPI_DOUBLE, 0, comm);
	// 	MPI_Barrier(comm);

	// 	// the ranks in the last column sends their last two columns (which have J-2 rows )
	// 	if (rank % sqnproc == sqnproc-1) {
	// 		for (int i=0; i<J-2; i++) { // loop over rows of square
	// 			MPI_Ssend(&(Ur[2][i][J-2]), 2, MPI_DOUBLE, 0, (J-2)*(floor(rank/sqnproc))+i, comm);
	// 		}
	// 	}

	// 	// the ranks in the last row send their last two rows (which have J-2 columns)
	// 	if (floor(rank/sqnproc) == sqnproc-1) {
	// 		MPI_Ssend(&(Ur[2][J-2][0]), J-2, MPI_DOUBLE, 0, 0, comm);
	// 		MPI_Ssend(&(Ur[2][J-1][0]), J-2, MPI_DOUBLE, 0, 1, comm);
	// 	}

	// 	// the last rank sends the 2 rows (2 cols each) (the bottom right corner)
	// 	if (rank == nproc-1) {
	// 		MPI_Ssend(&(Ur[2][J-2][J-2]), 2, MPI_DOUBLE, 0, 2, comm);
	// 		MPI_Ssend(&(Ur[2][J-1][J-2]), 2, MPI_DOUBLE, 0, 3, comm);
	// 	}

	// 	// assign values from temp array to U
	// 	if (rank == 0) {
	// 		for (int p=0; p<nproc; p++) {
	// 			for (int j=0; j<(J-2)*(J-2); j++){
	// 				irow = floor(j/(J-2)) + (J-2)*floor(p/sqnproc);
	// 				icol = j%(J-2) + (J-2)*(p%sqnproc);
	// 				U[2][irow][icol] = Utemp[p][j];
	// 			}
	// 		}
	// 	}

	// 	// receive last two columns
	// 	if (rank == 0) {
	// 		for (int p=0; p<nproc; p++) {
	// 			if ( p % sqnproc == sqnproc-1) {
	// 				for (int i=0; i<J-2; i++) {
	// 					irow = (J-2)*(floor(p/sqnproc))+i;
	// 					MPI_Recv(&(U[2][irow][M-1]), 2, MPI_DOUBLE, p, irow, comm, MPI_STATUS_IGNORE);
	// 				}
	// 			}
	// 			// receive the last two rows
	// 			if (floor(p/sqnproc) == sqnproc-1) {
	// 				icol = (J-2)* (p % sqnproc);  
	// 				MPI_Recv(&(U[2][M-1][icol]), J-2, MPI_DOUBLE, p, 0, comm, MPI_STATUS_IGNORE);
	// 				MPI_Recv(&(U[2][M][icol]), J-2, MPI_DOUBLE, p, 1, comm, MPI_STATUS_IGNORE);
	// 			}
	// 			if (p == nproc-1) {
	// 				MPI_Recv(&(U[2][M-1][M-1]), 2, MPI_DOUBLE, p, 2, comm, MPI_STATUS_IGNORE);
	// 				MPI_Recv(&(U[2][M][M-1]), 2, MPI_DOUBLE, p, 3, comm, MPI_STATUS_IGNORE);
	// 			}
	// 		}

	// 		// print initial U values to file, row by row.
	// 		stringstream ss;
	// 		ss << fixed << setprecision(2) << t*dt; // this ensures that the double value gets converted
	// 		string time = ss.str();						 // to string with only 2 trailing digits.
	// 		ofstream out {"sq_t"+ss.str()+".csv"};
	// 		out<<fixed<<setprecision(4);
	// 			for(int i=0; i<=M; ++i){
	// 				for(int j=0; j<=M; ++j){					
	// 					out<<U[2][i][j]<<" ";
	// 				}
	// 				out<<endl;
	// 			}	
	// 		out.close();

	// 		cout << "Rank 0 finished writing U values at t=" << t << endl;
	// 	}
	// }
	// update the previous times.
	for (int i=0; i<J; ++i){
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

// --------Assign Ur back into U (same as for the initial condition) -------------------------------------

// make shorter J-2 dim squares for sending blocks of initial conditions
for (int i=0; i<J-2; i++) {
	for (int j=0; j<J-2; j++) {
		Usend[i][j] = Ur[2][i][j];
	}
}

MPI_Gather(Usend[0], (J-2)*(J-2), MPI_DOUBLE, Utemp[0], (J-2)*(J-2), MPI_DOUBLE, 0, comm);
MPI_Barrier(comm);

// the ranks in the last column sends their last two columns (which have J-2 rows )
if (rank % sqnproc == sqnproc-1) {
	for (int i=0; i<J-2; i++) { // loop over rows of square
		MPI_Ssend(&(Ur[2][i][J-2]), 2, MPI_DOUBLE, 0, (J-2)*(floor(rank/sqnproc))+i, comm);
	}
}

// the ranks in the last row send their last two rows (which have J-2 columns)
if (floor(rank/sqnproc) == sqnproc-1) {
	MPI_Ssend(&(Ur[2][J-2][0]), J-2, MPI_DOUBLE, 0, 0, comm);
	MPI_Ssend(&(Ur[2][J-1][0]), J-2, MPI_DOUBLE, 0, 1, comm);
}

// the last rank sends the 2 rows (2 cols each) (the bottom right corner)
if (rank == nproc-1) {
	MPI_Ssend(&(Ur[2][J-2][J-2]), 2, MPI_DOUBLE, 0, 2, comm);
	MPI_Ssend(&(Ur[2][J-1][J-2]), 2, MPI_DOUBLE, 0, 3, comm);
}

// assign values from temp array to U
if (rank == 0) {
	for (int p=0; p<nproc; p++) {
		for (int j=0; j<(J-2)*(J-2); j++){
			irow = floor(j/(J-2)) + (J-2)*floor(p/sqnproc);
			icol = j%(J-2) + (J-2)*(p%sqnproc);
			U[2][irow][icol] = Utemp[p][j];
		}
	}
}

// receive last two columns
if (rank == 0) {
	for (int p=0; p<nproc; p++) {
		if ( p % sqnproc == sqnproc-1) {
			for (int i=0; i<J-2; i++) {
				irow = (J-2)*(floor(p/sqnproc))+i;
				MPI_Recv(&(U[2][irow][M-1]), 2, MPI_DOUBLE, p, irow, comm, MPI_STATUS_IGNORE);
			}
		}
		// receive the last two rows
		if (floor(p/sqnproc) == sqnproc-1) {
			icol = (J-2)* (p % sqnproc);  
			MPI_Recv(&(U[2][M-1][icol]), J-2, MPI_DOUBLE, p, 0, comm, MPI_STATUS_IGNORE);
			MPI_Recv(&(U[2][M][icol]), J-2, MPI_DOUBLE, p, 1, comm, MPI_STATUS_IGNORE);
		}
		if (p == nproc-1) {
			MPI_Recv(&(U[2][M-1][M-1]), 2, MPI_DOUBLE, p, 2, comm, MPI_STATUS_IGNORE);
			MPI_Recv(&(U[2][M][M-1]), 2, MPI_DOUBLE, p, 3, comm, MPI_STATUS_IGNORE);
		}
	}

	// print initial U values to file, row by row.
	ofstream out {"sq_final.csv"};
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
delete[] Usend;
delete[] Usend1;

MPI_Finalize();

}