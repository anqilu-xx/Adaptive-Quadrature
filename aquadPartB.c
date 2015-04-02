/**
 * Author: Anqi Lu <s1419429@sms.ed.ac.uk>
 *
 * I. Implementation
 *
 * 1. Implementation of Farmer
 *
 * I created an array with two times as many rooms as the given count of worker
 * numbers to store task descriptions. The task to assign to the i th worker will
 * be stored in position 2i and position 2i + 1.
 *
 * The farmer compute the chunks and put them into array. Then it sends tasks to
 * corresponding worker, respectively. The memory of task array will be freed after
 * sending phase.
 *
 * The farmer keeps receiving messages as long as the messages it has received are
 * less than number of workers. This is implemented with a while loop. The farmer
 * receives one message at a time. Then, it retrieves area and the count of tasks
 * from the message. Following that, it adds the chunk area to the main area
 * result, and sets the number of tasks the sending worker has executed.
 *
 * 2. Implementation of Worker
 *
 * The worker receives task description from the farmer, and then computes result
 * locally with an adaptive quadrature function. Specifically, the function will
 * return a pointer pointing a double array which contains area as the first
 * element and count of times invoked. The count of times the function has been
 * invoked is computed by a static variable defined inside function. The variable
 * increases by one each time the function called. To make it easy, the function
 * puts area and count in a double array and returns the pointer. Finally, the
 * worker sends the pointer to the farmer.
 *
 * II. MPI Primitives
 *
 * 1. The farmer sends all tasks out before receiving any messages. All tasks
 * descriptions are stored in an array. Thus each time, the farmer finds the buffer
 * by forwarding the pointer by 2 positions, and sends 2 items to the process id,
 * which is increased by 1. Compared to computing one tasks and sending it
 * immediately, this code looks cleaner.
 *
 * 2. In receiving phase, the farmer receives from any source in a while loop.
 * Compared to using a for loop and waiting for messages from given process id, it
 * is more efficient.
 *
 * 3. The worker sends two items (area and count) with one message, data type
 * marked as double. Compared to sending a double type message containing area and
 * an int type message containing count respectively, this method reduces total
 * counts of messages sent, thus improves efficiency of communication. And it will
 * not affect the value to transfer the count as double type.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define EPSILON 1e-3
#define F(arg)  cosh(arg)*cosh(arg)*cosh(arg)*cosh(arg)
#define A 0.0
#define B 5.0

#define NEWTASK 0
#define RESULT 1

#define FARMER 0

#define SLEEPTIME 1

int *tasks_per_process;

double farmer(int);

void worker(int);

double * quad(double, double, double, double, double);

int main(int argc, char **argv ) {
  int i, myid, numprocs;
  double area;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  if(numprocs < 2) {
    fprintf(stderr, "ERROR: Must have at least 2 processes to run\n");
    MPI_Finalize();
    exit(1);
  }

  if (myid == 0) { // Farmer
    // init counters
    tasks_per_process = (int *) malloc(sizeof(int)*(numprocs));
    for (i=0; i<numprocs; i++) {
      tasks_per_process[i]=0;
    }
  }

  if (myid == 0) { // Farmer
    area = farmer(numprocs);
  } else { //Workers
    worker(myid);
  }

  if(myid == 0) {
    fprintf(stdout, "Area=%lf\n", area);
    fprintf(stdout, "\nTasks Per Process\n");
    for (i=0; i<numprocs; i++) {
      fprintf(stdout, "%d\t", i);
    }
    fprintf(stdout, "\n");
    for (i=0; i<numprocs; i++) {
      fprintf(stdout, "%d\t", tasks_per_process[i]);
    }
    fprintf(stdout, "\n");
    free(tasks_per_process);
  }
  MPI_Finalize();
  return 0;
}

double farmer(int numprocs) {
  double area = 0;
  double *task, result[2];
  int i, numResults = 0, numWorkers = numprocs - 1, source;

  // worker status
  MPI_Status status;

  // allocate tasks to each worker
  task = (double *) malloc(sizeof(double)*(2*numWorkers));

  // the chunk allocated to worker i will be stored in task[2i] and task[2i + 1],
  // where i = 0, 1, 2, 3 when numprocs = 5
  for (i = 0; i < numWorkers; i ++) {
    task[2*i] = (double) (A + i*(B - A)/numWorkers);
    task[2*i + 1] = (double) (A + (i + 1)*(B - A)/numWorkers);
  }

  // send tasks to workers
  for (i = 0; i < numWorkers; i ++) {
    MPI_Send(task + 2 * i, 2, MPI_DOUBLE, i + 1, NEWTASK, MPI_COMM_WORLD);
  }
  free(task);

  while(numResults < numWorkers) {
    MPI_Recv(&result, 2, MPI_DOUBLE, MPI_ANY_SOURCE, RESULT, MPI_COMM_WORLD, &status);
    area += result[0];
    numResults ++;
    source = status.MPI_SOURCE;
    tasks_per_process[source] = (int) result[1];
  }

  return area;
}

void worker(int mypid) {
  double task[2], left, right, fleft, fright, lrarea, *result;

  // farmer status
  MPI_Status status;

  // receive task description from farmer
  MPI_Recv(task, 2, MPI_DOUBLE, FARMER, NEWTASK, MPI_COMM_WORLD, &status);

  left = task[0];
  right = task[1];
  fleft = F(left);
  fright = F(right);
  lrarea = (fleft + fright) * (right - left) / 2;

  result = (double *) malloc(sizeof(double)*(2));

  // compute result area and count of quad function invoked locally
  result = quad(left, right, fleft, fright, lrarea);

  // as the result contains area value and the count of tasks, send two items
  MPI_Send(result, 2, MPI_DOUBLE, FARMER, RESULT, MPI_COMM_WORLD);

}

double * quad(double left, double right, double fleft, double fright, double lrarea) {
  double mid, fmid, larea, rarea, *result;

  // a static value to hold how many times the function is invoked
  static int count=0;

  // each time invoked, increment the count
  count++;

  mid = (left + right) / 2;
  fmid = F(mid);
  larea = (fleft + fmid) * (mid - left) / 2;
  rarea = (fmid + fright) * (right - mid) / 2;
  if( fabs((larea + rarea) - lrarea) > EPSILON ) {
    larea = *(quad(left, mid, fleft, fmid, larea));
    rarea = *(quad(mid, right, fmid, fright, rarea));
  }
  result = (double *) malloc(sizeof(double)*(2));

  result[0] = larea + rarea;

  // to return count in a double array with area, convert it from int to double
  result[1] = (double) count;

  // return the pointer pointing to the array holding area and task count
  return result;
}
