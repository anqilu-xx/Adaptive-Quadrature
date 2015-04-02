/**
 * Author: Anqi Lu <s1419429@sms.ed.ac.uk>
 *
 * This is a program computing approximation to the integral by partitioning until
 * the trapezoidal approximation is "good enough", using Bag of Tasks pattern.
 *
 * I. Implementation
 *
 * In my implementation, the farmer process allocates tasks from bag to worker
 * processes, add tasks to the bag and accumulate the main area result, according
 * to the messages received from workers.
 *
 * 1. Implementation of Farmer
 *
 * The first step is to put the first task into bag, which is described by two
 * original endpoints, A and B.
 *
 * As long as there are still unfinished tasks in the bag or busy workers, the
 * farmer communicates with workers in round.
 *
 * In each round, the farmer sends messages with tasks descriptions and then wait
 * for one message from any source. When allocating tasks, the farmer tries to find
 * as many idle workers ( a worker is marked as "busy" if it is evaluating a task,
 * while it is marked as "idle" if it is waiting for new message from the farmer)
 * as remaining tasks in bag. I created an array to store workers' status: busy or
 * idle. Each time, the farmer checks if there are still tasks left and searches
 * for an idle worker in the array, then it sends new task message to the idle
 * worker.
 *
 * After sending message, it changes the receiver's status from idle to busy, and
 * decrease the idle worker count when increase the count of tasks finished by the
 * particular worker. When sending phase finished, the farmer waits for a new
 * message, which might come from any worker process. As the worker just sends one
 * message each time, it can be assumed as idle once farmer received a message from
 * it. Thus farmer marked the source process as idle. Then it checks the message
 * tag, which can be "new task" or "result". I used pointer pointing to an array
 * with four rooms to store any received message buffer. If it is a message with
 * two new task descriptions, the farmer put the two tasks into bag, respectively.
 * If it is a message containing the final result of a task, the farmer adds the
 * partition result to the main area result. And then it starts a new round sending
 * and receiving messages.
 *
 * Once there is no more task in the bag and all workers are idle, the farmer
 * breaks the while loop. It sends a TERMINATE message to every worker to indicate
 * them terminate.
 *
 * 2. Implementation of Worker
 *
 * The worker works in round, implemented with a while loop with permanent true
 * condition, as well.
 *
 * It waits for a new message coming from farmer. Once received message, it checks
 * if it is a TERMINATE message or a NEWTASK message. If it is a new task
 * description, the worker computes the area locally.
 *
 * Then it checks if the area is greater than given EPSILON. If true, it computes
 * the middle point of the given two points. And package them into two new tasks,
 * left point and middle one, and middle one and right one, in an array with four
 * rooms. Then it sends the pointer pointing the array to the farmer. If the result
 * is precise enough, it still puts the result in an array containing four rooms
 * and send the pointer to the farmer.
 *
 * Then it starts another round and waits for a new message. Once TERMINTE message
 * received, worker jumps out of the while loop and terminates.
 *
 * II. MPI Primitives
 *
 * 1. The farmer sends as many tasks out as possible each time. This will be more
 * efficient than one task allocated each time.
 *
 * 2. The farmer uses a buffer pointing to an array with 4 rooms to receive any
 * message, and always reads 4 items, regardless of its size, which could be 1 or
 * 4. As it cannot judge the size of the received data, it expects it at the
 * maximum possible size.
 *
 * 3. When sending TERMINATE messages, the farmer sends 0 item with a TERMINATE
 * flag, as the worker terminates once receiving the message tagged as TERMINATE
 * regardless of any other message content.
 *
 * 4. A worker only communicates with the farmer. Compared to sending to and
 * receiving from any unknown source, it avoids conflicts might occur.
 *
 * 5. A worker sends new tasks descriptions with a NEWTASK tag, and result with a
 * RESULT tag. The length of message content is 1 and 4 respectively. It helps the
 * farmer to notice what it just received.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "stack.h"

#define EPSILON 1e-3
#define F(arg)  cosh(arg)*cosh(arg)*cosh(arg)*cosh(arg)
#define A 0.0
#define B 5.0

#define SLEEPTIME 1

// message tags
#define NEWTASK 0
#define RESULT 1
#define TERMINATE 2

// worker status
#define IDLE 0
#define BUSY 1

// farmer process id
#define FARMER 0

int *tasks_per_process;

double farmer(int);

void worker(int);

double getArea(double, double);

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
  // tag could be either of the following: 0 for new task description, 1 for
  // result, 2 for terminate
  int i, j, who, tag;
  // first task, received task, new task
  double points[2], data[4], *task;
  // result area
  double area = 0.0;
  // an array to hold free workers available to process new task
  // in the array, there could be two flags suggesting the status of a specific worker process
  int workers[numprocs - 1];
  // an int to count how many free workers are there
  int idleWorkersNum = 0;
  // an int to tag the available worker id
  int idleWorker;

  // worker status
  MPI_Status status;

  // store the bag of tasks
  stack *taskBag = new_stack();

  // add the first task to bag
  points[0] = A;
  points[1] = B;
  push(points, taskBag);

  // initialize all workers as idle
  for (i = 0; i < numprocs - 1; i++) {
    workers[i] = IDLE;
    idleWorkersNum ++;
  }

  // the farmer keeps working as long as there are still tasks in bag or busy workers
  while(!is_empty(taskBag) || idleWorkersNum != numprocs - 1 ) {

      // find as many idle workers as possible when there are still tasks in bag
      for (j = 0; j < numprocs - 1 && !is_empty(taskBag); j++){

        if(workers[j] == IDLE){
          idleWorker = j + 1;

          // allocate task
          task = pop(taskBag);

          // send new task to an idle worker
          MPI_Send(task, 2, MPI_DOUBLE, idleWorker, NEWTASK, MPI_COMM_WORLD);

          // set the status of the worker who has just been assigned a task to BUSY
          workers[idleWorker - 1] = BUSY;
          idleWorkersNum --;

          // add a task to the worker who has just been assigened a task
          tasks_per_process[idleWorker] ++;

          // free memory allocated to task
          free(task);
        }
      }

      // receive data from any source
      MPI_Recv(data, 4, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      // free the worker just sent the message
      who = status.MPI_SOURCE;
      workers[who - 1] = IDLE;

      // increment the count od idle workers
      idleWorkersNum ++;

      tag = status.MPI_TAG;

      // check if the data received is new tasks descriptions or a final result
      if (tag == NEWTASK) {

        // put new tasks descriptions into bag
        push(data, taskBag);
        push(data+2, taskBag);

      } else if (tag == RESULT) {

        // accumulate the area
        area += data[0];
      }
  }

  // once there are no tasks in bag and all the workers are idle,
  // terminate all the workers
  for (i = 1; i < numprocs; i++) {
    MPI_Send(task, 0, MPI_DOUBLE, i, TERMINATE, MPI_COMM_WORLD);
  }

  return area;
}

void worker(int mypid) {
  // received task
  double left, right, mid, larea, rarea, lrarea;
  // new task description
  double task[4], result[4];
  // tag
  int tag;
  // farmer status
  MPI_Status status;
  
  while(1){
    // get message from farmer
    MPI_Recv(task, 2, MPI_DOUBLE, FARMER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    tag = status.MPI_TAG;

    // check if the message is a TERMINATE message or a new task description
    if (tag != TERMINATE){

      // get task description
      left = task[0];
      right = task[1];
      mid = (left + right) / 2;

      // execute task
      larea = getArea(left, mid);
      rarea = getArea(mid, right);
      lrarea = getArea(left, right);
      usleep(SLEEPTIME);

      // decide what to return
      if( fabs((larea + rarea) - lrarea) > EPSILON ) {
        // add new task description
        result[0] = left;
        result[1] = mid;
        result[2] = mid;
        result[3] = right;
  
        MPI_Send(result, 4, MPI_DOUBLE, FARMER, NEWTASK, MPI_COMM_WORLD);
      } else {
        // return result area
        result[0] = larea + rarea;
        MPI_Send(result, 1, MPI_DOUBLE, FARMER, RESULT, MPI_COMM_WORLD);
      }
    } else {
      // terminate process if TERMINATE message received
      break;
    }
  }
}

/**
 * A function to compute area with given left point and right point.
 */
double getArea(double left, double right) {
  double fleft, fright, area;
  fleft = F(left);
  fright = F(right);
  area = (fleft + fright) * (right - left) / 2;
  return area;
}
