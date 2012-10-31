/* Represents a task assigned to a worker */
typedef struct { 
  comp_t *a;	// base address of matrix A (input)
  comp_t *l;	// base address of matrix L (result)
} ff_task_t;
