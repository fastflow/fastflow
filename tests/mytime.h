
#include <sys/time.h> 

typedef struct{
  struct timeval first,  second,  lapsed; 
  
}cronos_t;

#define t_new_clock(NAME)			\
  cronos_t NAME;

#define t_start(NAME)					\
	gettimeofday (&((NAME).first), NULL);   

#define t_stop(NAME)					\
  gettimeofday (&((NAME).second), NULL);	\
  if ((NAME).first.tv_usec > (NAME).second.tv_usec) {	\
    (NAME).second.tv_usec += 1000000;			\
    (NAME).second.tv_sec--;				\
  } 

#define t_get_microsec(NAME)				\
  ((NAME).second.tv_usec - (NAME).first.tv_usec)

#define t_get_sec(NAME)				\
  ((NAME).second.tv_sec - (NAME).first.tv_sec) 

#define t_get_total_usec(NAME) \
	(((NAME).second.tv_sec-(NAME).first.tv_sec) * 1000000 +	\
	 ((NAME).second.tv_usec-(NAME).first.tv_usec))

