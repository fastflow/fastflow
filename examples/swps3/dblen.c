#include "fasta.h"
#include <stdio.h>


int main(int argc, char** argv)
{
   FastaLib *lib;
   int len;

   if(argc < 2) return 1;

   lib = swps3_openLib(argv[1]);

   while(swps3_readNextSequence(lib, &len)) {
      printf("%d\t%s\n",len,swps3_getSequenceName(lib));
   }

   swps3_closeLib(lib);
   return 0;
}

