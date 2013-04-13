// psnr_mae.cpp	
// date l'immagine da filtrare e l'immagine filtrata calcola il PSNR e il MAE

//#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
//#include <iostream>

FILE *fpa;
FILE *fpb;
unsigned char *pxa;
unsigned char *pxb;

typedef struct{		// attributi file, dimensioni x*y e nome del file
  int x;
  int y;
  char fname[30];
}f_att;

f_att fa;
f_att fb;

// funzione che controlla se l'estenzione è bmp
int ctr_bmp(char nomefile[30]){
  int l;
  char f_ext[5];
  l=strlen(nomefile);      //mi da la lunghezza della stringa
  if(l<5){
    //cout<<"\nErrore, il nome del file immesso e' troppo corto ...\n";
    printf("\nErrore, il nome del file immesso e' troppo corto ...\n");
    return 1;
  }
  for(int i=0;i<5;i++) f_ext[i]=nomefile[i+(l-4)];  // prendo gli ultimi 4 caratteri del filename
  if(strcmp(f_ext,".bmp")) return 1;
  return 0;
}

f_att get_bmp_dim(FILE *fp){	// prelevo dimensioni dalla bmp
  int i;
  f_att fatt;
	
  unsigned char pxy[6];	// px sarà 6 byte, 4 per x, 2 per y

  for(i=0;i<6;i++){          //  lettura dimensioni immagine bmp,  dal byte 19 al 22 della bmp
    fseek(fp,(i+18)*sizeof(unsigned char),SEEK_SET);
    fread(&pxy[i],sizeof(unsigned char),1,fp);
  }
  fatt.x=0;
  for(i=0;i<4;i++){		// i primi 4 byte di pxy[]
    fatt.x=fatt.x+pow(256,i)*(int)pxy[i];
  }
  fatt.y=0;
  for(i=0;i<2;i++){		// i primi 4 byte di pxy[]
    fatt.y=fatt.y+pow(256,i)*(int)pxy[i+4];
  }
  return fatt;
}


int create_grayscale_matrix(unsigned char *&p,int x,int y,FILE *fp){
  int i,j,f;	// f mi serve per saltare i byte nulli
  int k,t,pi;   // t sarà il num di byte di una riga incluso i byte nulli (t=3*x se non ce ne sono)
  unsigned char pxb;
  float div=-1.0;

  p=new unsigned char[x*y];
	
  // Controllo se il n° di byte di ogni riga è divisibile per 4
  // Se non è divisibile per 4 ci saranno byte nulli aggiunti per rendere la riga divisibile per 4
  t=x; 	// n°byte per riga (ogni pixel è 1 byte)
  while(div!=1){
    div=t/4.0;
    div=div/int(div);		// se multiplo di 4 div=1 altrimenti t++
    if(div!=1) t++;      //senza if incrementa t anche quando esce dal while
  }

  // Fine controllo divisibilità per 4
  // Comincio a mettere i pixel nella matrice
  pi=0;
  k=1078;       // devo saltare l'intestazione, lunga 1078 byte e k va da 0 in poi
  f=1078-(t-x);
  fseek(fp,0,SEEK_SET);
  // nella bitmap i pixel vanno prelevati da sinistra verso destra e dal basso verso l'alto
  for(i=(y-1);i>=0;i--){
    f=f+(t-x)+x;		// lo metto al primo byte nullo della riga corrente
    for(j=0;j<x;j++){
      fseek(fp,k*sizeof(unsigned char),SEEK_SET);
      fread(&pxb,sizeof(unsigned char),1,fp);
      k++;
      if(t>x && k==f) k=k+(t-x);   //se sono al byte nullo salto di t-x byte
      if(pi<x*y) p[pi]=pxb;
      pi++;
    }
  }
  return 0;
}

int get_k(int i,int j,f_att fatt){
  if(!(0<i<fatt.x+1) || !(0<j<fatt.y+1)) return -1;
  return j-1+fatt.x*(fatt.y-i);
}

void psnr_mae_calculator(){
  int i,j,k;
  long double psnr,mae,a;
  long double psnr_add=0.0,mae_add=0.0;
	
  for(i=1;i<(fa.y+1);i++){
    for(j=1;j<(fa.x+1);j++){
      k=get_k(i,j,fa);
      psnr_add=psnr_add + pow( abs((double)pxb[k]-(double)pxa[k]),2);
      mae_add=mae_add + abs((double)pxb[k]-(double)pxa[k]);
    }
  }
  a=255*255;
  a=a*(fa.x*fa.y);
  a=a/psnr_add;
  psnr=10*log10(a);
  mae=mae_add/(fa.x*fa.y);
  //cout<<"\nPSNR = "<<psnr<<" ; MAE = "<<mae<<endl;
  //printf("\nPSNR = %Lf ; MAE = %Lf\n",psnr,mae);
  printf("%Lf\t%Lf\n", psnr, mae);
}

int main(int argc,char* argv[]){
  f_att tmp;	

  if(argc==1){
    strcpy(fa.fname,"a.bmp");
    strcpy(fb.fname,"b.bmp");
  }
  if(argc==3){
    strcpy(fa.fname,argv[1]);
    strcpy(fb.fname,argv[2]);
  }
  if(argc==2 || argc>3){
    //cout<<"\nPassare i nomi delle due immagini, (es. ""psnr_mae original.bmp restored.bmp"")\n";
    printf("\nPassare i nomi delle due immagini, es. 'psnr_mae original.bmp restored.bmp'.\n");
    printf("Se non si passano i nomi, il programma cerca i files 'a.bmp' e 'b.bmp'.\n");
    exit(1);
  }
  fpa=fopen(fa.fname,"rb");
  if(fpa==NULL){
    printf("\nErrore nell'apertura del file, file '%s' non trovato o di un tipo non valido !!\n",fa.fname);
    //cout<<"\nErrore nell'apertura del file, file non trovato o di un tipo non valido !!\n";
    //getchar();
    exit(1);
  }
  fpb=fopen(fb.fname,"rb");
  if(fpb==NULL){
    printf("\nErrore nell'apertura del file, file '%s' non trovato o di un tipo non valido !!\n",fb.fname);
    //cout<<"\nErrore nell'apertura del file, file non trovato o di un tipo non valido !!\n";
    //getchar();
    exit(1);
  }
  tmp=get_bmp_dim(fpa);
  fa.x=tmp.x;
  fa.y=tmp.y;
  tmp=get_bmp_dim(fpb);
  fb.x=tmp.x;
  fb.y=tmp.y;
  //cout<<"Immagine originale  :   "<<fa.fname<<endl;
  //cout<<"Immagine filtrata  :   "<<fb.fname<<endl;
  //printf("Immagine originale  :  %s\n",fa.fname);
  //printf("Immagine filtrata  :  %s\n",fb.fname);
  create_grayscale_matrix(pxa,fa.x,fa.y,fpa);
  create_grayscale_matrix(pxb,fb.x,fb.y,fpb);
  psnr_mae_calculator();
  fclose(fpa);
  fclose(fpb);
  delete pxa;
  delete pxb;
  //getchar();
  return 0;
}
