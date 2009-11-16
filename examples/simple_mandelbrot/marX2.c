/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */

/*
   Author: Marco Aldinucci.   
   email:  aldinuc@di.unipi.it
           marco@pisa.quadrics.com
   date :  15/11/97

   Modified: Massimo Coppola
   email:    coppola@di.unipi.it
   date:     23/11/2001

*/

/*  gcc -Wall -ansi -c -O2 -I/usr/X11R6/include marX2.c  for Linux LAN */
/*  cc -fast -Xc -c -I/usr/openwin/include marX2.c for SUN or MEIKO */
/* Specify the path where X11 include are ... */

/* ar -rv marX2.a marX2.o */
/* ranlib marX2.a */


#include "marX2.h" 

#ifndef PI
#define PI 3.1415926535 
#endif

/*#define DEBUG(x) x;fflush(stdout);*/
/* defined empty to remove any  debug printf */
#define DEBUG(x)

void ShowTM ();
void ChooseColorMap(int);
void RainbowColorMap(int); 

static Display * display;
static Window window;

#define SCREEN DefaultScreen(display)

static GC gc;
static int fg;
static int bg;
static XImage * image;
static Colormap colormap;


static char *m_image_buffer=NULL; /* pointer to the image buffer area */

static int m_h,m_w; /* size of our window */

static int iScreen; /* integer id of the default screen of display */
static int dDepth;  /* display bit depth: only 8,12,24 bits are supported */
static unsigned int sizeofPixel;  /* rounded up pixel size  */
static int rounded_length;        /* rounded up length of a line in pixels */


static int primo=0; /* ?? 0 if the colormap has not yet been changed ?? */

/* function that opens a window over the default screen */
void SetupXWindows(int w, int h, int setup_color, char *display_name,
		   const char *window_title)
{
  XEvent event;
  XGCValues gcvalues;
  int mask;
  
  display = XOpenDisplay((display_name ? display_name : ""));
  if (!display)
	{
	  printf("Error in open X display\n");
	  exit (1);
	}

  iScreen = DefaultScreen(display);
  dDepth = DefaultDepth(display,iScreen);
  switch (dDepth)
    {
    case 8:
      sizeofPixel=1;
      break;
    case 16:
      sizeofPixel=2;
      break;
    case 24:
    case 32: /* It should work; on my system 24 bits pixels are word-aligned */
      sizeofPixel=4;
      break;
    default:
      printf("SetupXWindows : unsupported Display depth %d\n",dDepth);
    }

  DEBUG(printf("Default display depth %d\n", dDepth));

  m_w=w;
  m_h=h;

  /* scanlines are rounded to 32 pixels to avoid any misalignment */
  rounded_length = (m_w+31)&(~31); 
  /* allocate window buffer space */
  m_image_buffer = /*(XImage *)*/
    (char *) calloc(sizeof(char)*sizeofPixel,rounded_length*m_h);

  fg=Black();
  bg=White();

  window = XCreateSimpleWindow(display, DefaultRootWindow(display),
			       0,0,w,h,2,fg,bg);

  mask = ExposureMask | ButtonPressMask | KeyPressMask;
  XSelectInput(display, window,mask);
  XStoreName(display, window, window_title);
  XMapWindow(display, window);

  for (;;)
	{
	  XNextEvent(display, &event);
	  if (event.type == Expose)
		break;
	}
  
  gcvalues.foreground = fg;
  gcvalues.background = bg;
  mask = GCForeground | GCBackground;
  gc = XCreateGC(display, window, mask, &gcvalues);

  DEBUG(printf("Window opened\n"));
}


void CloseXWindows()
{
  /* should close the window and free all resources */

  XCloseDisplay(display);

}

/* service function to convert an array of bytes to an array of dDepth
   depth pixels. 
*/
static void ConvertLine(unsigned char *line, int line_len, XImage * dest)
{
  int i, value, mult=0;
  switch (dDepth)
    {
    case 32:
      mult = 0x01010101;
      break;
    case 24:
      mult = 0x000f0704;
      break;
    case 16:
      mult = 0x00000101;
      break;
    case 8:
      mult = 1;
      break;
    }
  for (i=0;i<line_len;i++)
    {
      value = line[i] * mult;
      XPutPixel(dest, i, 0, value); 
    }
  return;
}


/* draw a line of data in the window, copy it to a buffer area for
 *  refresh purposes (allocate the area if it's not already there) 
 *
 * It is assumed that the input is a line of 8bit values, to be mapped
 * to 256 colors in the default visual. 
 *
 * we assume a square image of identical lines 
 * this is actually a bug!!
 */

void ShowLine(void *line,int line_len,int position)
{ 
  char * image_buffer;
  XImage * image_line;

  /* I added some clipping! */
  if (line_len>rounded_length) line_len=rounded_length;
  if (position>m_h) return;

  /* alloc temp image area and XImage structure */
  DEBUG(printf("Showline - 2 pt %x len %d pos %d rlen %d  \n",
	       line, line_len, position, rounded_length));
  image_buffer = (char *)(calloc(sizeof(char)*sizeofPixel, rounded_length));

  DEBUG(printf("Showline - 3 tmpimage %x siz %x len %x \n",
	       image_buffer, sizeof(char)*sizeofPixel, rounded_length));

  image_line = XCreateImage(display,DefaultVisual(display, SCREEN),
		       dDepth,ZPixmap,0,
		       (char *)image_buffer,rounded_length,1,32,0); 

  DEBUG(printf("Showline - 4.1 image_line %x \n",image_line));

  /* convert the input data into the image */
  ConvertLine(line, line_len, image_line);
		   
  /* Put the line into the window */
  XPutImage(display,window,gc,image_line,0,0,0,position,line_len,1);
  XFlush(display);

  /* Save the line into our backing store */
  memcpy((m_image_buffer+(position*rounded_length*sizeofPixel)),
	 image_buffer,
	 line_len*sizeofPixel);
  /* destroy temporary image */
  XDestroyImage (image_line);
  DEBUG(printf("Showline - 9\n"));
}


/* manage some events coming from the window:  
 * refresh (Expose) events,
 * button press,
 * key presses: q,Q,c,C,r,R,m,M
 */
void HXI(int *px, int *py,int *dim,int *done)
{
  XEvent event;
  int something =0;
  static int next=0;
  const int clicks = 10; /* how many clicks before exiting anyway */

  *dim=1;
  *done=0;
  
  while ((XEventsQueued(display, QueuedAfterReading) > 0)|| something<clicks)
	{
	  XFlush(display);
	  XNextEvent(display, &event);
	  switch(event.type)
		{
		case ButtonPress:
		  {
			unsigned int button;
			XButtonPressedEvent * bpe = 
			  (XButtonPressedEvent *) &event;

			*px=bpe->x;
			*py=bpe->y;
			button=bpe->button;
			*dim = (button == Button1 ? 2 :
				button == Button2 ? 4 : 8);
			something++;
		  }
		  break;
		case Expose:
		  if (m_image_buffer!=NULL)
		    {
		      if (image == NULL)
			{
			  image = 
			    XCreateImage(display,
					 DefaultVisual(display,SCREEN),
					 dDepth,ZPixmap,0,
					 (char*) m_image_buffer,
					 rounded_length,m_h,32,0);
			  DEBUG(printf("image %x \n",image));
			}
		      XPutImage(display,window,gc,image,0,0,0,0,m_w,m_h);
		      XFlush(display);
		    }
		  break;
		case KeyPress:
		  {
		    XKeyEvent * kpe = (XKeyEvent *) &event;
		    KeySym ks = XLookupKeysym(kpe, 0);
			
		    switch (ks)
		      {
		      case 'q':
		      case 'Q':
			*done=1;
			something=clicks;
			break;
		      case 'r':
		      case 'R':
			RainbowColorMap(128);
			break;
		      case 'c':
		      case 'C':
			ChooseColorMap(next++); 
			break;
		      case 'M':
		      case 'm':
			ShowTM ();
			break;
		      default:
			printf("Keys:\n\n");
			printf(
			       "q) Quit !\n"
			       "c) change colormap (8bits display)\n"
			       "c + r) rainbow colormap (8bits display)\n");
			fflush(stdout);
			
		      }
		  }
		  break;
		}
	}
}


/* two functions to get the default fg/gb colours on the default screen*/
int Black(void){return(BlackPixelOfScreen(DefaultScreenOfDisplay(display)));}
int White(void){return(WhitePixelOfScreen(DefaultScreenOfDisplay(display)));}


/* show a short message on the window */
void ShowTM ()
{
  /* should clear the window */

  XGCValues gcvalues,tmp_val;
  int mask;
  char s1[]="Bacci Cantalupo Ravazzolo";
  char s2[]="Riaudo Pesciullesi";
  char s3[]="Aldinucci Coppola Torquati";
  
  mask=GCForeground|GCFunction;
  XGetGCValues(display,gc,mask,&tmp_val);
  gcvalues.foreground=1;
  gcvalues.function=GXcopy;
  XChangeGC(display,gc,mask,&gcvalues);
  XDrawString(display,window,gc,5,10,"QSW PISA are:",12);
  XDrawString(display,window,gc,10,25,s1,strlen(s1));
  XDrawString(display,window,gc,10,40,s2,strlen(s2));
  XDrawString(display,window,gc,10,55,s3,strlen(s3));
  /* Restore Graphic Context */
  XChangeGC(display,gc,mask,&tmp_val);

}

/* the following two functions change the colormap for 8-bit
   displays. They should check and do nothing on true color displays */
void ChooseColorMap(int which)
{

  Visual visual;
  XColor color;
  int i;

  if (dDepth!=8)return; /* only for 256 color display! */

  if (primo==0)
    {
      visual = *DefaultVisual(display, SCREEN);
      colormap= XCreateColormap(display, window, &visual, AllocAll); 
    }
  
  color.flags=DoRed | DoGreen | DoBlue;
  for (i=0; i< 256; i++)
    {
      color.pixel=i;
      color.red=0xffffL * ((long) (i+which) * 101 %256)/255L;
      color.green=0xffffL * ((long) (i+which) * 151 %256)/255L;
      color.blue=0xffffL * ((long) (i+which) * 171 %256)/255L;
      
      XStoreColor(display, colormap, &color);	  
    }
  
  if (primo==0)
    {
      XInstallColormap(display,colormap);
      XSetWindowColormap(display,window,colormap);
      primo=1;
    }
}

void RainbowColorMap(int n)
{
  int i, j;
  double d, e;
  XColor color;

  if (dDepth!=8)return; /* only for 256 color display! */

  if (primo!=0)
    {
      color.flags=DoRed | DoGreen | DoBlue;
      for (i = 1; i < n - 1; i++) {
	j = n - 1 - i;
	d = (d = cos((double)((j - n * 0.16) * (PI / n)))) < 0.0
	  ? 0.0 : d;
	color.blue = d * n;
	d = (d = cos((double)((j - n * 0.52) * (PI / n)))) < 0.0
	  ? 0.0 : d;
	color.green = d * n;
	d = (d = cos((double)((j - n * .83) * (PI / n)))) < 0.0
	  ? 0.0 : d;
	e = (e = cos((double)(j * (PI / n)))) < 0.0
	  ? 0.0 : e;
	color.red = d * n + e * (n / 2);
	color.pixel=i;
	XStoreColor(display, colormap, &color);
      }
      color.green=color.blue=color.red=color.pixel=i;
      XStoreColor(display, colormap, &color);
      color.green=color.blue=color.red=color.pixel=0;
      XStoreColor(display, colormap, &color);
    }
} 
