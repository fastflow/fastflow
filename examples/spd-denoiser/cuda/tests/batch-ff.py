#!/usr/bin/python
import os

def drange(start, stop, step):
    while start < stop:
            yield start
            start += step

print 'noise 10 - seq'
for k in drange(1, 3, 1):
	os.system('./spd_ff_time /home/drocco/svn/11_denoiser/testimages/classic/lena/lena512_noise10.bmp -c 300 -f -p1') 

for p in drange(4, 33, 4):
	print p
	for k in drange(1, 3, 1):
		os.system(' ./spd_ff_time /home/drocco/svn/11_denoiser/testimages/classic/lena/lena512_noise10.bmp -c 300 -f -p' +  str(p)) 

print 'noise 50 - seq'
for k in drange(1, 3, 1):
	os.system('./spd_ff_time /home/drocco/svn/11_denoiser/testimages/classic/lena/lena512_noise50.bmp -c 150 -f -p1') 

for p in drange(4, 33, 4):
        print p
        for k in drange(1, 3, 1):
                os.system(' ./spd_ff_time /home/drocco/svn/11_denoiser/testimages/classic/lena/lena512_noise50.bmp -c 150 -f -p' +  str(p)) 

print 'noise 90 - seq'
for k in drange(1, 3, 1):
	os.system('./spd_ff_time /home/drocco/svn/11_denoiser/testimages/classic/lena/lena512_noise90.bmp -c 100 -f -p1') 

for p in drange(4, 33, 4):
        print p
        for k in drange(1, 3, 1):
                os.system(' ./spd_ff_time /home/drocco/svn/11_denoiser/testimages/classic/lena/lena512_noise90.bmp -c 100 -f -p' +  str(p)) 
