#!/usr/bin/python
import os

def drange(start, stop, step):
    while start < stop:
            yield start
            start += step

print 'noise 10 - seq'
for k in drange(1, 3, 1):
	os.system('./spd_ff_time /home/peretti/fastflow-2.0.1/examples/spd-denoiser/cuda/tests/img_4096x4096_noise10.bmp -f -c100 -p1') 

for p in drange(4, 33, 4):
	print p
	for k in drange(1, 5, 1):
		os.system(' ./spd_ff_time /home/peretti/fastflow-2.0.1/examples/spd-denoiser/cuda/tests/img_4096x4096_noise10.bmp -f -c100 -p' +  str(p)) 

print 'noise 50 - seq'
for k in drange(1, 3, 1):
	os.system('./spd_ff_time /home/peretti/fastflow-2.0.1/examples/spd-denoiser/cuda/tests/img_4096x4096_noise50.bmp -f -c50 -p1') 

for p in drange(4, 33, 4):
        print p
        for k in drange(1, 5, 1):
                os.system(' ./spd_ff_time /home/peretti/fastflow-2.0.1/examples/spd-denoiser/cuda/tests/img_4096x4096_noise50.bmp -f -c50 -p' +  str(p)) 

print 'noise 90 - seq'
for k in drange(1, 3, 1):
	os.system('./spd_ff_time /home/peretti/fastflow-2.0.1/examples/spd-denoiser/cuda/tests/img_4096x4096_noise90.bmp -p1') 

for p in drange(4, 33, 4):
        print p
        for k in drange(1, 5, 1):
                os.system(' ./spd_ff_time /home/peretti/fastflow-2.0.1/examples/spd-denoiser/cuda/tests/img_4096x4096_noise90.bmp -p' +  str(p)) 
