#!/usr/bin/python
import os

executable = 'denoiser '
inputFile =  '/home/drocco/svn/ff-experimental-code/users/Maurizio/denoiser/test/test.mp4 '
configFile = '/home/drocco/svn/ff-experimental-code/users/Maurizio/denoiser/test/ff.conf'
perfFile = 'perf.csv'
perfFileFiltered = 'perf_filtered.csv'

print executable
print inputFile
print configFile

noises = range(10, 60, 20)
nframes = 320

#init perf file
fp = open(perfFile, "wb")
fp.write('# frame height, frame width, frames, noise %, farm workers, detector workers, denoiser workers, exec. time (s), throughput (fps), detector svc time (ms), denoiser svc time (ms)\n')
fp.close()

for noise in noises:

    # single worker 
    fo = open(configFile, "wb")
    fo.write('farm_workers 1\n')
    fo.write('detector_workers 1\n')
    fo.write('denoiser_workers 1\n') 
    fo.close()
    params = '-N1 -F' + str(nframes) + ' -n' + str(noise) + ' -v -c 50 -f '
    commandLine = executable +  params + inputFile +  configFile + ' 2>> ' + perfFile
    print commandLine
    os.system(commandLine)

    # multiple workers
    for fw in range(4, 33, 4):
        for detw in range(1, 2, 1):
            for denw in range(1, 2, 1):
                fo = open(configFile, "wb")
                fo.write('farm_workers ' + str(fw) + '\n')
                fo.write('detector_workers ' + str(detw) + '\n')
                fo.write('denoiser_workers '+ str(denw) + '\n') 
                fo.close()
                params = '-N1 -F' + str(nframes) +' -n' + str(noise) + ' -v -c 50 -f '
                commandLine = executable +  params + inputFile +  configFile + ' 2>> ' + perfFile
                print commandLine
                os.system(commandLine) 
