import sys, re

perfFile = sys.argv[1]
perfFileFiltered = "FILTERED_" + perfFile

print "output file: " + perfFileFiltered

fp = open(perfFile, "r")
fo = open(perfFileFiltered, "wb")
lines = fp.readlines()
for l in lines:
    if re.search("^\[.*$", l) == None:
           fo.write(l)
fp.close()
