import sys
import code

if len(sys.argv) != 4:
	print("Usage: python convert.py doc tag output")
	exit(-1)



fout = open(sys.argv[3], "w")
ftag = open(sys.argv[2], "r")
with open(sys.argv[1], "r") as fdoc:
	for line in fdoc:
		line = line.strip().split(" ")
		for w in line:
			tag = ftag.readline().strip()
			while tag == "":
				tag = ftag.readline().strip()
			tag = tag.split()
			if tag[0] == w:
				fout.write(tag[1] + " ")
			else:
				print("wrong!!!" + tag[0] + " " + w)
		fout.write("\n")
fout.close()
ftag.close()
