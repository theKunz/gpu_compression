import getopt, sys, os, subprocess, signal, re, json, resource, time, socket
from datetime import datetime
from collections import namedtuple, defaultdict

timestamp = str(datetime.now()).replace(" ", "_")
workdir = "testdir_" + timestamp

def setup_working_directory():
    os.mkdir(workdir)
    os.chdir(workdir)
    os.mkdir('originals')
    os.mkdir('compressed')
    os.mkdir('decompressed')

def gen_matrices():
	os.chdir('../')
	for i in range(10, 101):
		for j in range(1, 101):
			cmdline = './genMatrix ' + str(i) + ' ' + str(i) + ' ' + str(j)
			proc = subprocess.Popen(cmdline , stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    		proc.wait()
    	print '\tGenerated ' + str(i) + 'size matrices'

	for i in range(10, 101):
		for j in range(1, 101):
			cmdline = 'mv matrix_' + str(i) + '_' + str(i)  + '_' + str(j) + ' ./' + workdir + '/originals'
			proc = subprocess.Popen(cmdline , stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
			proc.wait()
	os.chdir(workdir)

def compress_matrices():
	os.chdir('compressed')
	resultfile = open('compression_time_results.csv', 'a+')
	executable = 'a.out'
	for i in range(10, 101):
		for j in range(1, 101):
			cmdline = '../../../gpu/' + executable + ' -c ../originals/matrix_' + str(i) + '_' + str(i) + '_' + str(j)
			start_time = time.time()
			proc = subprocess.Popen(cmdline , stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
			proc.wait()	
			compression_time = time.time() - start_time
			resultfile.write(str(compression_time) + ',')
		resultfile.write("\n")
		print '\tCompressed ' + str(i) + 'size matrices'
	resultfile.close()

def decompress_matrices():
	os.chdir('../decompressed')
	resultfile = open('decompression_time_results.csv', 'a+')
	executable = 'a.out'
	for i in range(10, 101):
		for j in range(1, 101):
			cmdline = '../../../gpu/' + executable + ' -u ../compressed/matrix_' + str(i) + '_' + str(i) + '_' + str(j) + '.crs'
			start_time = time.time()
			proc = subprocess.Popen(cmdline , stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
			proc.wait()	
			compression_time = time.time() - start_time
			resultfile.write(str(compression_time) + ',')
		resultfile.write("\n")
		print '\tDecompressed ' + str(i) + 'size matrices'
	resultfile.close()

def main():
	print ''
	print '=' * 80
	print 'Beginning GPU Compression test harness'
	print ''
	print 'Creating working directory...'
	setup_working_directory()
	print 'Generating matrices...'
	gen_matrices()
	print 'Compressing matrices and recording time...'
	compress_matrices()
	print 'Decompressing matrices and recording time...'
	decompress_matrices()

	#add further tests here. Be sure to keep track of current directory

	print 'Test harness completed successfully'

        

main()