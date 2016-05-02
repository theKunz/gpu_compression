import getopt, sys, os, subprocess, signal, re, json, resource, time, socket
from datetime import datetime
from collections import namedtuple, defaultdict

timestamp = str(datetime.now()).replace(" ", "_")
workdir = "testdir_" + timestamp + "_steps"

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
			cmdline = '../testfiles/genMatrix ' + str(i) + ' ' + str(i) + ' ' + str(j)
			proc = subprocess.Popen(cmdline , stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    		proc.wait()
    	print '\tGenerated ' + str(i) + ' size matrices'

	for i in range(10, 101):
		for j in range(1, 101):
			cmdline = 'mv matrix_' + str(i) + '_' + str(i)  + '_' + str(j) + ' ./' + workdir + '/originals'
			proc = subprocess.Popen(cmdline , stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
			proc.wait()

def compress_matrices():
	executable = 'sections.out'
	for i in range(10, 51):
		for j in range(1, 101):
			print 'Doing ./' + executable + ' -c ' + workdir + '/originals/matrix_' + str(i) + '_' + str(i) + '_' + str(j)
			cmdline = './' + executable + ' -c ' + workdir + '/originals/matrix_' + str(i) + '_' + str(i) + '_' + str(j)
			proc = subprocess.Popen(cmdline , stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
			proc.wait()	

			cmdline = 'mv '+ workdir + '/originals/matrix_' + str(i) + '_' + str(i) + '_' + str(j) + '.crs ' + workdir + '/compressed/'
			proc = subprocess.Popen(cmdline , stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
			proc.wait()	
		print '\tCompressed ' + str(i) + ' size matrices'

def decompress_matrices():
	os.chdir('../decompressed')
	executable = 'sections.out'
	for i in range(10, 51):
		for j in range(1, 101):
			cmdline = './' + executable + ' -c ' + workdir + '/originals/matrix_' + str(i) + '_' + str(i) + '_' + str(j)
			proc = subprocess.Popen(cmdline , stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
			proc.wait()	

			cmdline = 'mv '+ workdir + '/originals/matrix_' + str(i) + '_' + str(i) + '_' + str(j) + '.crs ' + workdir + '/decompressed/'
			proc = subprocess.Popen(cmdline , stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
			proc.wait()	
		print '\tDecompressed ' + str(i) + ' size matrices'

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
