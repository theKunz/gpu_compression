#!/usr/bin/python
import getopt, sys, os, subprocess, signal, re, json, resource, time, socket
from datetime import datetime
from collections import namedtuple, defaultdict

timestamp = str(datetime.now()).replace(" ", "_")
workdir = "testdir_" + timestamp


__compression_test = namedtuple('compression_test',
                             ['name',           # name of the test (quicksort, fib, etc)
                              'file',        # command line to execute
                              'description'])

__compression_lib = namedtuple('compression_lib',
                             ['dir',
                              'executable',
                              'name', 
                              'compression_switch',           # name of the test (quicksort, fib, etc)
                              'decompression_switch',        # command line to execute
                              'input_switch',
                              'output_switch'])

def compression_test(name, file, description):
    return __compression_test(name, file, description)

def compression_lib(dir, executable, name, compression_switch, decompression_switch, input_switch, output_switch):
    return __compression_lib(dir, executable, name, compression_switch, decompression_switch, input_switch, output_switch)

libraries = [
    compression_lib(
        dir= "cuda_compression",
        executable="cuda_compression/main",
        name="lzssGPU",
        compression_switch="",
        decompression_switch="-d 1",
        input_switch="-i",
        output_switch="-o",
    ),
    compression_lib(
        dir= "lzssCPU",
        executable="lzssCPU/sample",
        name="lzssCPU",
        compression_switch="-c",
        decompression_switch="-d",
        input_switch="-i",
        output_switch="-o",
    ),
    compression_lib(
        dir= "bcg"
        executable= "a.out"
        name= "bcg"
        compression_switch="-c"
        decompression_switch="-u"
        input_switch="-i"
        output_switch="-o"
    ),
]
tests = [
    compression_test(
        name="mobydick",
        file="mobydick.txt",
        description="Melville's Moby-Dick, based on the Hendricks House edition",
    ),
    compression_test(
        name="bible",
        file="bible10.txt",
        description="The King James Bible: Second Version, 10th Edition",
    ),
]


def setup_working_directory():
    os.mkdir(workdir)
    os.chdir(workdir)

def get_throughput(size, time):
    return size/time

def get_compression_ratio(original_size, compressed_size):
    return compressed_size/original_size

def run_single_test(library, test):
    original_file = '../../data/' + test.file
    compressed_file = test.file + '.lzss'
    decompressed_file = test.file + '.txt'
    original_size = os.path.getsize(original_file)*0.000001

    #Compresssion
    cmdline = '../../' + library.executable + ' ' + library.compression_switch + ' ' + library.input_switch + ' ' + original_file + ' ' + library.output_switch + ' '+ compressed_file
    start_time = time.time()
    proc = subprocess.Popen(cmdline , stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, shell=True)
    proc.wait()
    compression_time = time.time() - start_time
    compressed_size = os.path.getsize(compressed_file)*0.000001
    compression_throughput = get_throughput(original_size, compression_time)
    compression_ratio = get_compression_ratio(original_size, compressed_size)

    #Decompression
    cmdline = '../../' + library.executable + ' ' + library.decompression_switch + ' ' + library.input_switch + ' ' + original_file + ' ' + library.output_switch + ' '+ compressed_file
    start_time = time.time()
    proc = subprocess.Popen(cmdline , stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, shell=True)
    proc.wait()
    decompression_time = time.time() - start_time
    decompression_throughput = get_throughput(original_size, decompression_time)

    print '[' + test.file + ': ' +'%.3f' % original_size+ 'MB' + ' | '+ compressed_file + ': ' +'%.3f' % compressed_size + 'MB' + ']'
    print '-' * 80
    print 'Compresssion:'
    print '-Time: ' + '%.3f' % compression_time + " seconds"
    print '-Throughput: ' + '%.3f' % compression_throughput + " MB/s"
    print '-Ratio: ' + '%.3f' % compression_ratio + " %"
    print 'Decompresssion:'
    print '-Time: ' + '%.3f' % decompression_time + " seconds"
    print '-Throughput: ' + '%.3f' % decompression_throughput + " MB/s"
    #stdout, stderr = proc.communicate()
    #print "time :" + str(stop_time)
    #print cmdline
    #print 'Program output:'
    #print stdout
    #print 'StdErr output:'
    #print stderr

def build(library):
        cmdline = 'cd ../' + library.dir + '; make; cd ../'
        proc = subprocess.Popen(cmdline , stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, shell=True)
        proc.wait()
        #stdout, stderr = proc.communicate()
        #print stdout
        #print stderr


def main():
    setup_working_directory()
    for count, library in enumerate(libraries):
        print ''
        print '=' * 80
        print 'Starting Test Sequence #' + str(count + 1)
        print 'Building'
        build(library)
        os.mkdir(library.name)
        os.chdir(library.name)
        print 'Running compression library: ' + library.name
        print '=' * 80
        for test in tests:
            print ''
            print 'Starting test: ' + test.description
            print '=' * 80
            run_single_test(library, test)
            sys.stdout.flush()
        os.chdir("../")

main()
