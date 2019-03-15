# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:34:39 2019

@author: adrian.a.hood
"""

from multiprocessing import Pool, TimeoutError,Process,Array,Value,Queue
import time
import os

class myob:
    def __init__(self,val):
        self.val= val
        self.cubevalue =self.cube(val)
    def cube(self,val):
        self.cube=val*val*val
        return self.cube

def f(x,sleeptime,data):
    time.sleep(sleeptime)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    data.val=(x*x)
    print(data.val)
    #return data

def f2(x):
    return x*x

def f3(q):
    keep=[]
    data1=myob(2)
    data2=myob(4)
    keep.append(data1)
    keep.append(data2)
    q.put([keep])

    
def getdata():
    keep=[]
    with Pool(4) as pool:
        data1=pool.map(f,2)
        data2=pool.map(f,10)
        data3=range(20)
        keep.append(data1)
        keep.append(data2)
        keep.append(data3)
        print(len(keep))
        print(keep)

def getdataMP(x):
    # This technique allows each process to call a method and have a returned
    # Value. The key is to pass in the variable that needs to be changed
    # It is akin to passing as reference.  The variable must be an object
    # The 1st example that worked used the array object.
    keep=[]
    data1 = Array('f',2)
    data2 = Array('i',10)
    p1=Process(target=f,args=[2,1,data1])
    p2=Process(target=f,args=[4,5,data2])
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    keep.append(data1[0])
    keep.append(data2[0])
    print(keep)
    xx=myob(x)
    return xx.cube()

def getdataMP2(x):
    # This technique allows each process to call a method and have a returned
    # Value. The key is to pass in the variable that needs to be changed
    # It is akin to passing as reference.  So far, it works if I used the
    # multiprocessing.Array object. However, if I pass a custom object, that
    # object only appears to have limited scope and is not changed.
    keep=[]
    data1 = myob(2)
    data2 = Array(myob,10)
    p1=Process(target=f,args=[2,1,data1])
    p2=Process(target=f,args=[4,1,data2])
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    keep.append(data1.val)
    keep.append(data2.val)
    #print(keep)
    x.put(keep)
    return x

def getdataMP3():
    p=Pool(4)
    keep = p.map(getdataMP,[5])
    print("The Value is: {}".format(keep))
    print("keep is Type: ".format(type(keep)))
    print(type(keep))
    
def getdataMP4():
    # This works pretty well. I had to modify the code by using q.put()
    # Ideally I should be able to run each pyramid as a new process, and wait 
    # all of them to finish using .join(), combine them in a list, and then
    #return using q.get()[0]

    q=Queue()
    p=Process(target=f3,args=(q,))
    p.start()
    p.join()
    #print(q.get())
    d1=q.get()[0] # queue returns a list ( of a list of object that I made). This simply
    # extracts the list wihin a list.
    print(d1[0].val)
    print(d1[1].val)
    print(d1[0].cubevalue)
    print(d1[1].cubevalue)
    #print(type(d1))
    #print(len(d1))
    #print(d1)
    #print(d1[0][0].val)
    #print(d1[0][1].val)
    #print(d1[0][0].cube)
      
    
def run_multiple_processes(NumOfProcessors):
    with Pool(NumOfProcessors) as pool:

        # print "[0, 1, 4,..., 81]"
        print(pool.map(f, range(10)))

        # print same numbers in arbitrary order
        for i in pool.imap_unordered(f, range(10)):
            print(i)

        # evaluate "f(20)" asynchronously
        res = pool.apply_async(f, (20,))      # runs in *only* one process
        print(res.get(timeout=1))             # prints "400"

        # evaluate "os.getpid()" asynchronously
        res = pool.apply_async(os.getpid, ()) # runs in *only* one process
        print(res.get(timeout=1))             # prints the PID of that process

        # launching multiple evaluations asynchronously *may* use more processes
        multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
        print([res.get(timeout=1) for res in multiple_results])

        # make a single worker sleep for 10 secs
        res = pool.apply_async(time.sleep, (10,))
        try:
            print(res.get(timeout=1))
        except TimeoutError:
            print("We lacked patience and got a multiprocessing.TimeoutError")

        print("For the moment, the pool remains available for more work")

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")
        
if __name__ == '__main__':
    # start 4 worker processes
    #run_multiple_processes(4)
    #getdata()
    #getdataMP()
    #getdataMP2()
    #getdataMP3()
    getdataMP4()