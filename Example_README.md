# Numba Examples for Cuda

This repository contains examples of Cuda code using Numba. Each file is dedicated to demonstrating a specific feature or tool. 

---

## simpleKernel.py 

Run a kernel with a 1D (`kernel1D`) or a 2D Grid (`kernel2D`). The kernel simply prints a message for threads with local ID (0,0). The thread blocks and grids have fixed values and the global thread ID is computed with `cuda.grid`.
```python
@cuda.jit
def kernel1D() :
    local_id = cuda.threadIdx.x
    global_id = cuda.grid(1)
    if local_id==0:
        print("local_id",local_id, "global_id", global_id)
```

---

## memoryTransfer.py 

Generates a 1D array on the host and transfer it to the device(`cuda_to_device`), before executing a kernel. The kernel simply fills the array with each thread's global id. 
```python
@cuda.jit
def writeGlobalID(array) :
    local_id = cuda.threadIdx.x
    global_id = cuda.grid(1)
    array[global_id]= global_id
```
The array is the copied back on the host (`copy_to_host`). The kernel also check that a thread has some data to process. 
```python
    if global_id < array.shape[0]:
        array[global_id]= global_id
```
The size of the Grid is computed if the `size` of the array is specified 
```python
    threadsPerBlock=16
    blocksPerGrid=math.ceil(size/threadsPerBlock)
```

--- 
## deviceFunction.py 

Demonstrates how to write a function which can only be called from a kernel using `@cuda.jit(device=True)`

```python
@cuda.jit(device=True)
def deviceFunction(tab, index):
    tab[index] = index

@cuda.jit
def kernel1D(tab) :
    [...]
    if global_id< tab.shape[0]:
        deviceFunction(tab, global_id)    
```

---

## typesCost.py

This code execute a kernel on numpy array with different scalar types (`uint8, uint16, uint32, float16, float32, float64)`. The memory size required on the device is estimated. The kernel performs some numerical computation on each value of the array. 
```python
        array[global_id]= array[global_id]*array[global_id]/2
```
The program prints the estimated size for each kernel. 

---

## bench.py

Using `timeit` we mesure the execution time of the kernel (`runTypeFloat` from `typesCost.py`). The kernel is run multiple times and each measurement is added to an array. 
```python
       start = timer()
       typesCost.runTypeFloat(blocksPerGrid,threadsPerBlock,size,type)
       cuda.synchronize()
       dt = timer() - start
       result[i]=dt
```
The first value is discarded to avoid caching and JIT interference and the average is printed. 
```python
    print("Average :", threadsPerBlock, np.average(result[1:]))
```

---

## bench.py

This program demonstrates how to allocate shared memory (`cuda.shared.array`) on the device and copy data from global memory (array `toShare` as parameter). 

```python
    shared_filter=cuda.shared.array(THREAD_BLOCK, dtype=np.int32)
    #Each thread fill one element of the array
    shared_filter[cuda.threadIdx.x] = toShare[cuda.threadIdx.x]
```
The code is not optimized and doesn't take into account bank conflicts or coalescing. 
The kernel uses this shared array to perform some computation, avoiding access to global memory. 
```python
    if global_id<array.shape[0] :
        tmp = array[global_id]
        for i in range(0, THREAD_BLOCK):
            tmp+=toShare[i]
        array[global_id] = tmp
```
