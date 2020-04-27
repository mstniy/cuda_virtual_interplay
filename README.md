A small library to make virtual calls and field accesses across CUDA boundaries easier.

It is not that hard to pass C++ classes over CUDA, especially with unified memory.
Unfortunately however, only the host can call the virtual methods of the objects created on the host and vice versa.
See [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-functions).
The case is similar with virtual inheritance: Only the host can access the member fields and methods of the virtual bases of the objects created on the host and vice versa.
This library lets you smoothly migrate such objects from the host to the device and back.

## How does it work?
If not asked to migrate virtual bases, it simply move constructs the objects on the host/device to fix their vtable.  
If asked to migrate virtual bases, the library constructs, for each most derived type, two objects, one on the host and one on the device. It assumes that all the positions where these objects differ are part of the vtable. During migrations, bytes that are determined to belong to vtables get fixed.

It requires that:
* Your GPU supports unified memory (everything above and including Kepler supports it)
* Memory layouts of objects are the same on both the host and the device (unlikely to be true on Windows, not tested)
* The classes to be migrated are move-constructible
* The classes to be migrated do not need to be destructed after being move constructed

It can also migrate classes with virtual bases. In this case it also requires that:

* The classes to be migrated are default-constructible
* The default constructor either always initializes data fields to the same value, or does not initialize them. So this won't work:


    class A
    {
        ...
    	int* x=new int;
    	...
    };
    
It would cause the library to mistake data bytes for vtable bytes and silently overwrite them during migrations. 

But this will:

    class A
    {
        ...
    	int* x=NULL; // Always initialized to the same value
        int y; // Not initialized
        ...
    };

Another disadvantage to migrating virtual bases is that it is slower, since it needs to check every byte to see if it belongs to the vtable or not.

## Usage

`#include "virtual_interplay.h"` to access the library.  
**Objects to be migrated must be allocated on unified memory.** This can be achieved by using `unified_unique_ptr` or `make_unified_unique`.
Use `__host__ __device__` to make sure your move (and default, if you want to migrate virtual bases) constructor is callable from both the host and the device, if they are not implicit or defaulted.  
Mark the virtual functions with `__device__` to let them be used from the device, or with `__host__ __device__` to let them be used on both sides.  

To migrate an array of objects to the device, do:

    ClassMigrator<Type>::toDevice(objs, num_objs);

To migrate them back to the host, use `toHost()`

If you need to access virtual bases across CUDA boundaries, use `toDeviceWithVirtualBases` and `toHostWithVirtualBases`.  

For a complete example, see `demo.cu`.
