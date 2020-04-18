A small library to make virtual calls and field accesses across CUDA boundaries easier.

It is not that hard to pass C++ classes over CUDA, especially with unified memory.
Unfortunately however, only the host can call the virtual methods of the objects created on the host and vice versa.
See [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-functions).
The case is similar with virtual inheritance: Only the host can access the member fields and methods of the virtual bases of the objects created on the host and vice versa.
This library lets you smoothly migrate such objects from the host to the device and back.

## How does it work?
For each most derived type, the library constructs two objects, one on the host and one on the device. It assumes that all the positions where these objects differ are part of the vtable.
During migrations, bytes that are determined to belong to vtables are fixed.

It requires that:
* Your GPU supports unified memory (everything above and including Kepler supports it)
* Memory layouts of objects are the same on both the host and the device (unlikely to be true on Windows, not tested)
* The classes to be migrated are default- and move-constructible
* The classes to be migrated do not need to be destructed after being moved
* The default constructor always initializes data fields to the same value, or does not initialize them

So this won't work:

    class A
    {
        ...
    	int* x=new int;
    	...
    };

But this will:

    class A
    {
        ...
    	int* x=NULL; // Always initialized to the same value
        int y; // Not initialized
        ...
    };

The approach taken by the master branch does not have this restriction, if you do not need to access migrate virtual base classes.
One disadvantage of this method is that it can fail silently by mistaking data bytes for vtable bytes and overwriting them during migrations.
For a technique that fails loudly, see the master branch.

## Usage

`#include "virtual_interplay.h"` to access the library. The base class of the classes to be migrated shall inherit from `interplay_movable` and the most derived subclasses shall inherit from `implements_interplay_movable`.
Use `__host__ __device__` to make sure your move and default constructors are callable from both the host and the device.
Mark the virtual functions with `__device__` to let them be used from the device, or with `__host__ __device__` to let them be used on both sides.

To migrate an array of objects to the device, do:

    ClassMigrator<Base> migrator(objs);
    migrator.toDevice();


To migrate them back to the host, use `migrator.toHost()`

For a complete example, see `demo.cu`.
