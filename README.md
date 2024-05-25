This is an annotaed implementation of `selective_scan_cuda` from Mamba. This is not runnable code, but a reference for what is happening in the CUDA kernel. I wrote this in Python for people who don't know C++. There are 
some slight differences between the Python and C++ code, but the general idea is the same.

Additionally, I wrote comments in the code to explain what is going on and why they chose to do some things. A basic understanding of what a CUDA kernel is would be helpful, though you can still get a general idea of what is going on without it.

Something Things to Note:
  - Pointers don't exist in Python, thus some of the logic for that has been removed.
  - In the C++ implementation we need to update the location of the pointer after every chunk, however, in the annotated Python version I do not do this. (So don't get confused when it looks like every chunk is iterating over the same data)
  - Templates don't exist in Python, thus I kinda get around this by having functions be classes. But don't be fooled, these would be functions in C++
  - This is more like pseudo-code, which is written in a pythonish language (The comments are where I try to explain what we are doing and why)
