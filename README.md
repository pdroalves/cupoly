# cuPoly

[University of Campinas](http://www.unicamp.br), [Institute of Computing](http://www.ic.unicamp.br), Brazil.

Laboratory of Security and Cryptography - [LASCA](http://www.lasca.ic.unicamp.br),<br>
Multidisciplinary High Performance Computing Laboratory - [LMCAD](http://www.lmcad.ic.unicamp.br). <br>

Author: [Pedro G. M. R. Alves](http://www.iampedro.com), Ph.D. candidate @ IC-UNICAMP,<br/>

## About

cuPoly is a cyclotomic polynomial arithmetic module originally extracted from [cuYASHE](https://github.com/cuyashe-library/cuyashe). This library employs the CUDA platform and some algebraic techniques (like RNS, DGT, and optimizations on polynomial and modular reduction)  to obtain significant performance improvements. 

## Goal

cuPoly is an ongoing project and we hope to increase its performance and security in the course of time. Our focus is to provide:

 * Exceptional performance on modern GPGPUs.
 * A simple API, easy to use and very transparent.
 * Easily maintainable code. Easy to fix bugs and easy to scale.
 * The tools required for implementations of cryptographic schemes based on RLWE.


## Installation

Note that `stable` is generally a work in progress, and you probably want to use a tagged release version.

### Dependencies
cuPoly was tested in a Linux environment with the following packages:

| Package | Version |
| ------ | ------ |
| g++ | 8.4.0 |
| CUDA | 11.0 |
| cmake | 3.13.3 |
| googletest | v1.10.0 |
| [cxxopts](https://github.com/jarro2783/cxxopts/) | v2.2.0 | 
| [NTL](https://www.shoup.net/ntl/) | 11.3.2 |
| [gmp](https://gmplib.org/) | 6.1.2 |

### Compiling

The typical workflow for building cuPoly is:

1) Download and unzip the most recent commit or tagged release. The "stable" branch shall store the most recent version that was approved on all relevant tests. In the "unstable" branch you will ideas we are working on and may break something. Other branches are feature-related and may be merged some day.
2) Create cupoly/build.
3) Change to cupoly/build and run 
```
$ cmake ..
$ make
$ sudo make install
```
cmake will verify the environment to assert that all required packages are installed.

### Linking

Versions of cuPoly targetting BFV and CKKS are available. Some memory optimizations are done to fit all necessary precomputed data in the GPU constant memory, and because of that you must select which one you will need. For the BFV version, link against ``cupolybfv``.

### Tests and demos

cuPoly contains binaries for testing and benchmarking. Different tests, targetting BFV and CKKS, are offered in ``cupoly_bfv_test`` and ``cupoly_ckks_test``. They are built over googletest, thus you may apply its filters to select tests of interest. For instance,

```
cupoly_bfv_test --gtest_filter=*TestArithmetic*
```

runs tests for the basic polynomial arithmetic for all available cyclotomic ring settings.

There are demos written to improve the learning curve of the library. To compile them, run ``make demos``.

### Known-Issues

Known-issues at cuPoly, as bugs, may exist in a state that requires time for debugging without disturbing the most important use cases of the library. Those can be find at the [known-issues page](known-issues.md).

## How to use?


There is an embryonic version of a documentation made with doxygen. The most up-to-date source to understand how to use SPOG and cuPoly is by looking at the demos provided and the test suite.

If you wish to build the documentation, run cmake with ``BUILD_DOC`` flag turned on (it is off by default), and then compile. So, from the root directory,

```
mkdir build
cmake -DBUILD_DOC=ON ..
make doc
```

Remember to install doxygen and graphviz before that. The documentation will found at ``doc/html/index.html``.

## Citing
If you use SPOG/cuPoly, please cite using the template below:

```
@misc{cryptoeprint:2020:861,
    author = {Pedro Geraldo M. R. Alves and Jheyne N. Ortiz and Diego F. Aranha},
    title = {Faster Homomorphic Encryption over GPGPUs via hierarchical DGT},
    howpublished = {Cryptology ePrint Archive, Report 2020/861},
    year = {2020},
    note = {\url{https://eprint.iacr.org/2020/861}},
}
```

## Disclaimer
cuPoly is at most alpha-quality software. Implementations may not be correct or secure. Moreover, it was not tested with parameters different from those in the test set. Use at your own risk.

## Licensing

cuPoly is released under GPLv3.

**Privacy Warning:** This site tracks visitor information.
