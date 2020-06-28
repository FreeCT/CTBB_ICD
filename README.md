# FreeCT_ICD

**NOTE:  No further updates or changes will be made to this repository.  I am currently in the process of migrating the entirety of the FreeCT project into one project: https://github.com/FreeCT/FreeCT.   This should simplify building and installing in every single way, however in the mean time there are going to be some growing pains and the project will be non-functional.  If you have any questions or comments please reach out to johnmarianhoffman@gmail.com.**

## What is FreeCT_ICD?

FreeCT\_ICD is the model-based iterative reconstruction companion software to [FreeCT\_wFBP](https://github.com/FreeCT/FreeCT_wFBP).  It is free software (released under the GNU GPLv2) intended for use in research and education.  Modification and contributions are encouraged.

### Quick (Science) Facts:

FreeCT_ICD...

* Reconstructs third-generation, helical, diagnostic, multidector CT data (without gantry tilt)
* Uses **iterative coordinate descent** as an optimizer
* Employs a **stored system matrix (SSM)** 
* Employs a **rotating slice** geometry to keep the stored system matrix sizes small (J. Xu and B. M. W. Tsui, “Iterative image reconstruction in helical cone-beam x-ray CT using a stored system matrix approach.,” Phys. Med. Biol., vol. 57, no. 11, pp. 3477–97, 2012.)
* Uses a **modified Joseph's method** approach for the projector
* Supports **flying focal spots** (*Note: at present, only an in-plane flying focal spot is supported, however support for Z and Z+in-plane are under active development*)

### Quick (Computing) Facts:

* Is **low-dependency**.  You only need to have the BOOST (https://www.boost.org/), yaml-cpp (https://github.com/jbeder/yaml-cpp), and [FreeCT_Reader](https://github.com/FreeCT/FreeCT_Reader) libraries installed on your system.
* Uses OpenMP for acceleration (customizable number of cores)
* Reconstructs a clinical CT dataset in about 12 hours

## What it is not

### FreeCT_ICD is not what the manufacturers use

While our algorithms are based off of publications that are perhaps relevant to some of the current algorithms used in industry, they are not the algorithms used on clinical CT scanners and we make no claims to the similarity between our reconstructed images and what is arrived at clinically. In fact, moreso than FreeCT\_wFBP, FreeCT\_ICD employs a unique forward projector that has not been previously published.

Work has been done to objectively evaluate the quality of our reconstructions.  This can be found here:

*(insert link to technical note and/or whitepaper)* (**Coming soon**)

### FreeCT_ICD is not a library

There are many great reconstruction libraries out there (http://conrad.stanford.edu/, http://www.openrtk.org/ to name two), and perhaps one day FreeCT_ICD will be recast as a library.  Currently however, it is not a library, it is a program.

It is structured modularly, so that major subsections of the reconstruction process are easy to identify/customize/edit/etc. so there are library-like qualities to the project to make it easy to use.

FreeCT_ICD is designed to be compiled and run to reconstruct projection data from start to finish.

## Versions

The latest working version can be found on GitHub at https://github.com/FreeCT/FreeCT_ICD

Bleeding edge updates can be found at https://github.com/FreeCT/FreeCT_ICD/tree/develop

## Installation

*(to be added later)*

## Use

*(to be added later)*

## License

GNU GPLv2

*(more info to be added later)*

Copyright 2018 John Hoffman, Stefano Young, Frédéric Noo
