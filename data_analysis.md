### Data analysis for particle physics

This page is a tutorial on data analysis for particle ("high-energy") physics. It is a work-in-progress and thus may experience unexpected reorganization. 

It is forseen that this tutorial will cover programming in Python; libraries for data science such as numpy, pandas, matplotlib, numba, pytorch, etc.; and some probability and statistics. All topics will be covered with applications to particle physics. Most of this content does not yet exist.

### Programming in Python

## Getting started

You will need to install python, ideally via a package manager such as [miniforge](https://github.com/conda-forge/miniforge). If you are using Windows, install Windows Subsystem for Linux, after which the instructions will be the same. Open a shell ("terminal" on MAC) and install miniforge with:

```sh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
Once you have installed miniforge, you can create a python environment, which we will call `hep_default`:

```sh
conda create --name hep_default python=3.12
conda install conda-forge::coffea 
conda install jupyterlab
```

After creation, you can activate this environment on subsequent logins by running:

```sh
conda activate hep_default
```

To start your Jupyter server, run 

```sh
jupyter lab
```

This starts jupyter, and you should see many log lines appear in your shell. One of these lines will have a URL like `http://localhost:8888/lab?token=...`, which you can use to connect to the server in your preferred web browser. Once, connected, you can create a new notebook, or open an existing notebook. You are now ready to begin programming!

## Expressions and variables in python

A good introductory computer science course text in python can be found [here](https://www.composingprograms.com/). 

The first thing you can try is running code to do numeric calculations such as

```py
5+4**2
```

If you are using Jupyter, simply type this into a cell, then press shift+enter to run the cell. We'll cover functions in more detail later, but you can get access to some familiar mathematical functions by running

```py
import math
```

after which you can evaluate expressions like

```py
5.0*math.sqrt(7.0)-math.log(2.0)
```

The symbols `+`, `-`, `*`, `/`, and `**` are used to represent addition, subtraction, multiplication, division, and exponentiation respectively. You can see what functions are available from the math library using the [documentation page](https://docs.python.org/3/library/math.html).

You can also store values in variables using the `=` operator, for example:

```py
z_mass = 91.1876
```

This line stores the value `91.1876` in the computer memory at a location that we have named `z_mass`. We can then retrieve this information later

```py
math.sqrt(2)*z_mass
```

A common simple use of programming in high-energy physics is serve as a calculator. The exercise below 

# Exercise 1

In particle physics, we often use "natural units" where the speed of light c and the reduced Planck constant &#295; are equal to 1. To have c=1 when c is usually `3.00e+8` meters per second, we measure distance and time in the "same" units. So, we can define a new unit of *time* called meters such that 1 second is equal to about `3.00e+8` meters of time since then c is equal to `3.00e+8/3.00e+8=1`. Similarly, to set &#295;=1 when it is usually `1.054e-34` joule seconds, we measure time and energy (and distance and momentum) in inverse units. So, one inverse joule of *time* is equal to about `9.49e+33` seconds since then &#295; is `1.054e-34*9.49e+33=1`. To check your understanding, consider how far is 1 second of distance? How much is 1 inverse second of energy? 1 inverse joule of distance?

Given that the reduced planck constant &#295; is approximatley equal to `6.582e-16` eV s, use python to calculate the mean lifetime of the following particles in seconds given their measured decay rate ("width") in units of energy from the [particle data book](https://pdg.lbl.gov/). Note the mean lifetime is simply the inverse of the decay rate `mean_lifetime=1.0/decay_rate`.

 * D\*(2010) meson: 83.4 keV
 * J/psi meson: 92.6 keV
 * W boson: 2.1 GeV
 * Z boson: 2.5 GeV
 * Higgs boson: 3.7 MeV
 * top quark: 1.4 GeV

With these mean lifetimes, how far (in meters) would each of these particles travel on average if they were traveling at the speed of light, `3.00e+8` m/s.

## Functions and control in python
