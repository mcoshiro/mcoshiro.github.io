# Data analysis for particle physics

This page is a tutorial on data analysis for particle ("high-energy") physics. It is a work-in-progress and thus may experience unexpected reorganization. 

It is forseen that this tutorial will cover programming in Python; libraries for data science such as numpy, pandas, matplotlib, numba, pytorch, etc.; and some probability and statistics. All topics will be covered with applications to particle physics. Most of this content does not yet exist.

# Programming in Python

## Getting started and documentation in science

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

In scientific research, it is very important that you save all of your work in an complete and organized fashion since reproducibility is the backbone of science. Historically, scientists used lab notebooks for this purpose. This is why we suggest using Jupyter notebook: it can provide a record of the analyses you have performed. If you prefer programming using another methods such as python scripts, make sure they are saved in an organized fashion such as a git repository.

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

We can work with many different **types** of data. Some of the common basic types in python are:

| Type   | Description                               | Examples                |
|--------|-------------------------------------------|-------------------------|
| int    | Integer                                   | `4`, `-12`, `255`       |
| float  | "Floating point" approx. to a real number | `3.1416`, `6.022e23`    |
| string | Sequence of characters                    | `'hello'`, `"__main__"` |
| bool   | True or False value                       | `True`, `False`         |
| list   | Ordered collection of data                | `[3.8, -2.2, 4.0, 6.3]` |
| dict   | Map from "keys" to "values"               | `{1: 'CMS', 2: 'LHCb'}` |

A common simple use of programming in high-energy physics is serve as a calculator. The exercise below 

### Exercise 1.1: Natural units

In particle physics, we often use "natural units" where the speed of light c and the reduced Planck constant $\hbar$ are equal to 1. To have c=1 when c is usually about $3.0\times 10^8$ meters per second, we measure distance and time in the "same" units. For example, we can define a new unit of *time* called meters such that 1 second is equal to about $3.0\times 10^8$ meters of time and thus 1 meter of time is about $3.33\times 10^{-9}$ seconds. Then, c is equal to $\frac{3.0\times 10^8 \mathrm{ m}}{3.0\times 10^8 \mathrm{ m}}=1$. Similarly, to set $\hbar$ to 1 when it is usually $1.054\times 10^{-34}$ joule seconds, we measure time and energy (and distance and momentum) in inverse units. So, we define an inverse joule of *time* to be about $1.054\times 10^{-34}$ seconds, which means 1 second is about $9.49\times 10^{33}$ inverse joules. Then $\hbar$ is $(1.054\times 10^{-34} \mathrm{ J})(9.49\times 10^{33} \mathrm{ J}^{-1})=1$. To check your understanding, consider how far is 1 second of distance? How much is 1 inverse second of energy? 1 inverse joule of distance (hint: this requires $\hbar=c=1$)?

Given that the reduced planck constant is approximatley equal to $\hbar\approx 6.582\times 10^{-16}$ eV s, use python to calculate the mean lifetime of the following particles in seconds given their measured decay rate ("width") in units of energy from the [particle data book](https://pdg.lbl.gov/). Note the mean lifetime is simply the inverse of the decay rate `mean_lifetime=1.0/decay_rate`.

 * D\*(2010) meson: 83.4 keV
 * J/psi meson: 92.6 keV
 * W boson: 2.1 GeV
 * Z boson: 2.5 GeV
 * Higgs boson: 3.7 MeV
 * top quark: 1.4 GeV

With these mean lifetimes, how far (in meters) would each of these particles travel on average if they were traveling at the speed of light, $3.00\times10^{8} \mathrm{m}/\mathrm{s}$. Given that particle detectors are generally centimeters away from the point of particle creation, do we expect any of these particles to pass through a detector?

### Exercise 1.2: How far do cosmic rays travel?

In special relativity, many nonrelativistic calculations are corrected by the Lorentz factor $\gamma = \frac{1}{\sqrt{1-v^2/c^2}}=\frac{E}{m}$. For example, the magnitude of the momentum of a particle is $\vec{p}=\gamma m \vec{v}$ and the time measured by an outside observer after a time $t$ has elapsed in the frame of the particle will be $\gamma t$.

Muons are produced by cosmic rays hitting the earth's atmosphere. The mass of a muon is about $106 \mathrm{MeV}/c^2$ and its rest frame mean lifetime is $2.2\times 10^{-6}$ s. Consider muons with momentum 200 MeV/c, 1 GeV/c, and 5 GeV/c. Using the relation $E^2=m^2c^4+\vert\vec{p}\vert^2c^{2}$ in natural units, compute the gamma factor for each. The gamma factor is dimensionless and thus identical in all unit systems. Calculate the velocity in m/s for each of these muons using $1 \mathrm{ eV}=1.602\times 10^{-19} \mathrm{ J}$. Accounting for time dilation, the average distance travelled is $\gamma\tau v$. Calculate this for each muon. If a cosmic ray shower muon is typically produced 20 km above sea level, which of these might you expect to reach sea level?

## Functions and control in python

You can define a function in python using a format similar to the following:

```py
def hypotenuse(a, b):
  c = math.sqrt(a**2+b**2)
  return c
```

The keyword `def` comes first, followed by the function name (in this example `add`), the arguments (inputs) to the functions in parentheses, then a semicolon. The body of the function defining what it does it then specified by a series of indented lines. The output of the function can be specified by a line with the `return` keyword. One can then call this function

```py
hypotenuse(3.0, 4.0)
```

To have code that executes in certain cases, one can use `if` statements. For example:

```py
def absolute_value(x):
  if x > 0:
    return x
  elif x == 0:
    return 0
  else:
    return -1.0*x
```

If the first `if` condition (`x > 0`) is met, the subsequent indented code will be run. If it is not met, the code will check the next `elif` condition (`x == 0`) and run the subsequent code if that condition is met. This can continue for any number of `elif`s. Finally, if no conditions are met, the `else` block will run. Note that `elif` and `else` are not required--- you can just do nothing if the original condition(s) are not met.

To have code that executes repeatedly, one can use loops. For example:

```py
def factorial(x):
  x_factorial = 0
  while x > 0:
    x_factorial *= x
    x -= 1
  return x_factorial
```

In this case, the indented code block after the `while` will continue to run until the condition (`x > 0`) is no longer true. Another form of loop is the `for` loop:

```py
def count_electrons():
  # electrons have particle ID 11 or -11 (antielectrons)
  nelectrons = 0
  for id in id_list:
    if id==11:
      nelectrons += 1
  return nelectrons

id_list = [3, 21, -4, 11, -15, 23, 12, -2, 2, 11, 25, 21]
count_electrons(id_list)
```

The `for` loop allows one to iterate over the elements of a collection such as a list.

### Exercise 1.3: Newtonian simulation

Let us try writing a simple simulation of classical physics. Suppose we have a planar two-body system interacting through gravity such as the earth and the moon. We can specify the position $\vec{x}$ and velocity $\vec{v}$ vectors of each body in the x-y plane with a pair of numbers. We will use a python list for this:

```py
pos_body1 = [0.0, 0.0]
pos_body2 = [0.0, -10.0]
vel_body1 = [0.0, 0.0]
vel_body2 = [10.0, 0.0]
```

Here, we've just picked arbitrary values for the initial velocity and position. We'll also need some functions to implement the familiar vector operations of addition, subtraction, scalar multiplication, and norm/magnitude.

```py
def vector_add(a, b):
  return [a[0]+b[0], a[1]+b[1]]

def vector_sub(a, b):
  return [a[0]-b[0], a[1]-b[1]]

def scalar_multiply(scalar, vect):
  return [scalar*vect[0], scalar*vect[1]]

def vector_norm(x):
  return (x[0]**2+x[1]**2)**0.5
```

We know that $\vec{v}=\frac{d\vec{x}}{dt}$ and $\vec{a}=\frac{d\vec{v}}{dt}$. So, if we pick a time $\Delta t$ that is short enough that $\vec{v}$ is roughly constant, then the new position after $\Delta t$ time is $\vec{x}\sb{\mathrm{new}} = \vec{x}+\frac{d\vec{x}}{dt}\Delta t=\vec{x}+\vec{v}\Delta t$ and similarly $\vec{v}\sb{\mathrm{new}}=\vec{v}+\vec{a}\Delta t$. We can then just repeat this over and over to get the position and velocity at some later time.

The last ingrediant is calculating the accereration which depends on the force and mass $\vec{a}=\frac{F}/m$. Newtonian gravity says that the force on body 1 is $\vec{F}=\frac{Gm\sb{1}m\sb{2}}{\vert\vec{r}\vert^3}\vec{r}$ where $\vec{r}$ is the the displacement between the bodies $\vec{r}=\vec{x}\sb{2}-\vec{x}\sb{1}$, and an analogous expression holds for body 2. We have to decide on masses $m\sb{1}$ and $m\sb{2}$ as well as a system of units, which will determine the constant $G$. In code, our full simulation is then

```py
# functions for working with vectors implemented as lists
def vector_add(a, b):
  return [a[0]+b[0], a[1]+b[1]]

def vector_sub(a, b):
  return [a[0]-b[0], a[1]-b[1]]

def scalar_multiply(scalar, vect):
  return [scalar*vect[0], scalar*vect[1]]

def vector_norm(x):
  return (x[0]**2+x[1]**2)**0.5

# set initial conditions
pos_body1 = [0.0, 0.0]
pos_body2 = [0.0, -10.0]
vel_body1 = [0.0, 0.0]
vel_body2 = [1.0, 0.0]
mass_body1 = 10
mass_body2 = 0.01
G = 1.0

# set total time to simulate and calculate number of time steps 
delta_t = 0.1
total_time = 100
n_steps = int(total_time/delta_t)
istep = 0

while istep < n_steps:

  # calculate force/acceleration on body 1
  disp21 = vector_sub(pos_body2, pos_body1)
  force_on_body1 = scalar_multiply(
      G*mass_body2*mass_body1/vector_norm(disp21)**3, 
      disp21)
  acc_body1 = scalar_multiply(1.0/mass_body1, force_on_body1)

  # same for body 2
  disp12 = vector_sub(pos_body1, pos_body2)
  force_on_body2 = scalar_multiply(
      G*mass_body1*mass_body2/vector_norm(disp12)**3, 
      disp12)
  acc_body2 = scalar_multiply(1.0/mass_body2, force_on_body2)

  # update position
  pos_body1 = vector_add(pos_body1, scalar_multiply(delta_t, vel_body1))
  pos_body2 = vector_add(pos_body2, scalar_multiply(delta_t, vel_body2))

  # update velocity
  vel_body1 = vector_add(vel_body1, scalar_multiply(delta_t, acc_body1))
  vel_body2 = vector_add(vel_body2, scalar_multiply(delta_t, acc_body2))

  # go to next step
  istep += 1

# report final position and velocity
print(f'Body 1 position: {pos_body1}, velocity: {vel_body1}')
print(f'Body 2 position: {pos_body2}, velocity: {vel_body2}')

```

You can try the simulation out yourself. When you've convinced yourself you understand this example, try writing a simulation for throwing a ball on the surface of the earth. First, you can just assume gravity is a constant force pulling in the negative y direction: $\vec{F}=-gm\hat{y}$. How far does your simulation predict the ball would travel if you threw it from a height of 1.5 m with a velocity of $\vec{v}=(10, 3)$ m/s? To figure out when the ball "hits" the ground, you may want to use an `if` statement. What if you add in a force of air resistance $\vec{F}=-(0.00518 \mathrm{kg}/\mathrm{m})\vert\vec{v}\vert\vec{v}$?
