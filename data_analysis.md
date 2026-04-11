
# Data analysis for particle physics

This page is a tutorial on data analysis for particle ("high-energy") physics. It is a work-in-progress and thus may experience unexpected reorganization. 

It is forseen that this tutorial will cover programming in Python; libraries for data science such as numpy, matplotlib, numba, pytorch, etc.; and some probability and statistics. All topics will be covered with applications to particle physics. Much of this content does not yet exist.

1. [Programming in Python](#programming-in-python)

    1.1. [Getting started](#getting-started-and-documentation-in-science)

    1.2. [Expressions and variables](#expressions-and-variables-in-python)

    1.3. [Functions and control](#functions-and-control-in-python)

    1.4. [Classes and libraries](#classes-and-libraries-in-python)

2. [Data analysis libraries](#data-analysis-libraries)

    2.1. [Data manipulation](#data-manipulation-libraries:-numpy,-awkward,-pandas,-numba,-and-vector)

    2.2. [Data visualization](#histograms-and-data-visualization-with-hist,-matplolib,-and-mplhep)

    2.3. [Reading and writing data](#reading-and-writing-data-with-uproot-and-coffea)

    2.4. [Analysis of collider data](#the-structure-and-analysis-of-collider-data)

    2.5. [Machine learning](#machine-learning-with-xgboost-and-pytorch)

3. [Probability and statistics](#probability-and-statistics)

    3.1. [Concepts of probability and statistics](#concepts-of-probability-and-statistics)

    3.2. [Statistical models and fitting](#statistical-models-and-fitting)

# Programming in Python

## Getting started and documentation in science

You will need to install python, ideally via a package manager such as [miniforge](https://github.com/conda-forge/miniforge). If you are using Windows, install Windows Subsystem for Linux, after which the instructions will be the same. Open a shell ("terminal" on Mac) and install miniforge with:

```sh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
Once you have installed miniforge, you can create a python environment, which we will call `hep_default`:

```sh
conda create --name hep_default python=3.12
conda install conda-forge::coffea 
conda install jupyterlab
conda install fsspec-xrootd
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

A good introductory computer science course text in python can be found [here](https://www.composingprograms.com/). These notes will just give a somewhat abridged rundown of useful concepts.

A programming language allows one to build up programs by combining basic elements called **statements**. The first thing you can try is running code to do numeric calculations such as

```py
5+4**2
```

If you are using Jupyter, simply type this into a cell, then press shift+enter to run the cell. We'll cover functions in more detail later, but you can get access to some familiar mathematical functions by running

```py
import math
```

(note this import statement needs to be executed once per notebook/script) after which you can evaluate expressions like

```py
5.0*math.sqrt(7.0)-math.log(2.0)
```

The symbols `+`, `-`, `*`, `/`, `%`and `**` are used to represent addition, subtraction, multiplication, division, division remainder (modulo), and exponentiation respectively. You can see what functions are available from the math library using the [documentation page](https://docs.python.org/3/library/math.html).

You can also write assignment statements, which store values in **variables** using the `=` operator, for example:

```py
z_mass = 91.1876
```

This line stores the value `91.1876` in the computer memory at a location that we have named `z_mass`. We call `z_mass` a variable. We can then retrieve this information in later statements:

```py
math.sqrt(2)*z_mass
```

All data whether literals (fixed data like `91.1876` or `3`) or variables have a particular **type**. Some of the basic types in python are:

| Type   | Description                               | Examples                |
|--------|-------------------------------------------|-------------------------|
| int    | Integer                                   | `4`, `-12`, `255`       |
| float  | "Floating point" approx. to a real number | `3.1416`, `6.022e23`    |
| string | Sequence of characters                    | `'hello'`, `"__main__"` |
| bool   | True or False value                       | `True`, `False`         |
| list   | Ordered collection of data                | `[3.8, -2.2, 4.0, 6.3]` |
| dict   | Map from "keys" to "values"               | `{1: 'CMS', 2: 'LHCb'}` |

Trying to use data of a given type in the wrong way will result in an error. For example, both `5.0+3.2` and `"Hello "+"world"` are valid statements; one is normal addition of numbers while the `+` operator for strings acts as concatenation. But, trying `5.0 + "Hello"` will give an error since adding a floating point number and a string is not well defined. 

In an assignment statement, the right-hand side of the statement is evaluated before the new value is assigned. So if a variable `a` currently contains the integer 5, then

```py
a = a + 1
```

will calculate `a+1`, then overwrite the value stored in the variable `a` making it 6. This is common enough that there is shorthand for it: `x += y` is equivalent to `x = x + y` with analogous operators `-=`, `*=`, and `/=` as well.

Order of operations roughly follows the standards for math with which you are familiar with, with parentheses always coming first. So the expression

```py
max(math.log(33.2), math.sin(4.3))
```

first evaluates $\log(33.2)$ and $\sin(4.3)$, then takes the maximum of the two. When in doubt, you can use parentheses to make sure the operations are performed in the order you want.

Some other basic operations that are useful for comparisons are listed below

| Operation | Input types | Output type | Description                         |
|-----------|-------------|-------------|-------------------------------------|
| `a == b`  | any         | bool        | `a` equals `b`                      |
| `a < b`   | any         | bool        | `a` is less than `b`                |
| `a <= b`  | any         | bool        | `a` is less than or equal to `b`    |
| `a > b`   | any         | bool        | `a` is greater than `b`             |
| `a >= b`  | any         | bool        | `a` is greater than or equal to `b` |
| `a != b`  | any         | bool        | `a` is not equal to `b`             |
| `not a`   | bool        | bool        | inverts true and false              |
| `a or b`  | bool, bool  | bool        | true iff `a` or `b` is true         |
| `a and b` | bool, bool  | bool        | true iff `a` and `b` are true       |
| `a & b`   | int, int    | int         | bitwise and                         |
| `a | b`   | int, int    | int         | bitwise or                          |
| `~ a`     | int         | int         | bitwise not                         |

Lists and dictionaries can consist of multiple values, which are accessed using the `[]` operator.

```py
my_list = [1, 2, 3, 4]
print(my_list[2])

lhc_locations = {'ATLAS' : 1, 'ALICE' : 2, 'CMS' : 5, 'LHCb' : 8}
print(lhc_locations['CMS'])
```

Note that the indexing for lists starts from 0, so the 0th element of `my_list`  above is `1`. Using negative indices works backward from the end, with the -1st element of a list being the last one. You can add a new element to the end of a list using `my_list.append(new_element)`, and you can add a new element to dictionary with `my_dictionary[new_key]=new_element`. There are many more functions for working with lists and dictionaries that you can find in the [relevant](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range) [documentation](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict).

Finally, it is worth again emphasizing that keeping all your work organized and easily understandible is crucial for scientists. If you are using Jupyter, you can change the cell type from "Code" to "Markdown" or "Raw" to write down notes in [markdown](https://www.markdownguide.org/basic-syntax/) or plain text. If are not using Jupyter or you want to add comments directly into your code you can use a `#` to add a comment into your python code:

```
# calculate the energy and momentum
e = e1 + e2
p = p1 + p2
# now calculate the invariant mass
m = e**2 - p**2
```

A common simple use of programming in high-energy physics is serve as a calculator. The exercises below will help you get familiar with using python in this way.

### Exercise 1.1: Natural units

In particle physics, we often use "natural units" where the speed of light $c$ and the reduced Planck constant $\hbar$ are equal to 1. To have $c=1$ when $c$ is usually about $3.0\times 10^8$ meters per second, we measure distance and time in the "same" units. For example, we can define a new unit of *time* called meters such that 1 second is equal to about $3.0\times 10^8$ meters of time and thus 1 meter of time is about $3.33\times 10^{-9}$ seconds. Then, c is equal to $\frac{3.0\times 10^8 \mathrm{ m}}{3.0\times 10^8 \mathrm{ m}}=1$. Similarly, to set $\hbar$ to 1 when it is usually $1.054\times 10^{-34}$ joule seconds, we measure time and energy (and distance and momentum) in inverse units. So, we define an inverse joule of *time* to be about $1.054\times 10^{-34}$ seconds, which means 1 second is about $9.49\times 10^{33}$ inverse joules. Then $\hbar$ is $(1.054\times 10^{-34} \mathrm{ J})(9.49\times 10^{33} \mathrm{ J}^{-1})=1$. To check your understanding, consider how far is 1 second of distance? How much is 1 inverse second of energy? 1 inverse joule of distance (hint: this requires $\hbar=c=1$)?

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

Muons are produced by cosmic rays hitting the earth's atmosphere. The mass of a muon is about $\tau = 106 \mathrm{MeV}/c^2$ and its rest frame mean lifetime is $2.2\times 10^{-6}$ s. Consider muons with momentum 200 MeV/c, 1 GeV/c, and 5 GeV/c. Using the relation $E^2=m^2c^4+\vert\vec{p}\vert^2c^{2}$ in natural units, compute the gamma factor for each. The gamma factor is dimensionless and thus identical in all unit systems. Calculate the velocity in m/s for each of these muons using $1 \mathrm{ eV}=1.602\times 10^{-19} \mathrm{ J}$. Accounting for time dilation, the average distance travelled is $\gamma\tau v$. Calculate this for each muon. If a cosmic ray shower muon is typically produced 20 km above sea level, which of these might you expect to reach sea level?

## Functions and control in python

More complex control of a program can be acheived with compound statements including functions, conditionals, and loops. You can define a **function** in python using a format similar to the following:

```py
def hypotenuse(a, b):
  c = math.sqrt(a**2+b**2)
  return c
```

The keyword `def` comes first, followed by the function name (in this example `hypotenuse`), the arguments (inputs) to the functions in parentheses, then a semicolon. The body of the function defining what it does it then specified by a series of indented lines. The output of the function can be specified by a line with the `return` keyword. One can then call this function

```py
hypotenuse(3.0, 4.0)
```

Each variable has a **scope** from which it is visible. Variables assigned outside of functions are said to have global scope and can be accessed from anywhere, but variables assigned inside of a function can only be accessed within that function (including other functions defined in the same parent funtion). So, for example, the following code will result in an error since `b` is not visible outside of the `set_b` function

```py
def set_b(x):
  b = x

def add_b(x):
  return x+b

set_b(3)
add_b(5)
```

Functions can also be treated as a data, just like any other type. This means they can be assigned to variables or even provided as inputs or outputs from other functions ("higher-order functions"). The following code shows how one can create a function whose output is a function that adds a number to its input.

```py
def make_adder(summand):
  def add_summand(x):
    return x+summand
  return add_summand

add3 = make_adder(3)
add3(5)
```

Functions can be provided with "default arguments" that will be used if the user does not provide an argument. This can be seen in the following example:

```py
def get_energy(momentum, mass=0.0):
  return math.sqrt(momentum+mass)

print(get_energy(16.8))
print(get_energy(16.8, 0.45))
```

For those familiar with other programming languages, it is worth noting that Python does not support function overloading (multiple functions with the same name but different input types).

There are some other optional but common conventions you will likely see when looking at Python code. It is common to have a "docstring" in triple quotes at the beginning of a function explaining what the function does. You may also see the intended types of the input and output variables of a function specified with `:` and `->` respectively. Note that unlike other programming languages, this does not restrict the input types, it is simply a guide for users. These are demonstrated in the following code snippet:

```py
def hypotenuse(a: float, b: float) -> float:
  """Calculates the length of a hypotenuse of a right triangle with the 
  Pythagorean theorem

  Args:
    a: length of one side
    b: length of one side

  Returns:
    length of the hypotenuse
  """
  c = math.sqrt(a**2+b**2)
  return c
```

To have code that executes in certain cases, you can use **if statements**. For example:

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

To have code that executes repeatedly, one can use loops. For example, a **while loop** will execute a set of statements repeatedly as long as a particular condition is true:

```py
def factorial(x):
  x_factorial = 0
  while x > 0:
    x_factorial *= x
    x -= 1
  return x_factorial
```

In this case, the indented code block after the `while` will continue to run until the condition (`x > 0`) is no longer true. 

Another form of loop is the **for loop**:

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

The for loop allows one to iterate over the elements of a collection such as a list. You will often see `for` loops used together with `range(start, stop, step)`, which produces a sequence of number starting at `start` (defaultly zero), stopping at `stop` (not including `stop`), and incrementing by `step` (defaultly one) each time. So, we could implement the factorial function from before with a for loop using:

```py
def factorial(x):
  x_factorial = 0
  for y in range(1, x+1):
    x_factorial *= y
  return x_factorial
```

### Exercise 1.3: Get filenames

While most high-energy physics programming involves numerical data, it is not uncommon to work with strings as well. A typical case is dealing with filenames. Suppose we have three years of data files stored in folders named `2022`, `2023`, `2024`. In each folder are 6 files: `DY.root`, `TT.root`, `tW.root`, `WW.root`, `WZ.root`, and `ZZ.root`. Write a loop to generate a list of all filenames across all years (ex. t`2022/DY.root`). Note that you can use the `append` function to add data to the end of an existing list:

```py
my_list = [1, 2, 3]
my_list.append(4)
```

### Exercise 1.4: Newtonian simulation

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

We know that $\vec{v}=\frac{d\vec{x}}{dt}$ and $\vec{a}=\frac{d\vec{v}}{dt}$. So, if we pick a time $\Delta t$ that is short enough that $\vec{v}$ is roughly constant, then the new position after $\Delta t$ time is $\vec{x}\_{\mathrm{new}} = \vec{x}+\frac{d\vec{x}}{dt}\Delta t=\vec{x}+\vec{v}\Delta t$ and similarly $\vec{v}\_{\mathrm{new}}=\vec{v}+\vec{a}\Delta t$. We can then just repeat this over and over to get the position and velocity at some later time.

The last ingrediant is calculating the accereration which depends on the force and mass $\vec{a}=\vec{F}/m$. Newtonian gravity says that the force on body 1 is $\vec{F}=\frac{Gm\_{1}m\_{2}}{\vert\vec{r}\vert^3}\vec{r}$ where $\vec{r}$ is the the displacement between the bodies $\vec{r}=\vec{x}\_{2}-\vec{x}\_{1}$, and an analogous expression holds for body 2. We have to decide on masses $m\_{1}$ and $m\_{2}$ as well as a system of units, which will determine the constant $G$. In code, our full simulation is then

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

### Exercise 1.5: Four-vector coordinate conversion

For this exercise, we will work in natural units where $c=1$. In special relativity, space and time msut be treated together as spacetime as the space and time axes differ for different observers. This means points in spacetime are specified by a 4D vector $(t, x, y, z)$. The energy and spatial momentum also appear as a 4D vector called the four-momentum $(E, p\_{x}, p\_{y}, p\_{z})$.

In collider experiments, 4-momenta components are often given in another format that is related to spherical coordinates: the transverse momentum $p\_\mathrm{T}$, the [pseudorapidity](https://en.wikipedia.org/wiki/Pseudorapidity) $\eta$, the azimuthal angle $\phi$, and the invariant mass $m$. These are related to the rectilinear coordinates via the transformations:

$$E = \sqrt{(p_\mathrm{T}\mathrm{cosh}\eta)^2+m^2}$$

$$p_{x} = p_\mathrm{T}\cos\phi$$

$$p_{y} = p_\mathrm{T}\sin\phi$$

$$p_{z} = p_\mathrm{T}\mathrm{sinh}\eta$$

and in reverse:

$$p_\mathrm{T}=\sqrt{p_x^2+p_y^2}$$

$$\eta = -\log\left[\tan\left(\frac{\theta}{2}\right)\right] = \mathrm{arctanh}\left(\frac{p_z}{p_x^2+p_y^2+p_z^2}\right)$$

$$\phi = \mathrm{atan2}(y, x)$$

$$m = \sqrt{E^2-p_x^2-p_y^2-p_z^2}$$

where atan2 is the [2-argument arctangent](https://en.wikipedia.org/wiki/Atan2). Write functions to convert between coordinates given as Python lists `[e, px, py, pz]` and `[pt, eta, phi, m]` in both directions. You can find the relevant functions (sin, cos, arctanh, etc.) in the Python [math](https://docs.python.org/3/library/math.html) library. 

Suppose we have a pair of electrons whose collider coordinates in units of GeV are `[40.4872, -0.4971, 0.5084, 0.000511]` and `[40.4872, 0.4971, -2.6331, 0.000511]`. Convert these to rectilinear coordinates, add them together, and convert back to collider coordinates. What is the invariant mass $m$ of the sum? If the electrons came from the decay of a parent particle, this would be the mass of this particle. If you are curious, you can check the [PDG](https://pdg.lbl.gov/) for what particle this might be.

### Exercise 1.6: Higher-order functions

You can also use functions the same way you would use data of other types in terms of assigning functions to variables and providing them as inputs to other functions. 

Suppose we have some data about 20 photons formatted a list where the list indices indicate the photon:

```py
photon_px = [10.27, 10.28, -9.104, 14.25, 8.769, -2.154, 4.331, 6.5, 14.7,
             5.247, 17.56, -1.196, -29.26, 2.582, -4.645, 1.36, 20.46, 
             -16.98, 7.724, 16.49]
photon_py = [-24.6, 12.66, -6.222, 17.74, 5.169, -1.032, -16.75, 10.99,
             0.8502, -0.4955, 13.14, -5.817, -3.965, -4.662, 4.764, 1.399,
             31.12, 3.2, -4.735, 4.87]
photon_pz = [-4.75, -0.5737, 17.99, -2.512, 10.83, 6.137, -13.77, 7.15, 
             24.49, -4.812, 15.28, -43.1, 23.73, -23.94, -0.9966, -0.8781,
             10.87, 23.31, -3.082, -5.072]
```

We will take these momenta to be in GeV (using natural units). We can write a function that will evaluate a function `func` for each photon in the list:

```py
def photon_fn_evaluator(photon_px, photon_py, photon_pz, func):
  result = []
  for px, py, pz in zip(photon_px, photon_py, photon_pz):
    result.append(func(px, py, pz))
  return result
```

So, for example, the energy of a photon is given by $E=\sqrt{\vert\vec{p}\vert^2+m^2}=\sqrt{p\_x^2+p\_y^2+p\_z^2}$ as photons are massless. So, we can get the energy by combining a `get_energy` function with the `photon_fn_evaluator`:

```py
def get_energy(px, py, pz):
  return (px**2+py**2+pz**2)**0.5

photon_e = photon_fn_evaluator(photon_px, photon_py, photon_pz, get_energy)
print(photon_e)
```

Once you understand how this works, try writing analogous `get_pt`, `get_eta`, and `get_phi` functions using the conversions in the previous problem and run them with `photon_fn_evaluator`. What are the $p\_\text{T}$, $\eta$, and $\phi$ values for the photons in this example?

This example is basically how functions are applied on real physics data when using ex. numba and awkward, as you will learn later.

## Classes and libraries in python

You are probably familiar with the fact that computers can store much more complicated data than just numbers, booleans, and strings. These more complicated data are generally built up from simpler pieces. For example, an image can be decomposed into pixels, each of which consists of some numbers to determine it's color, or audio data can be broken down into a long sequence of numeric data representing the amplitude of the sound wave over time.

A **class** is effectively a custom type capable of storing both data and functions. Consider the example below:

```py
class Particle:
   def __init__(self, px, py, pz, m):
     self.px = px
     self.py = py
     self.pz = pz
     self.m = m

  def get_energy(self):
     return math.sqrt(self.px**2+self.py**2+self.pz**2+m**2)
```

This defines a `Particle` type, which is initialized with four numbers representing its spatial momenta and mass. The special `__init__` function defines how a class is initialized. The data stored in an instance of the class as well as the functions associated with a class can be using `.`. So if `x` is a variable of type `Particle`, `x.m` would be its mass and `x.get_energy()` would call the `get_energy` function. Variables belonging to an instance of the class are accessed from within the class by using the special variable `self`, which is always passed as the first argument of a class function when called with the `.` operator.

```py
x = Particle(4.5, 3.3, 6.5, 0.106)
print(x.m)
print(x.get_energy())
```

A class can "inherit" the data of a parent class in the following way:

```py
class Electron(Particle):
   def __init__(self, px, py, pz):
     self.px = px
     self.py = py
     self.pz = pz
     self.m = 0.511
```

Any functions that are not overridden (redefined) by child class are inherited from their parent. In this case, the `Electron` class does not override the `get_energy` function, so we can run the code

```py
x = Electron(4.5, 3.3, 6.5)
x.get_energy()
```

While implementing functionality from the ground up is useful for learning, in practice, most programming uses **libraries**, collection of useful elements like functions and classes that have already been written. You have already seen us use Python's `math` library to get access to functions like `log`, `sqrt`, etc. In python, libraries can be imported using an import statement

```py
import math
```

You can also define a shorthand name for the library

```py
import numpy as np
```

this would allow you to access functions like `numpy.sum` using `np.sum`. You can even import functions from a library so that you don't need the library name at all

```py
from numpy import sum
```

However, this should be done with caution since there are often functions with the same name in multiple libraries (ex. `numpy.sum` versus `awkward.sum`). There are a set of [standard libaries](https://docs.python.org/3/library/) that you always have access to with any python installation, but you can also add your own libraries by installing them as a package. This is what we did when we first set up our installation with

```sh
conda install <packagename>
```

Much of the rest of this tutorial will focus on working with specific libraries used for data analysis.

# Data analysis libraries

## Data manipulation libraries: numpy, awkward, pandas, numba, and vector

Interpreting python statements is actually very slow, so to analyze large amounts of data, we will need to have more efficient ways to perform data manipulation. This is where the `numpy` and `awkward` libaries come in. These libraries allow us to do "vectorized" operations across large amounts of data in just one statement. `awkward` records and `pandas` dataframes also allow for operations on many "columns" of data at once. `vector` provides new data types and operations for physics four vectors. Finally, when there are no built-in `numpy` or `awkward` functions for doing the desired data manipulation, the `numba` library can be used to compile python code and make custom fast functions.

### Getting started with numpy and awkward

<!--Content:-->
<!--Intro to numpy arrays and vector operations/functions-->
<!--Components and shape -->
<!--Reduction functions -->
<!--Broadcasting -->
<!--Intro to awkward arrays -->
<!--Creating arrays -->

Throughout this section, we will assume we have imported `numpy` and `awkward` as follows:

```py
import numpy as np
import awkward as ak
```

The basic class introduced by numpy is the `np.ndarray` (alias `np.array`). This acts similarly to a Python list, except that the data must all be the same type, i.e. `[3.0, 'Hello']` is a valid Python list, but `np.array([3.0, 'Hello'])` will throw an error. The advantage of numpy lists is that we can perform operations on all the components of a numpy array at the same time, so

```py
x_array = np.array([3.0, 7.2, 2.3, 8.4, 9.7, 4.1])
y_array = np.array([1.7, 7.0, 8.4, 5.3, 1.7, 8.9])

z_array = x_array * y_array
```

can be done quickly in one `*` operation, where as the analogous version with Python lists requires a loop and is much slower because the command is evaluated for each element individually.

```py
x_list = [3.0, 7.2, 2.3, 8.4, 9.7, 4.1]
y_list = [1.7, 7.0, 8.4, 5.3, 1.7, 8.9]
z_list = []

for x, y in zip(x_list, y_list):
  z_list.append(x*y)
```

It is said that numpy operations are "vectorized", not in the math/physics sense of the word vector, but in the sense that operations like `+`, `-`, `*`, `/`, `==`, `<`, and so forth are performed component-wise for each component in the array. Due to limitations in Python, boolean operators like `and`, `or`, and `not` can't be used, though for *booleans* specifically, you can use their bitwise versions `&`, `|`, and `~` (though you should be careful with order of operations with these operators, they are high priority than comparisons like `==` or `>`). `numpy` functions like `np.sqrt()`, `np.absolute()`, etc. can be used to perform component-wise operations on `numpy` or `awkward` arrays. 

```py
#each component of b is sqrt of the corresponding component of a
b = np.sqrt(a)
```

You can find a list of the numeric functions in the [documentation](https://numpy.org/doc/stable/reference/routines.math.html) and other functions in the associated documentation pages. One other common `numpy` function is `np.where`, which is effectively an if statment.

```py
lead_lepton_pt = np.where(lead_electron_pt > lead_muon_pt, 
                          lead_electron_pt, 
                          lead_muon_pt)
```

Arrays in numpy can have any number of dimentions. You can access a component of a numpy array using the `[]` operator.

```py
x = np.array([[1, 2], [3, 4], [5, 6]])
print(x[0, 0])
```

As shown in this example, when working with multidimensional arrays, multiple indices can be provided separated by a comma. The first index corresponds to the outermost array, while the last index corresponds to the innermost array. Like python lists, negative indices count backward from the end, so `x[-1]` is the last element of a 1D array. 

You can also use the range operator `:` to select a range of indices, inclusive of the starting point and exclusive of the ending point.

```py
print(x[1:3, :])
```

The range operator `:` has a default starting point of the first element (0) and a default ending point after the last element when not specified, meaning that `:` by itself with select all indices along a given axis.

You can get the dimensions of an array with using `shape`, which returns a tuple of the dimensions.

```py
print(x.shape)
```

There are also various numpy functions that can reduce dimensions such as `np.sum`, `np.mean`, and `np.std`, which take the sum, mean, and standard deviation of the entries of an array respectively. When used without an argument, they reduce the array to a single scalar, though an axis can be specified to just take the sum/mean/standard deviation of a particular axis and return a lower dimensional array.

An important feature of numpy is **broadcasting** whereby a user can perform operations on two objects, even if they do not have the same shape (i.e. are not arrays of the same size). The most common usage of this is to broadcast a scalar value to an array. For example, the code below adds 2 to *each* component of the array `x`.

```py
x = x + 2
```

Broadcasting also allows for operations on arrays of different shapes: any axes of length 1 can be extended (by duplication) to the length of the other array in the operation, and if one array is lower dimension than the other, it will be expanded by duplication adding dimensions "on the left".

```py
a = np.array([[2.4, 3.5], [5.1, 0.5] [2.5,8.0]])
b = np.array([[1.0], [2.0], [3.0]])
c = a+b
```

More details on numpy broadcasting can be found [here](https://numpy.org/doc/stable/user/basics.broadcasting.html).

Awkward array or `awkward` serves as a generalization of `numpy` for dealing with data that are not rectangular arrays. This is common in particle physics where the data might consist of a collection of particles for each "event", but the number of particles is different for each event.

`awkward`'s array class is just called `ak.Array` and for the most part can be used just like `np.array` in terms of providing vectorized operations, indexing, broadcasting, etc.

```py
x = ak.Array([[1.0, 8.7, 5.7], [4.8], [7.0, 1.5]])
y = ak.Array([[3.1, 8.0, 1.2], [7.8], [5.3, 0.2]])

z = x + y
```

The type and dimensions of an `awkward` array `x` can be retrieved with `x.type` and printed with `x.type.show()`. Each dimension can be ether regular (a fixed length) or ragged (variable length). Completely regular arrays work identically to numpy, though arrays with ragged dimensions do have some slight differences such as that broadcasting can add dimensions "on the right" rather than "on the left".

There are various useful functions for working with `awkward` arrays. The `ak.sum()` function will replace a given dimension of an array with the sum over it's components. This is particularly useful when used with `axis=-1`. When provided booleans, sum will treat `True` as `1` and `False` as `0`. This allows one to count the number of elements meeting certain criteria. For example, if `electron_pt` is the transverse momenta of electrons (inner index) by event (outer index), we can get the number of electrons in each event with transverse momentum greater than 20 as an array with `ak.sum`.

```py
ak.sum(electron_pt > 20, axis=-1)
```

Some functions that work analogously to `ak.sum` include `ak.prod`, `ak.min`, `ak.max`, `ak.argmin`, `ak.argmax`, `ak.count`, `ak.mean`, `ak.var`, and `ak.std`. 

You will also commonly need to generate new `numpy` and `awkward` arrays. A particularly common way to initialize them is with the `zeros_like`, `ones_like`, and `full_like` methods, which generate a new array with the same shape as an existing one but filled with all zeros, ones, or a fixed value respectively.

```py
# suppose we have an awkward array Muon_pt
# if we want to generate an array Muon_mass of matching size, we can use 
# full_like since the muon mass is a constant (0.106 GeV)

Muon_mass = ak.full_like(Muon_pt, 0.106)
```

You can find more information about `numpy` and `awkward` in their [respective](https://numpy.org/doc/stable/) [documentation](https://awkward-array.org/doc/main/index.html) pages.

### Indexing and manipulation with awkward

<!--Content: -->
<!--boolean indexing -->
<!--numeric indexing -->
<!--masking and working with none -->

In this section, you will learn some useful tricks for indexing and manipulating `numpy` and `awkward` arrays. As in the previous section, we will assume we have imported `numpy` and `awkward`.

```py
import numpy as np
import awkward as ak
```

One of the most useful tricks is boolean indexing. If `x` is a (`numpy`/`awkward`) array and `y` is a boolean array of the same shape, then `x[y]` refers to the components of `x` for which `y` is true. As an example, the following code subtracts 2 from all components of an array `x` that are greater than 4.

```py
# create boolean array with broadcasting
# greater_than_four will have the same shape as x
greater_than_four = x > 4

x[greater_than_four] -= 2
```

Boolean indexing is commonly used with awkward arrays for filtering.

```py
highpt_electron_pt = electron_pt[electron_pt > 50]
```

Another common indexing trick for `awkward` arrays is using arrays of integer indices. Indexing an awkward array with an array of integer indexes will produce an array the same shape as the indexing array, but with the values replaced by those in the array being indexed. This is perhaps best demonstrated via example.

```py
Electron_pt = ak.Array([[30.2, 21.4, 33.3], [42.0, 87.0], [58.3, 83.2]])
Electron_indexes = ak.Array([[0, 2, 2, 2, 1], [1], [1, 1, 0]])
Indexed_Electron_pt = Electron_pt[Electron_indexes]
# = [[30.2, 33.3, 33.3, 33.3, 21.4], [87.0], [83.2, 83.2, 58.3]]
```

This can be used, for example, to get the value of one awkward array corresponding to the maximum/minimum number in another array, or to sort an array by another.

```py
# suppose we have awkward arrays Photon_pt and Photon_eta

highest_pt_photon_index = ak.argmax(Photon_pt)
eta_of_highest_pt_photon = Photon_eta[highest_pt_photon_index]

photon_indexes_sorted_by_pt = ak.argsort(Photon_pt)
photon_eta_sorted_by_pt = Photon_eta[photon_indexes_sorted_by_pt]
```

Working with indexes will frequently expose you to cases where one has to deal with missing data, represented in awkward arrays as `None`. To set values to `None`, you can use `ak.mask`.

```py
# Suppose Photon_electronIdx encodes the index of an electron if an object is
# reconstructed as both a photon and an electron, but is -1 to denote that
# there is no equivalent electron object. We can mask the -1 values to None

Photon_electronIdx_masked = ak.mask(events.Photon_electronIdx,
      events.Photon_electronIdx != -1)
```

To do the reverse and replace `None` with a specified value, you can use `ak.fill_none`.

```py
Photon_isSignalElectron = ak.fill_none(Electron_sig[Photon_electronIdx_masked],
                                       False)
```

If you wanted to get the second value of each sub-array, but you are not sure if all subarrays have at least 2 values, you can use `ak.pad_none` to add `None` to all subarrays until they have at least two values.

```py
# suppose we want to get the second highest pT electron for each event
# or 0 for events without at least two electrons

sorted_Electron_pt = ak.sort(Electron_pt, axis=-1, ascending=False)
sorted_Electron_pt = ak.pad_none(Electron_pt, 2)
second_highest_Electron_pt = ak.fill_none(sorted_Electron_pt[:, 1], 0.0)
```
### Exercise 2.1. Manipulating arrays

We will generate some random data that is similar to what you might see in a collider experiment.

```py
import awkward as ak
import numpy as np

# initialize random number generator with seed for reproducibility

rng = np.random.default_rng(12345)

# we will generate random numbers with rng.random which takes the shape of
# array to be generated, and rng.integers, which takes the lowest allowed 
# value, highest allowed value, and shape of array to be generated

# we convert a numpy array to an awkward array using ak.from_numpy
# even though the arrays are regular-dimensioned, we want to treat the second
# dimension as if it were ragged. This is what ak.from_regular(,1) does

electron_pt = ak.from_regular(ak.from_numpy(rng.random((10000,2))*100.0),1)
electron_eta = ak.from_regular(ak.from_numpy(rng.random((10000,2))*5.0-2.5),1)
electron_id = ak.from_regular(ak.from_numpy(rng.integers(0,2,(10000,2))),1)
photon_pt = ak.from_regular(ak.from_numpy(rng.random((10000,3))*150.0),1)
photon_eta = ak.from_regular(ak.from_numpy(rng.random((10000,3))*10.0-5.0),1)
photon_id = ak.from_regular(ak.from_numpy(rng.integers(0,2,(10000,3))),1)
```

We will discuss more below, but the data are formatted so that the outer/left index represents the event and the inner/right index represents the particles in the event. How many events does our data correspond to? How many electrons/photons are in each event?

When using collider data, we typically apply some preselections to the objects. Filter all 6 arrays so that we consider only electrons with $p\_\mathrm{T}>20$ (GeV), $\vert\eta\vert<2.5$, and ID (representing some quality criteria) equal to 1 and only photons with $p\_\mathrm{T}>30$ (GeV), $\vert\eta\vert<2.5$, and ID equal to 1.

Create new arrays with the number of selected electrons and photons in each event using `ak.count(array,axis)`. How many have at least two electrons and at least one photon (recall you can use `ak.sum(array, axis)` which treats boolean true and false as 1 and 0)? Construct an array that contains the largest selected photon $p_\mathrm{T}$ for each event with at least 2 selected electrons.

### pandas and awkward records

In this section we will assume we have imported `numpy`, `awkward`, and `pandas`.

```py
import awkward as ak
import numpy as np
import pandas as pd
```

Suppose we are working with the full event data consisting of many different arrays representing the properties of detected particles. If we want to examine the transverse momentum ($p\_\mathrm{T}$) of electrons in events with at least one photon, we could use boolean indexing

```py
filtered_electron_pt = electron_pt[nphoton >= 1]
```

But what if we want to look at many different properties of electrons such as their pseudorapditiy $\eta$, azimuthal angle $\phi$, mass, etc? Rather than having to filter many arrays independently, `awkward` offers a data structure called a record that can store many arrays inside of it. You can think of a record as a dictionary/class whose elements are awkward arrays.

```py
electrons = ak.Array({
    'px' : [5.6, 8.7, 0.2],
    'py' : [0.1, 5.0, 3.7],
    'pz' : [3.7, 0.6, 5.5]
})

print(electrons.px)
print(electrons.py)
print(electrons.pz)
```

You can access a given array in a record either using member syntax like `electrons.px` or using element syntax like `electrons['px']`. The main advantage of using a record is that you can perform operations like filtering on all of the awkward arrays contained in the record in a single statement.

```py
electrons_pt = np.sqrt(electrons.px**2 + electrons.py**2)

highpt_electrons = electrons[electrons_pt > 40.0]
```

You can also easily add a new array to an existing record using the `[]` operator.

```py
electrons['quality'] = electron_quality
```

While `numpy` does not by itself have functionality similar to `awkward` records, you can get similar functionality using **dataframes** provided by the `pandas` library. Like records, you can think of a `pandas` dataframe like a dictionary of `numpy` arrays (technically they are `pandas` series, but we will not worry about the differences here) that allows you to perform operations on all contained arrays at once.

```py
rng = np.random.default_rng()
df = pd.DataFrame()
df['lead_electron_pt'] = rng.random((50,))*100
df['lead_muon_pt'] = rng.random((50,))*100
df_highpt = df[(df['lead_electron_pt']>50) & (df['lead_muon_pt']>50)]
```

### Physics vectors with vector

For this section we will assume we have imported

```py
import awkward as ak
import vector
```

It is very common for physics calculations to involve four-vectors, paritcularly the four-momentum vector $(E, p\_{x}, p\_{y}, p\_{z})$. In previous exercises, you have seen that there are many different ways to fully specify the four-momentum, with the most common version used in hadron collider experiments being in terms of transverse momentum $p\_\mathrm{T}$, pseudorapidity $\eta$, azimuthal angle $\phi$, and mass $m$.

One particularly common calculation is finding the **invariant mass** of a set of particles. You know that the mass of a particle is related to its energy and momentum via the equation $E^2=\vert\vec{p}\vert^2+m^2$ in natural units. Suppose we have a Higgs boson that decays into a pair of photons. Since energy and momentum are conserved, if we know the energy and momentum of the photons, we can simply sum them to get the energy and momentum of the Higgs boson candidate, from which we can calculate its mass. In an actual experiment the same relation is also used to calculate the momentum of the muons since it is actually the energy/direction that is measured (and the mass is known to be zero). The invariant mass of a set of particles is thus the answer to the question "*if* these particles came from the decay of another particle, what was the mass of that particle". In real data, events with pairs of photons will sometimes be from a Higgs decay, but the vast majority of the time will be not be from the decay of any heavy particle. If you were to plot the distribution of invariant mass, you would find a smooth distribution from "nonresonant" diphoton events, with the Higgs boson appearing as an excess of diphoton events with an invariant mass at 125 GeV, the Higgs boson mass, as can be seen [here](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-19-015/CMS-HIG-19-015_Figure_014.pdf).

The python `vector` library ([documentation](https://vector.readthedocs.io/en/latest/)) provides methods for working with four-vectors that make it easy to work with them without having to implement everything yourself. As you saw in a previous example, we can use the following code to calculate the invariant mass of two muons

```py
mu1_p = vector.obj(pt = Muon_pt[ievt][imu1],
                   eta = Muon_eta[ievt][imu1],
                   phi = Muon_phi[ievt][imu1],
                   m = 0.106)
mu2_p = vector.obj(pt = Muon_pt[ievt][imu2],
                   eta = Muon_eta[ievt][imu2],
                   phi = Muon_phi[ievt][imu2],
                   m = 0.106)
ll_p = mu1_p + mu2_p
ll_mass = (mu1_p+mu2_p).mass
```

You can similarly use `vector.obj` to initialize a vector with `E`, `px`, `py`, and `pz` coordinates. You can then read any of these properties out to easily convert between coordinate systems.

You can use `vector` together with awkward arrays by running the command

```py
vector.register_awkward()
```

at the beginning of your notebook/processing script. You can use vector operations on awkward arrays of vectors, which can be constructed using `ak.zip` together with the argument `with_name='Momentum4D'` as shown in the following example.

```py
# assume we have awkward/numpy arrays Muon*_pt, Muon*_eta, Muon*_phi

# we make awkward arrays of four-vectors
Muon1_p4 = ak.zip({'pt' : Muon1_pt,
                   'eta' : Muon1_eta,
                   'phi' : Muon1_phi,
                   'm' : ak.full_like(Muon1_pt, 0.106),
                   with_name='Momentum4D')
Muon2_p4 = ak.zip({'pt' : Muon2_pt,
                   'eta' : Muon2_eta,
                   'phi' : Muon2_phi,
                   'm' : ak.full_like(Muon2_pt, 0.106),
                   with_name='Momentum4D')
Dimuon_p4 = Muon1_p4+Muon2_p4

# and calculate and awkward array of invariant mass values
Dimuon_mass = Dimuon_p4.mass
```

For most purposes, this will be sufficient, however, for some purposes (such as writing data to a file), the type after vector calculations will need to be expressly specified.

```py
# force output to be a float
Dimuon_mass = ak.enforce_type(Dimuon_p4.mass, 'float32')
```

### Numba and compiled code

Sometimes, you will need to perform operations that are too complex to easily describe using numba and awkward's built-in functions. A common case of this is combinatoric matching, which is possible in [simple cases](https://awkward-array.org/doc/main/user-guide/how-to-combinatorics-best-match.html), but quickly becomes cumbersome as the matching procedure gets more complicated. In this case, you can use a manual loop, but because interpreted python is very slow, you will want to compile the loop code. This can be done with the `numba` library by using the `@nb.njit` decorator before the function. The following function shows an example that finds a Z boson candidate from awkward arrays of electrons and muons properties.

```py
Z_MASS = 91.1876

@nb.njit
def get_Dilepton(Electron_charge: ak.Array, Electron_pt: ak.Array,
                 Electron_eta: ak.Array, Electron_phi: ak.Array,
                 Muon_charge: ak.Array, Muon_pt: ak.Array,
                 Muon_eta: ak.Array, Muon_phi: ak.Array
                 ) -> tuple[np.array, np.array, np.array, np.array]:
  """Finds electron-antielectron or muon-antimuon pair with mass closest to
  Z_MASS for each event as a Z boson candidate and returns the properties of
  said candidate
  """
  Dilepton_pt = np.zeros(len(Electron_charge), dtype=np.float32)
  Dilepton_eta = np.zeros(len(Electron_charge), dtype=np.float32)
  Dilepton_phi = np.zeros(len(Electron_charge), dtype=np.float32)
  Dilepton_m = np.zeros(len(Electron_charge), dtype=np.float32)
  for ievt in range(len(Electron_pt)):
    for iel1 in range(len(Electron_pt[ievt])):
      for iel2 in range(iel1+1, len(Electron_pt[ievt])):
        if (Electron_charge[ievt][iel1]
            + Electron_charge[ievt][iel2])==0:
          el1_p = vector.obj(pt = Electron_pt[ievt][iel1],
                             eta = Electron_eta[ievt][iel1],
                             phi = Electron_phi[ievt][iel1],
                             m = 0.000511)
          el2_p = vector.obj(pt = Electron_pt[ievt][iel2],
                             eta = Electron_eta[ievt][iel2],
                             phi = Electron_phi[ievt][iel2],
                             m = 0.000511)
          ll_p = el1_p + el2_p
          if abs(ll_p.mass-Z_MASS) < abs(Dilepton_m[ievt]-Z_MASS):
            Dilepton_pt[ievt] = ll_p.pt
            Dilepton_eta[ievt] = ll_p.eta
            Dilepton_phi[ievt] = ll_p.phi
            Dilepton_m[ievt] = ll_p.mass
    for imu1 in range(len(Muon_pt[ievt])):
      for imu2 in range(imu1+1, len(Muon_pt[ievt])):
        if (Muon_charge[ievt][imu1]
            + Muon_charge[ievt][imu2])==0:
          mu1_p = vector.obj(pt = Muon_pt[ievt][imu1],
                             eta = Muon_eta[ievt][imu1],
                             phi = Muon_phi[ievt][imu1],
                             m = 0.106)
          mu2_p = vector.obj(pt = Muon_pt[ievt][imu2],
                             eta = Muon_eta[ievt][imu2],
                             phi = Muon_phi[ievt][imu2],
                             m = 0.106)
          ll_p = mu1_p + mu2_p
          if abs(ll_p.mass-Z_MASS) < abs(Dilepton_m[ievt]-Z_MASS):
            Dilepton_pt[ievt] = ll_p.pt
            Dilepton_eta[ievt] = ll_p.eta
            Dilepton_phi[ievt] = ll_p.phi
            Dilepton_m[ievt] = ll_p.mass
  return Dilepton_pt, Dilepton_eta, Dilepton_phi, Dilepton_m
```

Note that within `nb.njit`-compiled code, certain python functionalities are unavailable. Most notably, although you can pass awkward arrays as arguments and work with them, you cannot use any `ak` functions. If you need to generate a new awkward array, you must use `ak.ArrayBuiler` by initializing a builder outside of the function, passing it in, then calling its `snapshot` method. You can find the ArrayBuilder documentation [here](https://awkward-array.org/doc/main/reference/generated/ak.ArrayBuilder.html#ak.ArrayBuilder).

```py
@nb.njit
def get_Electron_mvacuts(Electron_pt, Electron_eta, Electron_mvacuts_builder):
  for ievt in range(len(Electron_pt)):
    Electron_mvacuts_builder.begin_list()
    for iel in range(len(Electron_pt[ievt])):
      mvacut = 0
      if (Electron_pt[ievt][iel] < 10.0
          and abs(Electron_eta[ievt][iel]) < 0.8):
        mvacut = 0.92661497
      elif (Electron_pt[ievt][iel] < 10.0
            and abs(Electron_eta[ievt][iel]) < 1.479):
        mvacut = 0.91376898
      elif (Electron_pt[ievt][iel] < 10.0):
        mvacut = 0.96821225
      elif (abs(Electron_eta[ievt][iel]) < 0.8):
        mvacut = 0.35267898
      elif (abs(Electron_eta[ievt][iel]) < 1.479):
        mvacut = 0.26008539
      else:
        mvacut = -0.4963113
      Electron_mvacuts_builder.real(mvacut)
    Electron_mvacuts_builder.end_list()
  return Electron_mvacuts_builder

# ...do some processing...
# assume we have Electron_pt and Electron_eta awkward arrays of the same shape

Electron_mvacuts_builder = ak.ArrayBuilder()
get_Electron_mvacuts(Electron_pt, Electron_eta, 
                     Electron_mvacuts_builder)
Electron_mvacuts = Electron_mvacuts_builder.snapshot()
```

Note that sometimes, such as when reading data from a file (described in more detail below), awkward will only initialize arrays on-demand. Since arrays must be initialized before providing them to a `numba`-compiled function, one should use the `ak.materialize` function.

```py
# suppose we read the data from a file into an awkward record "events". To use
# an array "Electron_charge" in a numba-compiled function, we need to 
# materialize it first
ak.materialize(events.Electron_charge)
```

## Histograms and data visualization with hist, matplotlib, and mplhep

Creating visual representation of data is a key part of data analysis that helps the analyzers understand the properties of the data. In particle physics, by far the most ubiquitous and useful data visualization is the **histogram**.

In particle physics, a histogram is more than just a type of plot; a histogram can be though of as an alternate data structure for storing low-dimensional data. Whereas standard columnar (tabular) data is represented as a series of rows (entries), each described as a tuple of numbers for the columns, the same data is represented in an n-dimensional histogram as a series of **bin** counts describing how many rows fall into each bin. When there are few columns (dimensions) and the histogram binning is not overly fine, histograms can provide nearly the same information as columnar data in much more compact form.

For very simple histograms, you can convert a numpy array representing a single column into a numpy array representing the bin counts of the histogram with the `np.histogram` function documented [here](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html). For more complex histograms (higher dimensions, irregular binning, histograms storing uncertainties) or for performing manipulations with histograms, one must use something more complex. We will use the `hist` [library](https://hist.readthedocs.io/en/latest/user-guide/quickstart.html).

```py
import hist
```

You can construct a histogram using `hist.Hist.new` followed by your axes and storage.

```py
my_onedim_hist = hist.Hist.new.Reg(30, 15, 75, name='pt').Weight()
my_twodim_hist = (hist.Hist.new.Reg(30, 15, 75, name='pt')
                  .Var([-2.5, -1.5, 0.0, 1.5, 2.5], name='eta').Weight())
```

In this example, we have used both `Reg[ular]` (evenly-spaced bins) and `Var[iable]` (irregularly-spaced bins) axes. There are other axis types as well such as integer axes `Int` and categorical axes like `IntCat` and `StrCat`. This example uses the `Weight` storage, which stores the counts for each bin as well as the uncertainty on the count. This is typically the most useful storage type for particle physics.

If you have some data in the form of ex. a `numpy` or flat `awkward` array, you can fill a histogram using the `fill` function, which takes arrays for each dimension of the histogram followed by an optional weight if your data are weighted.

```py
my_onedim_hist.fill(photon_pt, weight=weight)
my_twodim_hist.fill(photon_pt, photon_eta, weight=weight)
```

There are various manipulations you can perform with histograms. For example, you can add or subtract two histograms using the regular python `+`/`-`. You can also scale the histogram yields with the `*` and `/` operators, which is very common for normalizing histograms

```py
my_normalized_hist = my_hist / my_hist.sum().value
```

You can find more information about the types of manipulations you can perform in the hist documentation.

To actually create a visualization of a histogram, we will use the `mplhep` library, which is a wrapper around a library called `matplotlib`. You can find more information on their respective websites [here](https://mplhep.readthedocs.io/en/latest/) and [here](https://matplotlib.org/).

```py
import matplotlib.pyplot as plt
import mplhep as mh
```

For a 1D histogram, a typical workflow might looks like the following code.

```py
fig, ax = plt.subplots()
mh.histplot(my_onedim_hist, ax=ax, label='Data', histtype='step')
ax.legend(loc='upper right')
ax.set_xlabel(r'$p_{\mathrm{T}}$ [GeV]')
ax.set_ylabel('Events/2 GeV')
plt.savefig('plots/myplot.pdf')
```

We define a new plot with `plt.subplots()`, then plot our histogram with the `mh.histplot` function, to which you can provide either an numpy histogram unrolled with the `*` operator (`*np.histogram(...)`) or a `hist` histogram. In this example we've used the `'step'` plotting style, which plots the histogram as an outline, but the `'fill'` and `'errorbar'` styles are also commonly used. You can plot multiple histograms on the same plot by simply making multiple calls to `mh.histplot`. We then set our legend position, add x- and y-axis labels, and save the plot to a file.

In particle physics, 2D histograms are most commonly displayed as color maps. You can use the `mh.hist2dplot` for this.

```py
fig, ax = plt.subplots()
mh.hist2dplot(my_twodim_hist, ax=ax, cbar=True)
ax.set_xlabel(r'$p_{\mathrm{T}}$ [GeV]')
ax.set_ylabel(r'$\eta$')
plt.savefig('plots/my_twodim_plot.pdf')
```

It is very important in science to label your plots with axes labels, and legend in the cases where for example is there is more than one histogram in the same plot.

You can set logarithmic axes using the `set_xscale` and `set_ysclae` methods on the axis object from `plt.subplots`.

```py
ax.set_xscale('log')
ax.set_yscale('log')
```

You can also set the range of the y-axis with the `set_ylim` function.

```py
ax.set_ylim(0, 1.4)
```

<!-- TODO add ratio plot info, more styling? -->

## Reading and writing data with uproot and coffea

In this section we will assume you have imported the following libraries:

```py
import uproot as ur
```

Most data in high energy physics is stored as "TTree" objects in **root** files. The `uproot` library provides a way to read and write ROOT files. You can open a file by using

```py
root_file = ur.open(filename)
# use file below
```

or in a block with

```py
with uproot.open(filename) as root_file:
  # use file in this block
```

You can see the named data in the file using `root_file.keys()` and access a given data structure using the `[]` operator. For example, if the file has a TTree named `'tree'`, you can access it via `root_file['tree']` and check the type via `type(root_file['tree'])`.

TTrees consist of columnar data analogous to a awkward record. You can see the columns using `my_tree.keys()`. We will typically extract the TTree data into an awkward array. You can get an awkward array with the data for a single column (also called a TBranch in root data) of a TTree with

```py
tree = root_file['tree']
column_data = tree['columnname'].array()
```

You can get an awkward record with the arrays for many columns at once using

```py
event_data = tree.arrays(['column1','column2','column3'])
```

If you do not provide a list of columns, the `TTree.arrays` function will return a record with all columns, which can be quite large depending on the data set. You can also retrieve the data as a numpy array/dictionary of arrays or as a pandas series/dataframe using the optional argument `library='np'` or `library='pd'` for the `TBranch.array` and `TTree.arrays` functions.

Since collider data, even in very reduced formats, can still consist of many TBs, you frequently will not be able to read all of the data into memory at once. You can select only specific columns, but you also load only specific rows using the optional `entry_start` and `entry_stop` arguments to `TBranch.array` or `TTree.arrays`. 

```py
# just load the first 1000 events
# if you do not specify the columns, all columns will be loaded
event_data = tree.arrays(entry_stop=1000)
```

Once you have the data as an awkward array/record, you can easily inspect it in the usual way. For example, if your data has a column called `jet_pt`, you can inspect the `jet_pt` for the `n`th event using

```py
print(tree.jet_pt[n])
```

In particle physics, we often work with very large data sets, which contain a lot of information in order to be useful for people doing many different type of data analyses. For a given analysis, it is common to only need a small subset of the data, hence one typically creates reduced data sets by removing rows/events (skimming), removing columns (slimming), or removing particles that are not interesting in a given row/column (thinning). This can be done with the various awkward filtering mechanisms mentioned above. To save the reduced data set to a root file, one can use the `uproot`'s `mktree` method, which takes a name for the tree together with a dictionary of (awkward) arrays that comprise the column data.

```py
with ur.recreate(filename) as output_file:
  output_file.mktree('tree', {'column1' : column1_data, 'column2' : column2_data})
```

You can find more information about `uproot` on its [documentation page](https://uproot.readthedocs.io/en/latest/basic.html).

## Exercise 2.2. A first look at NanoAOD

Try opening the CMS open data file 

```
opendata_file = ur.open('root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/280000/6293BAC8-2AB6-4A4A-BFEA-83E328B9C44F.root')
```

This is a file in the NanoAOD format we will explore more later. This file contains a TTree with the name `Events`. What columns are in this tree? Try reading some of the data with the `TTree.arrays` function, noting you may need to specify columns and `entry_stop` to fit it in your computer's memory. How many jets are in the 10th event? What are their tranvserse momenta $p_{\mathrm{T}}$ (this is a branch `Jet_pt`)?

### Scaling up with coffea

As mentioned, data analysis in particle physics typically involves data sets consisting of many files whose contents do not all fit in memory at once. You can use uproot and manually loop over files and subsets of data within the files using the `entry_start` and `entry_stop` methods, but their is another library that will automate this for us called `coffea`, documented [here](https://coffea-hep.readthedocs.io/en/latest/index.html). We will import various libaries as usual.

```py
import awkward as ak
import coffea
import coffea.processor
import hist
import matplotlib.pyplot as plt
import mplhep as mh
```

The first thing we will need to is write a processor, the analysis code that will be applied to each block of data. Let's try making histograms from the `Jet_pt` and `Jet_eta` branches

```py
def my_processor(events):
  # coffea will call this function on each subset of data for us
  # events will be an awkward record (with some metadata)

  dataset_name = events.metadata['dataset']
  
  jet_pt_hist = hist.Hist.new.Reg(50,30,200,name='pt').Weight()
  jet_eta_hist = hist.Hist.new.Reg(50,-4.8,4.8,name='eta').Weight()

  jet_pt_hist.fill(ak.flatten(events['Jet_pt']))
  jet_eta_hist.fill(ak.flatten(events['Jet_eta']))

  return { dataset_name: { 'jet_pt_hist' : jet_pt_hist,
                           'jet_eta_hist' : jet_eta_hist } }
```

The processor should return a python dictionary. In our example, we return a dictionary indexed by the dataset name retrieved from the metadata, whose elements are themselves dictionaries: the output histograms indexed by their names. The data in the dictionaries such as histograms and numbers will automatically be summed across all of the chunks of data processed by coffea.

Once we have a processor, we will need to specify the datasets to process. This is also done with a dictionary

```py
datasets = {'TTTo2L': {'treename' : 'Events',
                       'files': ['root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/280000/6293BAC8-2AB6-4A4A-BFEA-83E328B9C44F.root'],
                       'metadata': {'year': 2016}}
           }
```

In the format we will use, the dataset dictionary consists of dictionaries indexed by the dataset names. For the sub-dictionaries, we need to specify the file names as a list of strings, the name of the TTree in the files, and any other metadata we might want to pass to the processor.

We can now use `coffea` to run our processor on the specified data sets.

```py
runner = coffea.processor.Runner(
    executor=coffea.processor.IterativeExecutor(),
    schema=coffea.nanoevents.BaseSchema)

result = runner(datasets, processor_instance=my_processor)
```

The executor tells coffea how to run your code. We are just using `IterativeExecutor`, which just runs the code on your computer in a single thread. You can also try `FuturesExecutor`, which runs in multiple threads. The real power of coffea is its ability to distribute jobs across parallel computing systems, though we will not use that functionality in this tutorial.

The above code will run our processor over the data sets specified and store the result in `result`. We can then get the results out of `result` and plot them with `mplhep`.

```py
pt_fig, pt_ax = plt.subplots()
mh.histplot(result['TTTo2L']['jet_pt_hist'], ax=pt_ax, label='TT')
pt_ax.set_xlabel(r'$p_{\mathrm{T}}$ [GeV]')
pt_ax.set_ylabel('Events/3.4 GeV')
plt.savefig('plots/jet_pt_plot.pdf')

eta_fig, eta_ax = plt.subplots()
mh.histplot(result['TTTo2L']['jet_eta_hist'], ax=eta_ax, label='TT')
ax.set_xlabel(r'$\eta$')
ax.set_ylabel('Events/0.192')
plt.savefig('plots/jet_eta_plot.pdf')
```

## The structure and analysis of collider data 

### The structure of collider data

We will look at the data from general purpose collider experiments like ATLAS and CMS. In the future, this guide may be expanded with data from other types of experiments such as heavy flavor and neutrino experiments. You can find ATLAS and CMS open data at the [CERN open data portal](https://opendata.cern.ch/).

Most analyses at experiments like CMS and ATLAS rely on **physics objects**, which serve as reconstructed proxies to high momentum fundamental particles. The correspondance is given in the following table.

| Particle      | Physics object                  |
|---------------|---------------------------------|
| Electron      | Electron                        |
| Muon          | Muon                            |
| Tau           | Electron, Muon, or Hadronic tau |
| Photon        | Photon                          |
| Up quark      | Jet                             |
| Down quark    | Jet                             |
| Strange quark | Jet                             |
| Charm quark   | (c) Jet                         |
| Bottom quark  | (b) Jet                         |
| Gluon         | Jet                             |
| Neutrino      | Missing transverse moomentum    |
| W Boson       | Combination of above            |
| Z Boson       | Combination of above            |
| Higgs Boson   | Combination of above            |
| Top quark     | Combination of above            |

Muons are relatively stable and simply pass through the detectors, leaving tracks that can be reconstructed as a muon physics object. Electrons are similarly stable, though they more commonly radiate away some of their energy as photons when passing through matter due to a process called bremsstrahlung. Photons are also relatively stable, but have some probability to convert into electron pairs. As a result, electron and photon physics objects are reconstructed as clusters of detected electrons and photons.

Taus are relatively unstable and quickly decay in one of three ways: into a pair of neutrinos and an electron (about 1/6 probability), a pair of neutrinos and a muon (about 1/6 probability), or a neutrino and a small collection of hadrons (composite particles) (about 2/3 probability). As neutrinos are not detected in these experiments, the first two cases result in the tau appearing as an electron or muon physics object. In the last case, the detected hadrons can be reconstructed a "hadronic tau".

The particles that interact through the strong interaction are called quarks and gluons. Due to the strong interaction, a high-energy quark or gluon will typically radiate away much of its energy as additional quarks and gluons after being produced. This cluster of quarks and gluons will then form composite particles, hadrons, and the resulting cluster of hadrons (and hadron decay products) is called a QCD jet. Reconstructed jets are thus used as a proxy to a high-energy quark or gluon, excluding the top quark, which decays before it is able to form a coherent jet. The jets from different flavors of quarks and gluons are typically hard to discriminate (it is tricky to even rigorously define jet flavor). However, jets from bottom quarks, and to a lesser extent charm quarks, can be discriminated from other jets. Jets from bottom and charm quarks are called b jets and c jets respectively.

Though neutrinos are not visible to the detectors employed in these experiments, the presence of neutrinos can be inferred using conservation of momentum. In general, we do not know the energy/momentum of the colliding quarks/gluons inside of the proton since the proton's energy/momentum will be randomly divided amongst its constituents. However, we do know that the collisions are roughly head-on, meaning that the momentum of the colliding particles is roughly 0 in the plane transverse to the collision. Based on the particles detected emerging from the collision, we can thus use conservation of momentum to calculate the missing transverse momentum ($p\_\mathrm{T}^\mathrm{miss}$), which can be attributed to particles that are not detected such as neutrinos. Small amounts of missing transverse momentum will be generated by mismeasurement, but large missing transverse momentum is typically indicative of neutrino production. Missing transverse momentum is also sometimes called "MET" in data sets for historical reasons.

Finally, the W boson, Z boson, Higgs boson, and top quark have very short lifetimes ($< 10^{-20}$ s) and are typically reconstructed via their decays into some combination of the physics objects above. W bosons have about a 2/3 probability to decay into two quarks, generating two jets, and about a 1/3 probability of decaying into a charged lepton (electron, muon, or tau) plus a neutrino. Z bosons decays into a pair of quarks with about a 70% probability, into a pair of neutrinos with about a 20% probability, and into a pair of charged leptons with about a 10% probability. Top quarks almost always decay into a bottom quark and a W boson. Finally, the Higgs boson has a rather complicated set of [decay channels](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-higgs-boson.pdf), with five having been observed so far: two photons (0.2%), two Z bosons (2.6%), two W bosons (21%), two taus (6.3%), and two bottom quarks (58%). Simulated samples are often divided based on the heavy particles (W, Z, top, and Higgs) in the process; hard scattering processes without heavy particles are dominated by events with high-energy quarks and gluons and are "QCD multijet". The vast majority of proton-proton collisions do not involve any hard scattering and are sometimes called "minimum bias" events.

As a final note, particle physicists often do not distinguish between particles and antiparticles, so an "electron" really means an electron or an antielectron. At the analysis level, one might requires one to have negative charge and the other to have positive charge, but both particles are referred to as "electrons" in practice.

### Data sets and triggers

As mentioned in the previous section, simulated data sets (typically called "Monte Carlo" or "MC" in particle physics jargon) are generally split by the heavy particles (W, Z, top, and Higgs) simulated. The rough cross sections for some processes are given for 13 TeV proton-proton collisions. Cross sections are just a measure of probability: you can convert between the two by dividing the cross section for a given process by the total cross section for the given process and energy. For example, the total cross section for proton-proton collisions at 13 TeV is about 100 mb.

| Process      | Cross section | Probability       |
|--------------|---------------|-------------------|
| All          | 100 mb        | 1                 |
| QCD Multijet | 200 $\mu$b    | 1 in 500          |
| W            | 100 nb        | 1 in 1 million    |
| Z            | 50 nb         | 1 in 2 million    |
| ttbar        | 900 pb        | 1 in 111 million  |
| single top   | 200 pb        | 1 in 500 million  |
| WW           | 80 pb         | 1 in 1.2 billion  |
| H            | 50 pb         | 1 in 2 billion    |
| WZ           | 30 pb         | 1 in 3 billion    |
| ZZ           | 15 pb         | 1 in 6 billion    |

The vast majority of collisions (sometimes called minimum bias) are soft scattering where no high $p_\mathrm{T}$ particles and thus no hard physics objects are produced. Since the strong interaction is considerably stronger than the electroweak interactions, the vast majority of hard scattering events are QCD multijet events, where one observes high $p_\mathrm{T}$ quarks, gluons, and occaisionally photons. Since the colliding quarks or gluons can always emit gluons, quarks, or photons, any number of jet and photon physics objects can be produced in any process.

The rarer "electroweak" processes such as W or Z production are often distinguished from the large QCD multijet background by requiring electron, muon, or hadronic tau physics objects, or by large missing transverse momentum from neutrinos. Having said this, the majority of W and Z bosons due decay into quarks, producing jets that are hard to distinguish from the large QCD multijet background.

The processes studied in collider physics are quite rare, with probabilities often on the scale of 1 in a billion or smaller (the process HH$\to\mathrm{b}\overline{\mathrm{b}}\gamma\gamma$ that researchers are currently searching for has a probability less than 1 in a quadrillion!). In order to be able to perform these studies, the collider provides a very high rate of collisions averaging over a billion per second.

The high rate of collisions is a major challenge for experiments. Each collision can produce tens or even hundreds of particles, meaning that the data rate produced by the experiment (after zero-suppression where readout is only performed for detectors that actually detected a particle) is around 40 TB/s currently, and will increase by an order of magnitude in the future. Since only a few GB/s of data can be written to disk, the vast majority of data must be discarded. Luckily, the processes that we are the most interested are very rare so we can try to save just the data corresponding to the interesting processes. The system that does this is called the **trigger system**.

The trigger system uses criteria called triggers to decide which events to save and which to discard, typically using the presence and properties of physics objects as criteria. For rarer particles like electrons/muons/taus the trigger thresholds can be looser and most events with high $p_\mathrm{T}$ charged leptons are saved. In contrast, the high rate of QCD multijet events and the high amounts of spurious missing transverse momentum ($p_\mathrm{T}^\mathrm{miss}$) from mismeasurement mean that the thresholds on jet triggers and $p_\mathrm{T}^\mathrm{miss}$ triggers are much higher so only events with high $p_\mathrm{T}$ (or a large number of) jets or large $p_\mathrm{T}^\mathrm{miss}$ are saved. 

Actual data (called "data" in particle physics jargon, contrast with simulation) is generally split into datasets based on which triggers collected the data set such as "SingleMuon", "SingleElectron", "JetMET", etc.

### Data analysis of collider data

In this section we will assume you have imported the following libraries

```py
import awkward as ak
improt matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
import uproot as ur
```

A typical data analysis workflow involves reading the data, calculating new quantities, filtering the data, and producing a human-understandable output such as a (table) numbers or histograms. The code below shows a simple example:

```py
# get input
dataset_file = ur.open('intro_tree.root')
events = dataset_file['Events'].arrays()

# calculate the number of electrons passing ID criteria
events['n_good_el'] = ak.sum(events['Electron_mvaFall17V2Iso_WP90'], axis=-1)

# make histogram of number of good electrons
fig, axis = plt.subplots()
hist = np.histogram(events.n_good_el,4,(-0.5,3.5))
mh.histplot(*hist,
    label='Mystery dataset',
    ax=axis)
axis.set_xlabel('# Electrons (WP90)')
axis.set_ylabel('# Events')
plt.savefig('plots/mystery_nel.pdf')
```

Histograms are ubiquitous in high-energy physics since they are a basic representation of the data projected onto one dimension that can easily be compared with appropriate probability densities (see section on probability). You can reduce a set of data into a histogram using the `np.histogram` function:

`hist = np.histogram(data, nbins, (lower_bound, upper_bound))`

You can then plot this histogram using the `matplotlib` library. We will use the `mplhep` library as a wrapper around `matplotlib`. Using `mplhep`, we can plot a histogram `hist` with just a few lines, as in the previous example.

```py
fig, axis = plt.subplots()
mh.histplot(*hist,
    label='Mystery dataset',
    ax=axis)
axis.set_xlabel('# Electrons (WP90)')
axis.set_ylabel('# Events')
plt.savefig('plots/mystery_nel.pdf')
```

## Machine learning with xgboost and pytorch

This section doesn't exist yet. In the meanwhile, you can reference [this tutorial](https://hsf-training.github.io/deep-learning-intro-for-hep/00-intro.html).
<!-- also optuna -->

# Probability and statistics

## Concepts of probability and statistics

## Statistical models and fitting
