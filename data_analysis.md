# Data analysis for particle physics

This page is a tutorial on data analysis for particle ("high-energy") physics. It is a work-in-progress and thus may experience unexpected reorganization. 

It is forseen that this tutorial will cover programming in Python; libraries for data science such as numpy, matplotlib, numba, pytorch, etc.; and some probability and statistics. All topics will be covered with applications to particle physics. Most of this content does not yet exist.

1. [Programming in Python](#programming-in-python)

    1.1. [Getting started](#getting-started-and-documentation-in-science)

    1.2. [Expressions and variables](#expressions-and-variables-in-python)

    1.3. [Functions and control](#functions-and-control-in-python)

    1.4. [Classes and libraries](#classes-and-libraries-in-python)

2. [Data analysis libraries](#data-analysis-libraries)

    2.1. [numpy and awkward](#getting-started-with-numpy-and-awkward)

    2.2. [uproot](#reading-and-writing-data-with-uproot)

    2.3. [Collider data](#the-structure-of-collider-data)

    2.4. [Analysis of collider data](#data-analysis-of-collider-data)

    2.5. [Scaling up with coffea](#scaling-up-with-coffea)

    2.5. [xgboost and pytorch](#machine-learning-with-xgboost-and-pytorch)

3. [Probability and statistics](#probability-and-statistics)

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
conda install awkward-pandas
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
| not a     | bool        | bool        | inverts true and false              |
| a or b    | bool, bool  | bool        | true iff `a` or `b` is true         |
| a and b   | bool, bool  | bool        | true iff `a` and `b` are true       |
| a & b     | int, int    | int         | bitwise and                         |
| a | b     | int, int    | int         | bitwise or                          |
| ~ a       | int         | int         | bitwise not                         |

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
import numpy
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

## Getting started with numpy and awkward

Interpreting python statements is actually rather slow, so we would like to manipulate large amounts of data with just a few commands. This is where the `numpy` and `awkward` libaries come in. These libraries allow us to do "vectorized" operations across large amounts of data in just one statement. Throughout this section, we will assume we have imported `numpy` and `awkward` as follows:

```py
import numpy as np
import awkward as ak
```

The basic class introduced by numpy is the `np.ndarray` (alias `np.array`). This acts similarly to a Python list, except that the data must all be the same type, i.e. `[3.0, 'Hello']` is a valid Python list, but `np.array([3.0, 'Hello'])` will throw an error. The advantage of numpy lists is that operations are "vectorized" so

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

As demonstrated above, "vectorized" here does not mean in the math sense of the word vector. Rather, operations like `+`, `-`, `*`, `/`, `==`, `<`, and so forth are performed component-wise for each component in the array. Due to limitations in Python, boolean operators like `and`, `or`, and `not` can't be used, though for *booleans* specifically, you can use their bitwise versions `&`, `|`, and `~`.

Arrays in numpy can have any number of dimentions. You can access a component of a numpy array using the `[]` operator.

```py
x = np.array([[1, 2], [3, 4], [5, 6]])
print(x[0, 0])
```

As shown in this example, when working with multidimensional arrays, multiple indices can be provided separated by a comma. The first index corresponds to the outermost array, while the last index corresponds to the innermost array.

You can also use the range operator `:` to select a range of indices.

```py
print(x[1:3, :])
```

The range operator `:` has a default starting point of the first element (0) and a default ending point after the last element when not specified, meaning that `:` by itself with select all indices along a given axis.

An important feature of numpy is **broadcasting** whereby a user can perform operations on two objects, even if they do not have the same shape (i.e. are not arrays of the same size). The most common usage of this is to broadcast a scalar value to an array. For example, the code below adds 2 to *each* component of the array `x`.

```py
x = x + 2
```

Broadcasting is frequently combined with boolean indexing. If `x` is an array and `y` is a boolean array of the same shape, then `x[y]` refers to the components of `x` for which `y` is true. As an example, the following code subtracts 2 from all components of an array `x` that are greater than 4.

```py
# create boolean array with broadcasting
# greater_than_four will have the same shape as x
greater_than_four = x > 4

x[greater_than_four] -= 2
```

`numpy` functions like `np.sqrt()`, `np.absolute()`, etc. can be used to perform component-wise operations on `numpy` or `awkward` arrays.

Awkward array or `awkward` serves as an extension to `numpy` for dealing with data that are not rectangular arrays. This is common in particle physics where the data might consist of a collection of particles for each "event", but the number of particles is different for each event.

`awkward`'s array class is just called `ak.Array` and can be used just like `np.array` in terms of providing vectorized operations, indexing, broadcasting, etc.

```py
x = ak.Array([[1.0, 8.7, 5.7], [4.8], [7.0, 1.5]])
y = ak.Array([[3.1, 8.0, 1.2], [7.8], [5.3, 0.2]])

z = x + y
```

Boolean indexing is commonly used with awkward arrays for masking/filtering.

```py
z = z[z > 8.0]
```

We will often work with awkward Records. You can think of a record as a dictionary/class whose elements are awkward arrays.

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

The advantage of using a record is that you can filter all the awkward arrays contained in the record in a single statement.

```py
electrons_pt = np.sqrt(electrons.px**2 + electrons.py**2)

highpt_electrons = electrons[electrons_pt > 4.0]
```

You can also easily add new data to an existing record using the `[]` operator.

```py
electrons['quality'] = electron_quality
```

There are several useful functions for working with `awkward` arrays. The `ak.sum()` function will replace a given dimension of an array with the sum over it's components. This is particularly useful when used with `axis=-1` (the default), which will sum over the inner most index. When provided booleans, sum will treat `True` as `1` and `False` as `0`. This allows one to count the number of elements meeting certain criteria. For example, if `electron_pt` is the transverse momenta of electrons (inner index) by event (outer index), we can get the number of electrons in each event with transverse momentum greater than 20 as an array with `ak.sum`.

```py
ak.sum(electron_pt > 20, axis=-1)
```

<!-- masking -->
<!-- more on awkward -->

You can find more information about `numpy` and `awkward` in their [respective](https://numpy.org/doc/stable/) [documentation](https://awkward-array.org/doc/main/index.html) pages.

## Reading and writing data with uproot

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

TTrees consist of columnar data analogous to a pandas dataframe or an awkward record. You can see the columns using `my_tree.keys()`. We will typically extract the TTree data into an awkward array. You can get an awkward array with the data for a single column (also called a TBranch in root data) of a TTree with

```py
tree = root_file['tree']
column_data = tree['columnname'].array()
```

You can get an awkward record with the arrays for many columns at once using

```py
event_data = tree.arrays(['column1','column2','column3'])
```

If you do not provide a list of columns, the `TTree.arrays` function will return a record with all columns, which can be quite large depending on the data set. You can also retrieve the data as a numpy array/dictionary of arrays or as a pandas series/dataframe using the optional argument `library='np'` or `library='pd'` for the `TBranch.array` and `TTree.arrays` functions.

Once you have the data as an awkward array/record, you can easily inspect it in the usual way. For example, if your data has a column called `jet_pt`, you can inspect the `jet_pt` for the `n`th event using

```py
print(tree.jet_pt[n])
```

<!-- output file.mktree('treename',tree)? -->

You can find more information about `uproot` on its [documentation page](https://uproot.readthedocs.io/en/latest/basic.html).

## The structure of collider data 

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

Though neutrinos are not visible to the detectors employed in these experiments, the presence of neutrinos can be inferred using conservation of momentum. In general, we do not know the energy/momentum of the colliding quarks/gluons inside of the proton since the proton's energy/momentum will be randomly divided amongst its constituents. However, we do know that the collisions are roughly head-on, meaning that the momentum of the colliding particles is roughly 0 in the plane transverse to the collision. Based on the particles detected emerging from the collision, we can thus use conservation of momentum to calculate the missing transverse momentum ($p\_\mathrm{T}^\mathrm{miss}$), which can be attributed to particles that are not detected such as neutrinos. Small amounts of missing transverse momentum will be generated by mismeasurement, but large missing transverse momentum is typically indicative of neutrino production.

Finally, the W boson, Z boson, Higgs boson, and top quark have very short lifetimes ($< 10^{-20}$ s) and are typically reconstructed via their decays into some combination of the physics objects above. W bosons have about a 2/3 probability to decay into two quarks, generating two jets, and about a 1/3 probability of decaying into a charged lepton (electron, muon, or tau) plus a neutrino. Z bosons decays into a pair of quarks with about a 70% probability, into a pair of neutrinos with about a 20% probabiliyt, and into a pair of charged leptons with about a 10% probability. Top quarks almost always decay into a bottom quark and a W boson. Finally, the Higgs boson has a rather complicated set of [decay channels](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-higgs-boson.pdf), with five having been observed so far: two photons (0.2%), two Z bosons (2.6%), two W bosons (21%), two taus (6.3%), and two bottom quarks (58%). Simulated samples are often divided based on the heavy particles (W, Z, top, and Higgs) in the process; hard scattering processes without heavy particles are dominated by events with high-energy quarks and gluons and are "QCD multijet". The vast majority of proton-proton collisions do not involve any hard scattering and are sometimes called "minimum bias" events.

As a final note, particle physicists often do not distinguish between particles and antiparticles, so an "electron" really means an electron or an antielectron. At the analysis level, one might requires one to have negative charge and the other to have positive charge, but both particles are referred to as "electrons" in practice.

## Data analysis of collider data

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

You can plot multiple plots on the same axis by simply making multiple calls to `mh.histplot`.

**TODO** More information on matplotlib and mplhep

You can find more information about `matplotlib` and `mplhep` in their [respective](https://matplotlib.org/stable/api/pyplot_summary.html) [documentation](https://mplhep.readthedocs.io/en/latest/) pages.

<!-- matplotlib/mplhep somewhere -->

## Scaling up with coffea

## Machine learning with xgboost and pytorch

# Probability and statistics
