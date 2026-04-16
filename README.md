# [2024-] Moments optimization

This project aims to find the quantum states with maximum and minimum entanglement, given the set of second-order moments or Bloch lengths.
The code is structured to work with systems of any number of parties and local dimension.

## Project structure

```txt
entanglement-optimization/
├── .venv/
├── src/
│   ├── moments/
│   │   ├── __init__.py
│   │   ├── bloch.py
│   │   ├── initialization.py
│   │   ├── quantum.py
├── notebooks/
│   ├── space_exploration.ipynb
│   ├── test/
│   │   ├── bloch.ipynb
│   │   ├── initialization.ipynb
│   │   ├── quantum.ipynb
├── data/
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
```

- .gitignore: Folders and files that git should ignore for commiting and pushing.
- pyproject.toml: Configuration file for the local instalation of the package moments.
- README.md: Markdown file explaining the project.
- requirements.txt: Dependencies for all libraries used.

### src/moments

This folder contains reusable code for the different Jupyter notebooks.

- bloch.py: Functions to manage the global and local Bloch vector and the computation of the Bloch lengths.
- initialization.py: Functions required for generating and parametrizing density matrices with given Bloch lengths.
- quantum.py: Functions to generate and validate density matrices, and implementing several entanglement measures.

### notebooks

This folder contains all Jupyter notebooks for all experiments.

- space_exploration.ipynb: Optimization of the entanglement measures restrained to the local Bloch lengts.

#### test

This folder is inside "notebooks/" and implements tests to check that auxiliary code works properly. The test are incomplete but guive a rough idea.

- bloch.ipynb: Some tests to check that the functions on bloch.py work properly.
- initialization.ipynb: Some tests to check that the functions on initialization.py work properly.
- quantum.ipynb: Some tests to check that the functions on quantum.py work properly.

## To-Do

In space_exploration.ipynb you can find the first tests of optimizing entanglement over the set of density matrices parametrized by Bloch lengts. For two qubits, the inequalities that define this region (can be seen in "space_exploration.ipynb" and "initialization.ipynb") are the following:

$$
x, y, z \in [0, 1] \times [0, 1] \times [0, \sqrt{3}] \, , \quad z \ge x + y - 1 \, , \quad z^2 \le 3 + x^2 + y^2 - 4xy - 4|x-y| \, .
$$

- First, I have run a test for the case $x=y$, which corresponds to the intersection between the region of interest and the plane that bisects the first quadrant passing through the z-axis. Right now, the code is failing to optimize in around $5 %$ of points for both the maximization and minimization. So one step would be to detect which points the algorithm fails to optimize, and try to improve the code to solve those points.

$$
x, z \in [0, 1] \times [0, 1] \times [0, \sqrt{3}] \, , \quad z \ge 2x - 1 \, , \quad z^2 \le 3 - 2x^2 \, .
$$

- Then, it will also be necessary to extend the code, not only to the $x=y$ plane, but also to the rest of the region. In the $2$-qubit case, there is a symetry with respect to changes of the local Hilbert spaces. Thus, the space of quantum states will be symetric with respect to the $x = y$ plane. As a consequence, it is only necessary to iterate over half of the state space ($x \geq y$ or $x \leq 0$), with respect to this plane, in order to classify the hole space.

$$
x, y, z \in [0, 1] \times [0, 1] \times [0, \sqrt{3}] \, , \quad z \ge x + y - 1 \, , \quad z^2 \le 3 + x^2 + y^2 - 4xy - 4|x-y| \, , x \geq y \, .
$$