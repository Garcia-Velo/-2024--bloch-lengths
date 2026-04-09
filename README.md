# [2024-] Moments Optimization

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