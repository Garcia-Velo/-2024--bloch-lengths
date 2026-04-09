# entanglement-optimization

## Project structure

entanglement-optimization/

```txt
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

- bloch.py: Functions regarding the global and local Bloch vector and the computation of the Bloch lengths.
- initialization.py: Functions required for initializing and parametrizing the density matrices for given Bloch lengths.
- quantum.py: Functions to generate and validate density matrices, and implementing several entanglement measures.

### notebooks

- space_exploration.ipynb: Optimization of the entanglement measures restrained to the Bloch lengts.

#### test

- bloch.ipynb: Some tests to check that the functions on bloch.py work propperly.
- initialization.ipynb: Some tests to check that the functions on initialization.py work propperly.
- quantum.ipynb: Some tests to check that the functions on quantum.py work propperly.