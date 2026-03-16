# Missing ToDos

- [ ] Check / Debug for errors turning ratio $\varphi$: Currently problematic due to shape errors with torch / numpy
- [ ] Investigate problems / errors with static ZIP loads
- [ ] Add (rough) documentation

- [ ] Re-engineer modular backend torch vs numpy: no wildcard import (maybe with a class Backend as Wrapper around the choosable numpy or python?)
- [ ] Initialization of Models with correct and robust arguments (**kwargs, **parameters, and individual passing)


---
# Paper Replication Package

- [ ] Include Validation
- [ ] Include Application Study
- [ ] Include Theoretical derivations
- [ ] Do as scripts and notebooks

---
# Future Improvements

## Code Quality

1. Pre-commit hooks
     - [ ] pylint
     - [ ] mypy
2. Unified Type annotations
3. GitHub Workflows
     - [ ] Automatic Release strategy
     - [ ] Automatic Documentation Generation / Host

## Testing

- [ ] Write additional test functions
- [ ] Set-up adequate testing suite

## Functionalities

Ideas for further functionalities:

1. Induction Machines
2. ZIP Load model
3. Improve holding details about all the grid elements; Output of variables (choosable dict or pandas DataFrame?)
4. Make addition of customizable controllers possible
5. Extend module with static library: More / better load flow calculation methods, OPF approaches, ...

6. Make negative and zero sequence calculation possible
7. Add converters, to load grids from e.g. pandapower