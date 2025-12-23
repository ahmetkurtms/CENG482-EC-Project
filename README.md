# CENG482-EC-Project
Genetic Algorithm-based solution to the Exam Timetabling Problem — CENG482 (Evolutionary Computation) Term Project 

## Running the GA demo
```
python3 -m venv .venv && source .venv/bin/activate
pip install numpy
PYTHONPATH=src python3 src/ga_solver.py
```
The defaults use the `car-s-91` dataset with a 35-slot cap; adjust parameters in `src/ga_solver.py` if you want to tune population size, generations, or the timeslot limit.
