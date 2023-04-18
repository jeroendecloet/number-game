# Number game

This repository contains the solver for two type of games:

- Make-24 game: Given four positive integers (1-9), try to make 24.
- Seven sevens game: Given seven sevens, try to make positive integers (0, 1, 2, ...).

The possible solutions are given as equations. For example, making 24 with 1, 2, 5 and 8 gives the following solutions:

- `((5 - (1 * 2)) * 8)`
- `(((1 + 5) / 2) * 8)`
- ...


### Dependencies and installation
This repository only depends on `numpy`, so a simple Python environment with numpy should suffice. 

# Games CLI
This repository contains a command-line interface (`cli.py`) for easy use.

## Make 24
The first argument for the CLI should be `ng24`. This refers to 'number game 24'. Next, the four numbers should be given. For example:
```commandline
>>> python cli.py ng24 1 5 5 6
['(((1 + 5) * 5) - 6)' '(((5 * 6) - 1) - 5)' '((5 * 6) - (1 + 5))'
 '(((5 * 6) - 5) - 1)']
```
The CLI returns a list of solutions (strings). 

Another options is to give in one integer consisting of four numbers, e.g.:
```commandline
>>> python cli.py ng24 1556
['(((1 + 5) * 5) - 6)' '(((5 * 6) - 1) - 5)' '((5 * 6) - (1 + 5))'
 '(((5 * 6) - 5) - 1)']
```
Under the hood, the integer is split into four numbers which are passed to the solver.

## Seven sevens
The first argument for the CLI should be `ng77s`. This refers to 'number game seven sevens'. Next, the start and end integers should be given. For example:
```commandline
>>> python cli.py ng77s 0 5
{0: array(['((((((7 * 7) * 7) / 7) / 7) - 7) * 7)'], dtype='<U40'), 
 1: array(['((((((7 * 7) * 7) / 7) + 7) / 7) - 7)'], dtype='<U40'), 
 2: array(['((((((7 * 7) * 7) / 7) / 7) + 7) / 7)'], dtype='<U40'), 
 3: array(['((((((7 * 7) + 7) + 7) + 7) / 7) - 7)'], dtype='<U40'), 
 4: array(['((((((7 * 7) / 7) + 7) + 7) + 7) / 7)'], dtype='<U40'), 
 5: array(['((((((7 * 7) * 7) / 7) - 7) - 7) / 7)'], dtype='<U40')}
```
This will calculate the solutions for 0, ..., 5. By default, the start is 0, so `python cli.py ng77s 5` would give the same results.

Note that the calculation takes some time (~ 30 seconds). This is because it calculate all possible solutions. However, the solutions cannot be stored. For multiple request, consider accessing the underlying class directly:
```python
import time
from number_game import games
seven_sevens = games.SevenSevensGame()

# The first iteration, it will calculate all solutions
start = time.time()
_ = seven_sevens(range(10))
print("First time: ", time.time() - start)  # First time: 23.51871418952942

# The following iterations are only look-ups of the solutions and thus much faster.
start = time.time()
_ = seven_sevens(range(100))
print("Second time: ", time.time() - start)  # Second time: 0.0036156177520751953
```
