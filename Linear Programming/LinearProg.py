
# execute "pip install pulp" command in terminal or here

import pulp as p
import TensorFlow as tf

# Create the minimization problem for Linear program. It can also be maximized if needed
LP_optimize = p.LpProblem('Problem', p.LpMinimize)

# Create the variables with lower bound of 0
x = p.LpVariable("x", lowBound=0)  # Create a variable x >= 0
y = p.LpVariable("y", lowBound=0)  # Create a variable y >= 0

# Objective Function
LP_optimize += 3 * x + 5 * y

# Constraints:
LP_optimize += 2 * x + 3 * y >= 12
LP_optimize += -x + y <= 3
LP_optimize += x >= 4
LP_optimize += y <= 3

# Display the problem
print(LP_optimize)

status = LP_optimize.solve()  # Solver
print(p.LpStatus[status])  # The status of the solution

# Printing the final solution
print(p.value(x), p.value(y), p.value(LP_optimize.objective))