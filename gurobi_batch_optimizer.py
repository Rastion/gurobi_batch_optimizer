import gurobipy as gp
from gurobipy import GRB
from qubots.base_optimizer import BaseOptimizer
from collections import defaultdict

class GurobiBatchOptimizer(BaseOptimizer):
    """
    Batch scheduling optimizer using Gurobi.
    
    This model creates start and end time variables for each task, enforces non-overlap
    constraints for tasks sharing the same resource (via binary ordering variables), and 
    applies any precedence constraints. The objective is to minimize the makespan.
    """
    
    def __init__(self, time_limit=5):
        self.time_limit = time_limit

    def optimize(self, problem, initial_solution=None, **kwargs):
        nb_tasks = problem.nb_tasks
        time_horizon = problem.time_horizon
        durations = problem.duration

        model = gp.Model("batch_scheduling")
        model.Params.TimeLimit = self.time_limit
        # Optionally disable output:
        # model.Params.OutputFlag = 0

        # Create variables for task start and end times.
        starts = {}
        ends = {}
        for t in range(nb_tasks):
            starts[t] = model.addVar(lb=0, ub=time_horizon, vtype=GRB.INTEGER, name=f"start_{t}")
            ends[t] = model.addVar(lb=0, ub=time_horizon, vtype=GRB.INTEGER, name=f"end_{t}")
            # Enforce: end = start + duration
            model.addConstr(ends[t] == starts[t] + durations[t], name=f"duration_{t}")

        # Non-overlap constraints for tasks sharing the same resource.
        resource_tasks = defaultdict(list)
        for t in range(nb_tasks):
            resource = problem.resources[t]
            resource_tasks[resource].append(t)

        M = time_horizon  # Big-M constant.
        # For each pair of tasks on the same resource, create a binary decision variable
        # that determines their order.
        for resource, tasks in resource_tasks.items():
            for i in range(len(tasks)):
                for j in range(i+1, len(tasks)):
                    t1, t2 = tasks[i], tasks[j]
                    y = model.addVar(vtype=GRB.BINARY, name=f"y_{t1}_{t2}")
                    # If y == 1 then t1 precedes t2; if y == 0 then t2 precedes t1.
                    model.addConstr(starts[t1] + durations[t1] <= starts[t2] + M * (1 - y),
                                    name=f"order_{t1}_{t2}_1")
                    model.addConstr(starts[t2] + durations[t2] <= starts[t1] + M * y,
                                    name=f"order_{t1}_{t2}_2")

        # Precedence constraints (if any).
        for t in range(nb_tasks):
            for s in problem.successors[t]:
                model.addConstr(ends[t] <= starts[s], name=f"prec_{t}_{s}")

        # Define makespan as the maximum end time among all tasks.
        makespan = model.addVar(lb=0, ub=time_horizon, vtype=GRB.INTEGER, name="makespan")
        for t in range(nb_tasks):
            model.addConstr(makespan >= ends[t], name=f"makespan_{t}")

        model.setObjective(makespan, GRB.MINIMIZE)

        model.optimize()

        # Check if we have an optimal or feasible solution.
        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return self._extract_solution(model, problem, starts, ends)
        else:
            return problem.random_solution(), float('inf')

    def _extract_solution(self, model, problem, starts, ends):
        """
        Extracts the solution from the Gurobi model and groups tasks into batches
        per resource based on their start times.
        """
        nb_tasks = problem.nb_tasks
        batches = defaultdict(list)
        for t in range(nb_tasks):
            resource = problem.resources[t]
            s = int(round(starts[t].X))
            e = int(round(ends[t].X))
            batches[resource].append((s, e, t))
        
        batch_schedule = []
        for resource, tasks in batches.items():
            # Sort tasks by start time.
            sorted_tasks = sorted(tasks, key=lambda x: x[0])
            current_batch = []
            current_end = -1
            
            for s, e, t in sorted_tasks:
                if s >= current_end:
                    # If there's a gap, finalize the previous batch.
                    if current_batch:
                        batch_schedule.append({
                            'resource': resource,
                            'tasks': [task for (_, _, task) in current_batch],
                            'start': current_batch[0][0],
                            'end': current_batch[-1][1]
                        })
                    current_batch = [(s, e, t)]
                    current_end = e
                else:
                    # Otherwise, add the task to the current batch.
                    current_batch.append((s, e, t))
                    current_end = max(current_end, e)
            
            if current_batch:
                batch_schedule.append({
                    'resource': resource,
                    'tasks': [task for (_, _, task) in current_batch],
                    'start': current_batch[0][0],
                    'end': current_end
                })
        return {'batch_schedule': batch_schedule}, model.objVal
