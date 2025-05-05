
import numpy as np
from numba import njit


# class ODESolver:
#     def __init__(self, options):
#         self.options = options
#     def solve(self): 
#         pass 


# @njit
def jacobian(fun, x, funx, *args, dx=1e-6):
    """Compute the Jacobian matrix using finite differences."""
    nx = len(x)
    nf = len(funx)
    J = np.zeros((nf, nx))

    for n in range(nx):
        delta = np.zeros(nx)
        delta[n] = dx
        dF = fun(x + delta, *args) - funx
        J[:, n] = dF.flatten() / dx  # Derivatives

    return J

# self.options = options if options else {'TolX': 1e-12, 'TolFun': 1e-6, 'MaxIter': 1000}
# self.TOLX = self.options['TolX']
# self.TOLFUN = self.options['TolFun']
# self.MAXITER = self.options['MaxIter']
# self.ALPHA = 1e-4  # Backtracking parameter
# self.MIN_LAMBDA = 0.1
# self.MAX_LAMBDA = 0.5

# self.iter_count = 0
# self.lambda_val = 1  # Backtracking initial lambda value

# @njit
def newton_raphson_solver(fun, 
                          x, 
                          x0,
                          init_lambda=1., 
                          tolx=1e-12,
                          tolfun=1e-6,
                          maxiter=1000, 
                          alpha=1e-4,
                          min_lambda=0.1,
                          max_lambda=0.5): 

    # Make initial guess
    F = fun(x, x0)  # Evaluate residuals at initial guess
    J = jacobian(fun, x, F, x0)
    
    resnorm = np.linalg.norm(F, np.inf)  # Compute residual norm
    resnorm0 = 100 * resnorm
    dx = np.zeros_like(x)

    exitflag = 1  # Normal exit
    if np.any(np.isnan(J)) or np.any(np.isinf(J)):
        exitflag = -1  # Jacobian is singular
        return x, F, exitflag

    # Iterative solver
    lambda_val = init_lambda
    iter_count = 0

    while (resnorm > tolfun or lambda_val < 1) and exitflag >= 0 and iter_count <= maxiter:
        if lambda_val == 1:
            iter_count += 1
            # Recalculate Jacobian if necessary
            if resnorm / resnorm0 > 0.2:
                J = jacobian(fun, x, F, x0)

            if np.linalg.cond(J) <= np.finfo(float).eps:
                dx = np.linalg.pinv(J) @ -F
            else:
                dx = np.linalg.solve(J, -F)

            g = np.dot(F, J.T)
            slope = np.dot(g, dx)
            fold = np.dot(F, F)
            xold = x.copy()
            lambda_min = tolx / np.max(np.abs(dx) / np.maximum(np.abs(xold), 1))
        
        if lambda_val < lambda_min:
            exitflag = 2  # Too small step
            break

        elif np.any(np.isnan(dx)) or np.any(np.isinf(dx)):
            exitflag = -1  # Step is NaN or Inf
            break

        # Update solution
        x = xold + dx * lambda_val
        F = fun(x, x0)
        f = np.dot(F, F)

        # Backtracking line search
        lambda1 = lambda_val 

        # Set these to fix numba compilation issues
        lambda2 = lambda_val
        f2 = f 

        if f > fold + alpha * lambda_val * slope:
            if lambda_val == 1:
                lambda_val = -slope / (2 * (f - fold - slope))
            else:
                A = 1 / (lambda1 - lambda2)
                B = np.array([[1 / lambda1 ** 2, -1 / lambda2 ** 2], [-lambda2 / lambda1 ** 2, lambda1 / lambda2 ** 2]])
                C = np.array([f - fold - lambda1 * slope, f2 - fold - lambda2 * slope])
                # coeff = np.linalg.solve(A @ B, C)
                coeff = np.linalg.solve(A*B, C)
                a, b = coeff[0], coeff[1]

                if a == 0:
                    lambda_val = -slope / (2 * b)
                else:
                    discriminant = b ** 2 - 3 * a * slope
                    if discriminant < 0:
                        lambda_val = max_lambda * lambda1
                    elif b <= 0:
                        lambda_val = (-b + np.sqrt(discriminant)) / (3 * a)
                    else:
                        lambda_val = -slope / (b + np.sqrt(discriminant))

            lambda_val = min(lambda_val, max_lambda * lambda1)

        elif np.any(np.isnan(f)) or np.any(np.isinf(f)):
            lambda_val = max_lambda * lambda1
        else:
            lambda_val = 1  # Fraction of Newton step

        # Adjust lambda if needed
        if lambda_val < 1:
            lambda2 = lambda1 
            f2 = f
            lambda_val = np.maximum(lambda_val, min_lambda * lambda1)

        resnorm0 = resnorm
        resnorm = np.linalg.norm(F, np.inf)

    return x, F, exitflag


# class NewtonRaphsonSolver(ODESolver):

#     def __init__(self, fun, x0, options=None):
#         self.fun = fun  # function to solve (residuals function)
#         self.x = np.array(x0)  # initial guess, converted to a numpy array
#         self.options = options if options else {'TolX': 1e-12, 'TolFun': 1e-6, 'MaxIter': 1000}
        
#         # Solver options
#         self.TOLX = self.options['TolX']
#         self.TOLFUN = self.options['TolFun']
#         self.MAXITER = self.options['MaxIter']
#         self.ALPHA = 1e-4  # Backtracking parameter
#         self.MIN_LAMBDA = 0.1
#         self.MAX_LAMBDA = 0.5

#         self.iter_count = 0
#         self.lambda_val = 1  # Backtracking initial lambda value

#     # def jacobian(self, x, funx, dx=1e-6):
#     #     """Compute the Jacobian matrix using finite differences."""
#     #     nx = len(x)
#     #     nf = len(funx)
#     #     J = np.zeros((nf, nx))

#     #     for n in range(nx):
#     #         delta = np.zeros(nx)
#     #         delta[n] = dx
#     #         dF = self.fun(x + delta) - funx
#     #         J[:, n] = dF.flatten() / dx  # Derivatives
#     #     return J

#     def solve(self):
        
#         # Make initial guess
#         F = self.fun(self.x)  # Evaluate residuals at initial guess
#         J = self.jacobian(self.x, F)
        
#         resnorm = np.linalg.norm(F, np.inf)  # Compute residual norm
#         resnorm0 = 100 * resnorm
#         dx = np.zeros_like(self.x)

#         exitflag = 1  # Normal exit
#         if np.any(np.isnan(J)) or np.any(np.isinf(J)):
#             exitflag = -1  # Jacobian is singular
#             return self.x, F, exitflag

#         # Iterative solver
#         while (resnorm > self.TOLFUN or self.lambda_val < 1) and exitflag >= 0 and self.iter_count <= self.MAXITER:
#             if self.lambda_val == 1:
#                 self.iter_count += 1
#                 # Recalculate Jacobian if necessary
#                 if resnorm / resnorm0 > 0.2:
#                     J = self.jacobian(self.x, F)

#                 if np.linalg.cond(J) <= np.finfo(float).eps:
#                     dx = np.linalg.pinv(J) @ -F
#                 else:
#                     dx = np.linalg.solve(J, -F)

#                 g = np.dot(F, J.T)
#                 slope = np.dot(g, dx)
#                 fold = np.dot(F, F)
#                 xold = self.x.copy()
#                 lambda_min = self.TOLX / np.max(np.abs(dx) / np.maximum(np.abs(xold), 1))
            
#             if self.lambda_val < lambda_min:
#                 exitflag = 2  # Too small step
#                 break
#             elif np.any(np.isnan(dx)) or np.any(np.isinf(dx)):
#                 exitflag = -1  # Step is NaN or Inf
#                 break

#             # Update solution
#             self.x = xold + dx * self.lambda_val
#             F = self.fun(self.x)
#             f = np.dot(F, F)

#             # Backtracking line search
#             lambda1 = self.lambda_val 

#             if f > fold + self.ALPHA * self.lambda_val * slope:
#                 if self.lambda_val == 1:
#                     self.lambda_val = -slope / (2 * (f - fold - slope))
#                 else:
#                     A = 1 / (lambda1 - lambda2)
#                     B = np.array([[1 / lambda1 ** 2, -1 / lambda2 ** 2], [-lambda2 / lambda1 ** 2, lambda1 / lambda2 ** 2]])
#                     C = np.array([f - fold - lambda1 * slope, f2 - fold - lambda2 * slope])
#                     # coeff = np.linalg.solve(A @ B, C)
#                     coeff = np.linalg.solve(A*B, C)
#                     a, b = coeff[0], coeff[1]

#                     if a == 0:
#                         self.lambda_val = -slope / (2 * b)
#                     else:
#                         discriminant = b ** 2 - 3 * a * slope
#                         if discriminant < 0:
#                             self.lambda_val = self.MAX_LAMBDA * lambda1
#                         elif b <= 0:
#                             self.lambda_val = (-b + np.sqrt(discriminant)) / (3 * a)
#                         else:
#                             self.lambda_val = -slope / (b + np.sqrt(discriminant))

#                 self.lambda_val = min(self.lambda_val, self.MAX_LAMBDA * lambda1)

#             elif np.any(np.isnan(f)) or np.any(np.isinf(f)):
#                 self.lambda_val = self.MAX_LAMBDA * lambda1
#             else:
#                 self.lambda_val = 1  # Fraction of Newton step

#             # Adjust lambda if needed
#             if self.lambda_val < 1:
#                 lambda2 = lambda1 
#                 f2 = f
#                 self.lambda_val = np.maximum(self.lambda_val, self.MIN_LAMBDA * lambda1)

#             resnorm0 = resnorm
#             resnorm = np.linalg.norm(F, np.inf)

#         return self.x, F, exitflag


# # # Example usage:
# # def fun_example(x):
# #     """Example residual function."""
# #     return np.array([x[0] ** 2 + x[1] ** 2 - 4, x[0] - x[1] ** 2])
# # # Initial guess
# # x0 = [1, 1]
# # # Solver options
# # options = {'TolX': 1e-6, 'TolFun': 1e-6, 'MaxIter': 1000}
# # # Create NewtonRaphsonSolver instance
# # solver = NewtonRaphsonSolver(fun_example, x0, options)
# # # Solve the system of equations
# # solution, residuals, exitflag = solver.solve()
# # print("Solution:", solution)
# # print("Residuals:", residuals)
# # print("Exit Flag:", exitflag)
