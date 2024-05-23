using LinearAlgebra
A=randn(500,100)
b=randn(500)

#----------------------------Part A------------------------------------#
# Using the backslash method
theta1 = A \ b

# Using θ⋆ = (A⊤A)−1A⊤b
theta2 = inv(transpose(A) * A)*(transpose(A)*b)

# Using θ⋆ = A†b
theta3 = (inv(transpose(A)*A))*transpose(A)*b
theta3 = pinv(A)*b

# Similarity of 1 and 2 using the Euclidean Distance
similarity1_2 = norm(theta1 - theta2) # 6.067447566793674e-16
# The Euclidean Distance is essentially 0

# Similarity of 2 and 3 using the Euclidean Distance
similarity2_3 = norm(theta2 - theta3) # 1.252322586775844e-15
# The Euclidean Distance is essentially 0

# Similarity of 1 and 3 using the Euclidean Distance
similarity1_3 = norm(theta1 - theta3) #1.3592647490150705e-15
# The Euclidean Distance is essentially 0

#----------------------------Part B------------------------------------#
θ = randn(100)
not_optomized = norm((A * θ) - b)^2 # 49933.925933466606
optomized = norm((A * theta1) - b)^2 # 407.1095251268927
check1 = not_optomized > optomized # true

θ = randn(100)
not_optomized = norm((A * θ) - b)^2 # 55661.5513540332
optomized = norm((A * theta2) - b)^2 # 407.1095251268927
check2 = not_optomized > optomized # true

θ = randn(100)
not_optomized = norm((A * θ) - b)^2 # 43568.004965745546
optomized = norm((A * theta3) - b)^2 # 407.1095251268927
check3 = not_optomized > optomized # true

