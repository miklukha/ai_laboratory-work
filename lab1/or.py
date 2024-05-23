import numpy as np

weights = np.array([1, 1])

def activation_function(s):
  if s < 0.5:
    return 0
  else:
    return 1

def predict(numbers):
  s = np.dot(numbers, weights)
  y = activation_function(s)
  return y

y00 = predict(np.array([0, 0]))
y01 = predict(np.array([0, 1]))
y10 = predict(np.array([1, 0]))
y11 = predict(np.array([1, 1]))

print('x1 = 0 OR x2 = 0')
print('Y =', y00)
print()
print('x1 = 0 OR x2 = 1')
print('Y =', y01)
print()
print('x1 = 1 OR x2 = 0')
print('Y =', y10)
print()
print('x1 = 1 OR x2 = 1')
print('Y =', y11)