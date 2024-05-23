import numpy as np

weights1 = np.array([[1, -1], [-1, 1]])
weights2 = np.array([1, 1])

def activation_function(s):
  if s < 0.5:
    return 0
  else:
    return 1

def predict(x):
    hidden_input = np.dot(x, weights1)  
    hidden_output = np.array([activation_function(x) for x in hidden_input])
    output = np.dot(hidden_output, weights2)  
    y = activation_function(output)
    return y

y00 = predict(np.array([0, 0]))
y01 = predict(np.array([0, 1]))
y10 = predict(np.array([1, 0]))
y11 = predict(np.array([1, 1]))

print('x1 = 0 XOR x2 = 0')
print('Y =', y00)
print()
print('x1 = 0 XOR x2 = 1')
print('Y =', y01)
print()
print('x1 = 1 XOR x2 = 0')
print('Y =', y10)
print()
print('x1 = 1 XOR x2 = 1')
print('Y =', y11)