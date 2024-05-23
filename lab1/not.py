weight = -1.5

def activation_function(s):
  if s < -1:
    return 0
  else:
    return 1

def predict(x):
    s = x * weight
    y = activation_function(s)
    return y

y1 = predict(0)
y2 = predict(1)

print('x = 0')
print('Y =', y1)
print()
print('x = 1')
print('Y =', y2)
