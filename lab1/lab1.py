import numpy as np

# [1.92, 4.01, 1.48, 5.45, 1.56, 5.42, 1.28, 4.34, 1.51, 5.49, 1.32, 4.00, 0.49, 4.19, 1.53]
dataset = [(np.array([1.92, 4.01, 1.48]), 5.45),
           (np.array([4.01, 1.48, 5.45]), 1.56),
           (np.array([1.48, 5.45, 1.56]), 5.42),
           (np.array([5.45, 1.56, 5.42]), 1.28),
           (np.array([1.56, 5.42, 1.28]), 4.34),
           (np.array([5.42, 1.28, 4.34]), 1.51),
           (np.array([1.28, 4.34, 1.51]), 5.49),
           (np.array([4.34, 1.51, 5.49]), 1.32),
           (np.array([1.51, 5.49, 1.32]), 4.00),
           (np.array([5.49, 1.32, 4.00]), 0.49)]

LEARNING_RATE = 0.03
EPOCHS = 10000

weights = np.array(np.random.randn(3))

def activation_function(s):
    return 1 / (1 + np.exp(-s)) * 10

def predict(numbers):
    s = np.dot(numbers, weights)
    y = activation_function(s)
    return y

previous_error_sum = 0

for e in range(EPOCHS):
    current_error_sum = 0 
    
    for i in range(len(dataset)):
        numbers, answer = dataset[i]
        
        s = np.dot(numbers, weights)
        y = activation_function(s)

        error = y - answer
        error_sq = error ** 2
        current_error_sum += error_sq

        error_d = error * (np.exp(-s) / (1 + np.exp(-s)) ** 2) * numbers
        
        delta_w = (-LEARNING_RATE) * error_d 
        delta_w_av = delta_w / 10
   
        weights += delta_w_av 

    if abs(current_error_sum - previous_error_sum) < 0.0001:
        break

    print("Epoch:", e)
    print("Precision:", abs(previous_error_sum - current_error_sum))
    print("S", s)
    print(f"Y: {y}  error: {error}")
    print()


    previous_error_sum = current_error_sum 

x1 = np.array([1.32, 4.00, 0.49]) # 4.19
x2 = np.array([4.00, 0.49, 4.19]) # 1.53

predicted_answer1 = predict(x1)
predicted_answer2 = predict(x2)

print('Actual number 4.19, predicted: ', predicted_answer1)
print('Actual number 1.53, predicted: ', predicted_answer2)