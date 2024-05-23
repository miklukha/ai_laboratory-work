# import numpy as np

# def process_file(filename):
#     dataset = []
#     with open(filename, 'r') as file:
#         lines = file.readlines()
#         lines = lines[2:]

#         length = len(lines)
#         answers = []
#         pictures = []

#         for i in range(length):
            
#             if lines[i] == '\n':
#                 b = [[int(num) for sublist in [e.split(' ') for e in answers] for num in sublist]]
#                 dataset.append(b)
#                 answers = []
#                 continue

#             if len(lines[i]) == 4:
#                 answers.append(lines[i].strip())
#                 b = [[int(num) for sublist in [e.split(' ') for e in pictures] for num in sublist]]
#                 dataset.append(b)
#                 pictures = []
            
#             if len(lines[i]) == 12:
#                 pictures.append(lines[i].strip())

#     return dataset

# datasetPev = process_file('lab2/test1.train')

# dataset = []

# for e in range(len(datasetPev)):
#     dataset.append(np.array(datasetPev[e]))

# IN_LAYER = 36
# H_LAYER = 36
# OUT_LAYER = 2

# LEARNING_RATE = 0.01
# EPOCHS = 1000

# w1 = np.random.randn(IN_LAYER, H_LAYER)
# w2 = np.random.randn(H_LAYER, OUT_LAYER)

# def reLU(s):
#     return np.maximum(s, 0)

# def reLU_deriv(s):
#     s[s <= 0] = 0
#     s[s > 0] = 1
#     return s

# def sigmoid(s):
#     return 1 / (1 + np.exp(-s))

# def sigmoid_deriv(s):
#     return s * (1 - s)

# def predict(numbers):
#     s1 = numbers.dot(w1) 
#     hidden = reLU(s1)

#     s2 = hidden.dot(w2) 
#     y = sigmoid(s2)
#     return y


# for e in range(EPOCHS):
#     for i in range(0, len(dataset), 2):
#         numbers =  dataset[i]
#         result = dataset[i + 1]

#         s1 = numbers.dot(w1)  
#         hidden = reLU(s1)

#         s2 = hidden.dot(w2) 
#         y = sigmoid(s2)

#         error = y - result
        
#         hidden_error = np.array(error).dot(w2.T)
#         delta_w2 = hidden.T.dot(error)

#         hidden_delta = hidden_error * reLU_deriv(s1)
#         delta_w1 = numbers.T.dot(hidden_delta)

#         w2 -= LEARNING_RATE * delta_w2
#         w1 -= LEARNING_RATE * delta_w1


# def get_letter(data, value):
#     for item in data:
#         if item['value'] == value:
#             return item['letter']
#     return None

# data = [
#     {'letter': 'Y', 'value': [0, 0]},
#     {'letter': 'X', 'value': [0, 1]},
#     {'letter': 'Z', 'value': [1, 0]},
#     {'letter': 'U', 'value': [1, 1]}
# ]

# testData = process_file('lab2/test1.test')

# for i in range(0, len(testData), 2):
#     print('---------------------------')
#     approximate_answer = predict(np.array(testData[i]))
#     answer = np.round(approximate_answer).astype(int).tolist()

#     letter = get_letter(data, answer[0])
#     print('answer ->', answer) 
#     print('letter ->', letter) 


###!!! лише сігмоїд
import numpy as np

def process_file(filename):
    dataset = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        lines = lines[2:]

        length = len(lines)
        answers = []
        pictures = []

        for i in range(length):
            if lines[i] == '\n':
                b = [[int(num) for sublist in [e.split(' ') for e in answers] for num in sublist]]
                dataset.append(b)
                answers = []
                continue

            if len(lines[i]) == 4:
                answers.append(lines[i].strip())
                b = [[int(num) for sublist in [e.split(' ') for e in pictures] for num in sublist]]
                dataset.append(b)
                pictures = []
            
            if len(lines[i]) == 12:
                pictures.append(lines[i].strip())

    return dataset

datasetPev = process_file('lab2/test1.train')

dataset = []

for e in range(len(datasetPev)):
    dataset.append(np.array(datasetPev[e]))

IN_LAYER = 36
H_LAYER = 36
OUT_LAYER = 2

LEARNING_RATE = 0.01
EPOCHS = 300

w1 = np.random.randn(IN_LAYER, H_LAYER)
w2 = np.random.randn(H_LAYER, OUT_LAYER)

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def sigmoid_deriv(s):
    return s * (1 - s)

def predict(numbers):
    s1 = numbers.dot(w1) 
    hidden = sigmoid(s1)

    s2 = hidden.dot(w2)
    y = sigmoid(s2)
    return y

for e in range(EPOCHS):
    for i in range(0, len(dataset), 2):
        numbers =  dataset[i]
        result = dataset[i + 1]

        s1 = numbers.dot(w1)  
        hidden = sigmoid(s1)

        s2 = hidden.dot(w2) 
        y = sigmoid(s2)

        error = y - result
        
        hidden_error = np.array(error).dot(w2.T)
        delta_w2 = hidden.T.dot(error)

        hidden_delta = hidden_error * sigmoid_deriv(hidden)
        delta_w1 = numbers.T.dot(hidden_delta)

        w2 -= LEARNING_RATE * delta_w2
        w1 -= LEARNING_RATE * delta_w1

def get_letter(data, value):
    for item in data:
        if item['value'] == value:
            return item['letter']
    return None

data = [
    {'letter': 'Y', 'value': [0, 0]},
    {'letter': 'X', 'value': [0, 1]},
    {'letter': 'Z', 'value': [1, 0]},
    {'letter': 'U', 'value': [1, 1]}
]

testData = process_file('lab2/test1.test')

for i in range(0, len(testData), 2):
    print('---------------------------')
    approximate_answer = predict(np.array(testData[i]))
    answer = np.round(approximate_answer).astype(int).tolist()

    letter = get_letter(data, answer[0])
    print('answer ->', answer) 
    print('letter ->', letter)


##!!!! лише relu
# import numpy as np

# def process_file(filename):
#     dataset = []
#     with open(filename, 'r') as file:
#         lines = file.readlines()
#         lines = lines[2:]

#         length = len(lines)
#         answers = []
#         pictures = []

#         for i in range(length):
#             if lines[i] == '\n':
#                 b = [[int(num) for sublist in [e.split(' ') for e in answers] for num in sublist]]
#                 dataset.append(b)
#                 answers = []
#                 continue

#             if len(lines[i]) == 4:
#                 answers.append(lines[i].strip())
#                 b = [[int(num) for sublist in [e.split(' ') for e in pictures] for num in sublist]]
#                 dataset.append(b)
#                 pictures = []
            
#             if len(lines[i]) == 12:
#                 pictures.append(lines[i].strip())

#     return dataset

# datasetPev = process_file('lab2/test1.train')

# dataset = []

# for e in range(len(datasetPev)):
#     dataset.append(np.array(datasetPev[e]))

# IN_LAYER = 36
# H_LAYER = 36
# OUT_LAYER = 2

# LEARNING_RATE = 0.01
# EPOCHS = 300

# w1 = np.random.randn(IN_LAYER, H_LAYER)
# w2 = np.random.randn(H_LAYER, OUT_LAYER)

# def reLU(s):
#     return np.maximum(s, 0)

# def reLU_deriv(s):
#     s[s <= 0] = 0
#     s[s > 0] = 1
#     return s

# def predict(numbers):
#     s1 = numbers.dot(w1) 
#     hidden = reLU(s1)

#     s2 = hidden.dot(w2)
#     y = reLU(s2)
#     return y

# for e in range(EPOCHS):
#     for i in range(0, len(dataset), 2):
#         numbers =  dataset[i]
#         result = dataset[i + 1]

#         s1 = numbers.dot(w1)  
#         hidden = reLU(s1)

#         s2 = hidden.dot(w2) 
#         y = reLU(s2)

#         error = y - result
        
#         hidden_error = np.array(error).dot(w2.T)
#         delta_w2 = hidden.T.dot(error)

#         hidden_delta = hidden_error * reLU_deriv(s1)
#         delta_w1 = numbers.T.dot(hidden_delta)

#         w2 -= LEARNING_RATE * delta_w2
#         w1 -= LEARNING_RATE * delta_w1

# def get_letter(data, value):
#     for item in data:
#         if item['value'] == value:
#             return item['letter']
#     return None

# data = [
#     {'letter': 'Y', 'value': [0, 0]},
#     {'letter': 'X', 'value': [0, 1]},
#     {'letter': 'Z', 'value': [1, 0]},
#     {'letter': 'U', 'value': [1, 1]}
# ]

# testData = process_file('lab2/test1.test')

# for i in range(0, len(testData), 2):
#     print('---------------------------')
#     approximate_answer = predict(np.array(testData[i]))
#     answer = np.round(approximate_answer).astype(int).tolist()

#     letter = get_letter(data, answer[0])
#     print('answer ->', answer) 
#     print('letter ->', letter)

#!!! один шар    
# import numpy as np

# def process_file(filename):
#     dataset = []
#     with open(filename, 'r') as file:
#         lines = file.readlines()
#         lines = lines[2:]

#         length = len(lines)
#         answers = []
#         pictures = []

#         for i in range(length):
#             if lines[i] == '\n':
#                 b = [[int(num) for sublist in [e.split(' ') for e in answers] for num in sublist]]
#                 dataset.append(b)
#                 answers = []
#                 continue

#             if len(lines[i]) == 4:
#                 answers.append(lines[i].strip())
#                 b = [[int(num) for sublist in [e.split(' ') for e in pictures] for num in sublist]]
#                 dataset.append(b)
#                 pictures = []
            
#             if len(lines[i]) == 12:
#                 pictures.append(lines[i].strip())

#     return dataset

# datasetPev = process_file('lab2/test1.train')

# dataset = []

# for e in range(len(datasetPev)):
#     dataset.append(np.array(datasetPev[e]))

# IN_LAYER = 36
# OUT_LAYER = 2

# LEARNING_RATE = 0.01
# EPOCHS = 300

# w = np.random.randn(IN_LAYER, OUT_LAYER)

# def sigmoid(s):
#     return 1 / (1 + np.exp(-s))

# def sigmoid_deriv(s):
#     return s * (1 - s)

# def predict(numbers):
#     s = numbers.dot(w)
#     y = sigmoid(s)
#     return y

# for e in range(EPOCHS):
#     for i in range(0, len(dataset), 2):
#         numbers = dataset[i]
#         result = dataset[i + 1]

#         s = numbers.dot(w)
#         y = sigmoid(s)

#         error = y - result
        
#         delta_w = numbers.T.dot(error * sigmoid_deriv(y))

#         w -= LEARNING_RATE * delta_w

# def get_letter(data, value):
#     for item in data:
#         if item['value'] == value:
#             return item['letter']
#     return None

# data = [
#     {'letter': 'Y', 'value': [0, 0]},
#     {'letter': 'X', 'value': [0, 1]},
#     {'letter': 'Z', 'value': [1, 0]},
#     {'letter': 'U', 'value': [1, 1]}
# ]

# testData = process_file('lab2/test1.test')


# for i in range(0, len(testData), 2):
#     print('---------------------------')
#     approximate_answer = predict(np.array(testData[i]))
#     answer = np.round(approximate_answer).astype(int).tolist()

#     letter = get_letter(data, answer[0])
#     print('answer ->', answer) 
#     print('letter ->', letter)
