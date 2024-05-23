import numpy as np

MATRIX_SIZE = 10
TARGET_STATE = [9]
GAMMA = 0.8
EPOCHS = 500
INITIAL_STATE = 4

R = np.array([
    [-1,  0,  -1, -1, -1, -1, -1, -1, -1, 100],
    [ 0,  -1,  0,  -1, -1, -1, -1, -1, -1, -1],
    [-1,  0,  -1, -1, -1,  0,  -1, -1, -1, -1],
    [-1, -1, -1, -1,  0,  -1,  0,  -1, -1, -1],
    [-1, -1, -1,  0,  -1,  0,  -1, -1, -1, -1],
    [-1, -1,  0,  -1,  0,  -1, -1, -1,  0,  -1],
    [-1, -1, -1,  0,  -1, -1, -1,  0,  -1, -1],
    [-1, -1, -1, -1, -1, -1,  0,  -1,  0,  100],
    [-1, -1, -1, -1, -1,  0,  -1,  0,  -1, -1],
    [ 0,  -1, -1, -1, -1, -1, -1,  0,  -1, 100]
])

Q = np.zeros((MATRIX_SIZE, MATRIX_SIZE))

def get_available_states(state):
    available_states = np.where(R[state] >= 0)[0]
    return available_states

def get_random_next_state(available_states) :
    return int(np.random.choice(available_states))

def get_best_next_state(state):
    max_index = np.where(Q[state] == np.max(Q[state]))[0]
    if len(max_index) > 1:
        max_index = int(np.random.choice(max_index))
    else:
        max_index = int(max_index[0])
    return max_index

def update_q_matrix(current_state, next_state, gamma):
    best_next_state = get_best_next_state(next_state)
    Q[current_state, next_state] = R[current_state, next_state] + gamma * Q[next_state, best_next_state]


for epoch in range(EPOCHS):
    current_state = np.random.randint(0, MATRIX_SIZE)
    available_states = get_available_states(current_state)
    next_state = get_random_next_state(available_states)
    update_q_matrix(current_state, next_state, GAMMA)

print("Trained Q matrix:")
for row in Q:
    print(" ".join(f"{val:10.2f}" for val in row))

current_state = INITIAL_STATE
path = [current_state + 1]  

while current_state not in TARGET_STATE:
    next_state = get_best_next_state(current_state)
    current_state = next_state
    path.append(current_state + 1)

print("Most shortest path:", path)
