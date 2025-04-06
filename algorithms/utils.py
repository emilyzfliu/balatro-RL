def enumerate_possible_hand_indices(hand_size):
    indices = []
    # size 1
    for i in range(hand_size):
        indices.append([i])
    
    # size 2
    for i in range(hand_size):
        for j in range(i + 1, hand_size):
            indices.append([i, j])
    
    # size 3
    for i in range(hand_size):
        for j in range(i + 1, hand_size):
            for k in range(j + 1, hand_size):
                indices.append([i, j, k])
    
    # size 4
    for i in range(hand_size):
        for j in range(i + 1, hand_size):
            for k in range(j + 1, hand_size):
                for l in range(k + 1, hand_size):
                    indices.append([i, j, k, l])

    # size 5
    for i in range(hand_size):
        for j in range(i + 1, hand_size):
            for k in range(j + 1, hand_size):
                for l in range(k + 1, hand_size):
                    for m in range(l + 1, hand_size):
                        indices.append([i, j, k, l, m])
    
    return indices
