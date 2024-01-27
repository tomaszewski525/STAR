def get_parent_child_joint(joint):
    connections = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7],
                   [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
                   [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20],
                   [19, 21], [20, 22], [21, 23]]

    kin = [[0, 1, 4, 7, 10], [0, 2, 5, 8, 11], [0, 3, 6, 9, 12, 15], [0, 3, 6, 9, 13, 16, 18, 20, 22],
           [0, 3, 6, 9, 14, 17, 19, 21, 23]]

    joints_that_influence = []
    for chain in kin:
        for j in chain:
            if j == joint:
                return joints_that_influence
            else:
                joints_that_influence.append(j)

        joints_that_influence = []

    return joints_that_influence


print(get_parent_child_joint(11))