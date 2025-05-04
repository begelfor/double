import galois

def get_lines(order):
    """
    Returns a list of lists of integers such that every two lists have exactly one element in common.
    The lists represent the lines of a projective plane above a Galois field of order 'order'.
    Each list is of size order+1.
    The number of lists is order^2+order+1.
    """
    
    GF = galois.GF(order)
    def hash(a):
        v = 0
        for z in a:
            v *= order
            v += z.item()
        return v

    points = []
    for i in range(order):
        for j in range(order):
            points.append(GF([1, i, j]))
    for i in range(order):
        points.append(GF([0, 1, i]))
    points.append(GF([0, 0, 1]))

    points_dict = {}
    for i, p in enumerate(points):
        for q in range(1,order):
            points_dict[hash(GF(q)*p)] = i
    
    lines = []
    for p in points:
        line = []
        space = GF([p]).null_space()
        v1 = space[0]
        v2 = space[1]
        for i in range(order):
            line.append(points_dict[hash(v1+GF(i)*v2)])
        line.append(points_dict[hash(v2)])
        lines.append(line)
    return lines
