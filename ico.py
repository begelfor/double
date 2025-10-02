import numpy as np
import networkx as nx
from collections import Counter
from numpy.linalg import norm
import svgwrite

def make_icosahedron():
    phi = (1 + np.sqrt(5)) / 2
    points = [
        np.array([0, 1, phi]),
        np.array([0, 1, -phi]),
        np.array([0, -1, phi]),
        np.array([0, -1, -phi]),
        np.array([1, phi, 0]),
        np.array([1, -phi, 0]),
        np.array([-1, phi, 0]),
        np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]),
        np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]),
        np.array([-phi, 0, -1]),
    ]
    return points

def find_min_dist(points):
    min_dist = 10**5
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < min_dist:
                min_dist = dist
    return min_dist

def subdivide(points):
    min_dist = find_min_dist(points)    
    new_points = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < min_dist+0.01:
                new_points.append((points[i] + points[j]) / 2)
    return points + new_points

def make_dome(divisions=2):
    points = make_icosahedron()
    for _ in range(divisions):
        points = subdivide(points)
    min_dist = find_min_dist(points)
    G = nx.Graph()
    for i in range(len(points)):
        G.add_node(i, pos=points[i])
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if norm(points[i] - points[j]) < min_dist+0.01:
                G.add_edge(i, j)
    graph_R = 3 * 2**(divisions-1)
    distances, _  = nx.single_source_dijkstra(G, 0)
    G.remove_nodes_from([i for i in range(len(points)) if distances[i] > graph_R])
    for v, d in G.nodes(data=True):
        d['pos'] = d['pos']/norm(d['pos'])
    for v, u, d in G.edges(data=True):
        d['length'] = float(np.round(norm(G.nodes[v]['pos'] - G.nodes[u]['pos']), 3))

    edge_lengths = Counter()
    for _, _, d in G.edges(data=True):
        edge_lengths[d['length']] += 1
    edge_lengths = dict(edge_lengths)
    edge_type = dict({length: i for i, length in enumerate(edge_lengths.keys())})
    for _, _, d in G.edges(data=True):
        d['type'] = edge_type[d['length']]


    v_0 = G.nodes[0]['pos']
    # Create rotation matrix to map v_0 to z-axis (0,0,1)
    z_axis = np.array([0, 0, 1])
    v = np.cross(v_0, z_axis)
    s = np.linalg.norm(v)
    c = np.dot(v_0, z_axis)
    v_skew = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
    M = np.eye(3) + v_skew + (v_skew @ v_skew) * (1 - c)/(s*s) if s > 1e-10 else np.eye(3)
    for v, d in G.nodes(data=True):
        d['pos'] = M @ d['pos']
        d['flat'] = d['pos'][:2]+np.array([1, 1])


    # create svg of the half-dome
    scale = 400
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple'}
    dwg = svgwrite.Drawing('dome.svg', profile='tiny')
    dwg = svgwrite.Drawing('dome.svg', size=(f'{scale*2}px', f'{scale*2}px'), profile='tiny')
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='white')) # Add a background rectangle


    for v, u, d in G.edges(data=True):
        x1, y1 = G.nodes[v]['flat']*scale
        x2, y2 = G.nodes[u]['flat']*scale
        dwg.add(dwg.line(start=(int(x1), int(y1)), end=(int(x2), int(y2)), stroke=colors[d['type']], stroke_width=1))
    dwg.save()


    D = 3.25
    for (c, color), (length, count) in zip(colors.items(), edge_lengths.items()):
        print(f'{color}: {int(length*D/2*1000)}mm {count}')

    print(G)


    

    
    # print(equator_length)
    # D = 3.25
    # print(np.pi*D/len(equator))


if __name__ == "__main__":
    make_dome(divisions=2)