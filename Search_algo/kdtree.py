class KDTree(object):
    
    """
    The points can be any array-like type, e.g: 
        lists, tuples, numpy arrays.

    Usage:
    1. Make the KD-Tree:
        `kd_tree = KDTree(points)`
    2. You can then use `get_knn` for k nearest neighbors or 
       `get_nearest` for the nearest neighbor
    """

    def __init__(self, points, dim=None, dist_sq_func=None):
        """Makes the KD-Tree for fast lookup.

        Parameters
        ----------
        points : list<point>
            A list of points.
        dim : int 
            The dimension of the points. 
            If omitted, takes the dimension of the first point.
        dist_sq_func : function(point, point), optional
            A function that returns the squared distance between the two points. 
            If omitted, it uses the default Manhattan squared implementation.
        """

        if dist_sq_func is None:
            dist_sq_func = lambda a, b: sum((x - b[i]) ** 2 for i, x in enumerate(a))
            
        if dim is None:
            dim = len(points[0])
            
        def make(points, i=0):
            """Creates a KDTree from the list of points.
            Create a structure of [left branch, right branch, median] for each node in the tree.
            
            points : list<point>
                A list of points.
            i : int 
                The current dimension to split. 
            """
            if len(points) > 1:
                points.sort(key=lambda x: x[i])
                i = (i + 1) % dim
                m = len(points) >> 1
                return [make(points[:m], i), make(points[m + 1:], i), points[m]]
            if len(points) == 1:
                return [None, None, points[0]]
        
        def add_point(node, point, i=0):
            if node is not None:
                dx = node[2][i] - point[i]
                for j, c in ((0, dx >= 0), (1, dx < 0)):
                    if c and node[j] is None:
                        node[j] = [None, None, point]
                    elif c:
                        add_point(node[j], point, (i + 1) % dim)

        import heapq
        def get_knn(node, point, k, return_dist_sq, heap, i=0, tiebreaker=1):
            if node is not None:
                dist_sq = dist_sq_func(point, node[2])
                dx = node[2][i] - point[i]
                if len(heap) < k:
                    heapq.heappush(heap, (-dist_sq, tiebreaker, node[2]))
                elif dist_sq < -heap[0][0]:
                    heapq.heappushpop(heap, (-dist_sq, tiebreaker, node[2]))
                i = (i + 1) % dim
                # Goes into the left branch, then the right branch if needed
                for b in (dx < 0, dx >= 0)[:1 + (dx * dx < -heap[0][0])]:
                    get_knn(node[b], point, k, return_dist_sq, 
                        heap, i, (tiebreaker << 1) | b)
            if tiebreaker == 1:
                return [(-h[0], h[2]) if return_dist_sq else h[2] for h in sorted(heap)][::-1]

        self._add_point = add_point
        self._get_knn = get_knn 
        self._root = make(points)

    def add_point(self, point):
        """Adds a point to the kd-tree.
        
        Parameters
        ----------
        point : array-like
            The point.
        """
        if self._root is None:
            self._root = [None, None, point]
        else:
            self._add_point(self._root, point)

    def get_knn(self, point, k, return_dist_sq=True):
        """Returns k nearest neighbors.

        Parameters
        ----------
        point : array-like
            The point.
        k: int 
            The number of nearest neighbors.
        return_dist_sq : boolean
            Whether to return the squared Euclidean distances.

        Returns
        -------
        list<array-like>
            The nearest neighbors. 
            If `return_dist_sq` is true, the return will be:
                [(dist_sq, point), ...]
            else:
                [point, ...]
        """
        return self._get_knn(self._root, point, k, return_dist_sq, [])

    def get_nearest(self, point, return_dist_sq=True):
        """Returns the nearest neighbor.

        Parameters
        ----------
        point : array-like
            The point.
        return_dist_sq : boolean
            Whether to return the squared Euclidean distance.

        Returns
        -------
        array-like
            The nearest neighbor. 
            If the tree is empty, returns `None`.
            If `return_dist_sq` is true, the return will be:
                (dist_sq, point)
            else:
                point
        """
        l = self._get_knn(self._root, point, 1, return_dist_sq, [])
        return l[0] if len(l) else None

import numpy as np

if (__name__ == '__main__'):
    # Generate 10 1*5 vectors filled with random numbers
    vectors = [[np.random.rand() for _ in range(5)] for _ in range(10)]
    query = [np.random.rand() for _ in range(5)]
    
    print("Vectors:")
    for vec in vectors:
        print(vec)
        
    print("Query:")
    print(query)
    
    tree = KDTree(vectors)
    for result in tree.get_knn(query, 3):
        print(f'Distance: {result[0]}\t| Vector: {result[1]}')
        
    dist_sq_func = lambda a, b: sum((x - b[i]) ** 2 for i, x in enumerate(a))
    actual_min = np.inf
    for vec in vectors:
        actual_min = min( dist_sq_func(vec, query), actual_min )
        
    print("Actual minimal distance:", actual_min)