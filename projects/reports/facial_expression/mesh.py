#coding: utf-8
import utils
from scipy import sparse
import numpy as np


class Edge(object):

    def __init__(self, v1, v2, ids):
        """
        Construct an Edge from two vertex index

        :param v1:  First vertex index
        :param v2:  Second vertex index
        :param ids:  Edge ID
        """
        self._v1 = v1
        self._v2 = v2
        self.ids = ids
        self._f1 = None
        self._f2 = None

    def add_face(self, f_idx):
        if self._f1 is None:
            self._f1 = f_idx
        elif self._f2 is None:
            self._f2 = f_idx
        else:
            raise ValueError('Error, edge can not be linked to more than two faces')

    def num_face(self):
        """
        Indicate how many faces are attached to this edge

        :return: Number of face assigned
        """
        nf = 0
        nf += 1 if self._f1 else 0
        nf += 1 if self._f2 else 0
        return nf

    @property
    def f1(self):
        """
        Return first face associated to this edge

        :return: Face 1
        """
        return self._f1

    @property
    def f2(self):
        """
        Return second face associated to this edge

        :return: Face 2
        """
        return self._f2


class Mesh(object):

    def __init__(self, vertices=None, trilist=None):
        """
        Create a Mesh object from an array of vertices [Nx3] and an array of triangle [Kx3]

        :param vertices: Array of vertices
        :param trilist: Array of triangles
        """
        # List of vertices
        self.vertex = vertices
        # List of triangles
        self.tri = trilist
        # List of vertices connect to each vertices (neighbours)
        self.neighbour = None
        # List of edges connected to a given vertex
        self._eneighbour = dict()
        # List of edges
        self.edge = None
        # Adjacency matrix
        self._adj = None
        # Voronoi area
        self._voronoi_area = self._compute_voronoi_area()

    def save(self, filename):
        """
        Dump current mesh into an *.obj file

        :param filename:    Filename
        """
        utils.save_obj(self.vertex, self.tri, filename)

    def load(self, filename):
        """
        Load mesh from an *.obj file

        :param filename:    Path to obj file
        """
        self.vertex, self.tri = utils.load_obj(filename)
        self.neighbour = None
        self._eneighbour = dict()
        self.edge = None
        self._voronoi_area = self._compute_voronoi_area()

    def compute_laplacian(self, type):
        """
        Compute Laplacian operator for this mesh

        :param type:    Type of Laplacian (Combinatorial, Normalized, Cotan, ...)
        :return:    Degree, Adjacency matrix, Laplacian operator
        """
        if type == 'combinatorial':
            return self._combinatorial_laplacian()
        elif type == 'normalized':
            return self._normalized_laplacian()
        elif type == 'cotan':
            return self._cotan_laplacian()
        elif type == 'cotan_normalized':
            return self._cotan_laplacian(True)
        else:
            return None

    def _combinatorial_laplacian(self):
        """
        Compute combinatorial Laplacian, L = D - A

        :return:    Degree, Adjacency matrix, Laplacian operator
        """
        deg = self._compute_degree(self.tri)
        N = len(deg)
        adj = self._compute_adjacency_matrix()
        lap = sparse.spdiags(deg.T, [0], N, N, format='coo')
        lap -= adj
        return deg, adj, lap

    def _normalized_laplacian(self):
        """
        Compute combinatorial Laplacian, L = I - D ** -0.5 @ A @ D ** -0.5

        :return:    Degree, Adjacency matrix, Laplacian operator
        """
        deg = self._compute_degree(self.tri)
        N = len(deg)
        adj = self._compute_adjacency_matrix()
        lap = sparse.identity(N, dtype=np.float32, format='coo')
        D = sparse.spdiags(deg.T, [0], N, N, format='coo')
        D = D.power(-0.5)
        lap -= D.dot(adj.dot(D)) #@ (adj @ np.diagflat(deg ** -0.5))
        return deg, adj, lap

    def _cotan_laplacian(self, normalized=False):
        """
        Compute cotan laplacian operator

        :param normalized:  Indicate if the operator is normalized by the voronoi area
        :return:            Degree, Adjacency matrix, Laplacian
        """
        # Find edge's index between two vertex neighbour
        def _edge_common(v1, n1):
            for ve, idx in self._eneighbour[v1]:
                if ve == n1:
                    return self.edge[idx]
        # Compute degree
        deg = self._compute_degree(self.tri)
        # Adjacency
        adj = self._compute_adjacency_matrix()
        # Laplacian
        data = []
        ridx = []
        cidx = []
        for key, value in self._eneighbour.items():
            vi = key
            weights = []
            for neighbour, idx in value:
                edge = _edge_common(vi, neighbour)
                faces = [edge.f1, edge.f2]
                cotan = []
                ridx.append(vi)
                cidx.append(neighbour)
                for face in faces:
                    if face is not None:
                        # Compute cotan
                        v_idx = [v for v in self.tri[face, :] if v != vi and v != neighbour][0]
                        vp = self.vertex[v_idx, :]
                        u = self.vertex[vi, :] - vp
                        v = self.vertex[neighbour, :] - vp
                        cot = np.dot(u, v) / np.linalg.norm(np.cross(u, v))
                        cotan.append(cot)
                num = (self._voronoi_area[vi] * self._voronoi_area[neighbour])** 0.5 if normalized else len(cotan)
                w = -1.0 / num * np.sum(cotan)
                data.append(w)
                weights.append(w)
            # Add diag element
            ridx.append(vi)
            cidx.append(vi)
            data.append(-1.0 * np.sum(weights))
        # Create laplacian
        lap = sparse.coo_matrix((data, (ridx, cidx)), shape=adj.shape, dtype=adj.dtype)
        return deg, adj, lap

    def _compute_degree(self, trilist):
        """
        Compute node's degree

        :param trilist: Array of triangle
        :return:        Vertex degrees
        """
        return np.asarray(list(map(len, self.neighbour)),
                          dtype=np.float32).reshape((len(self.neighbour), 1))

    def _compute_adjacency_matrix(self):
        """
        Compute adjacency matrix

        :return:        Adjacency matrix
        """
        if self._adj is None:
            N = len(self.neighbour)

            ridx = []
            cidx = []
            data = []
            visited = []
            for idx, n_list in enumerate(self.neighbour):
                for n in n_list:
                    if (idx, n) and (n, idx) not in visited:
                        ridx.append(idx)
                        cidx.append(n)
                        data.append(1.0)
                        cidx.append(idx)
                        ridx.append(n)
                        data.append(1.0)
                        visited.append((idx, n))
                        visited.append((n, idx))
                    else:
                        a=0
            self._adj = sparse.coo_matrix((data, (ridx, cidx)), shape=(N,N), dtype=np.float32)
        return self._adj

    def _compute_edges(self, trilist):
        """
        Compute for each vertex the edges that are connected to them

        :param trilist: List of triangles
        :return:        Edge list
        """
        def _in_list(collections, value):
            f = [v for v in collections if v[0] == value]
            return f

        def _processed(p, v1, v2, ids):
            v1p = v1 in p
            v2p = v2 in p
            if v1p or v2p:
                f1 = []
                f2 = []
                if v1p:
                    f1 = _in_list(p[v1], v2)
                if v2p:
                    f2 = _in_list(p[v2], v1)
                if f1 and f2:
                    assert f1[0][1] == f2[0][1]
                    return f1[0][1]
                elif not f1 and not f2:
                    if v1p:
                        p[v1].append((v2, ids))
                    else:
                        p[v1] = [(v2, ids)]
                    if v2p:
                        p[v2].append((v1, ids))
                    else:
                        p[v2] = [(v1, ids)]
                    return -1
                else:
                    a = 0
                    raise ValueError('Should not reach this points')
            else:
                # New entry
                p[v1] = [(v2, ids)]
                p[v2] = [(v1, ids)]
                return -1

        edges = []
        self._eneighbour = dict()
        spoly = trilist.shape[1]
        # Iterate over all triangle
        for k, tri in enumerate(trilist):
            for i in range(spoly):
                v0 = tri[i]
                v1 = tri[(i + 1) % spoly]
                # Edge already in added ?
                idx = _processed(self._eneighbour, v0, v1, len(edges))
                if idx == -1:
                    # Create edge
                    idx = len(edges)
                    e = Edge(v0, v1, idx)
                    # Update list
                    edges.append(e)
                # Add face
                edges[idx].add_face(k)
        return edges

    def _compute_voronoi_area(self):
        """
        Compute for each vertex the voronoi cell's area
        :return:    Area
        """
        areas = []
        if self.vertex is not None:
            N = self.vertex.shape[0]
            for vidx in range(N):
                # Get all triangles connected to this vertex
                faces = set()
                for neighbour, eidx in self._eneighbour[vidx]:
                    edge = self.edge[eidx]
                    if edge.f1 is not None:
                        faces.add(edge.f1)
                    if edge.f2 is not None:
                        faces.add(edge.f2)
                # Compute area over all tri
                area = 0.0
                for f in faces:
                    # Query face
                    tri = self.tri[f,:]
                    v0 = tri[tri == vidx][0]
                    verts = np.where(tri != vidx)[0].tolist()
                    assert len(verts) == 2
                    v1 = tri[verts[0]]
                    v2 = tri[verts[1]]
                    # Compute Centroid + mid edges
                    centroid = np.mean(self.vertex[[v0, v1, v2], :], axis=0) - self.vertex[v0, :]
                    v1m = (self.vertex[v1, :] - self.vertex[v0, :]) / 2.0
                    v2m = (self.vertex[v2, :] - self.vertex[v0, :]) / 2.0
                    # Compute area
                    a = np.linalg.norm(np.cross(centroid, v1m)) / 2.0
                    a += np.linalg.norm(np.cross(centroid, v2m)) / 2.0
                    area += a
                # add to the list
                areas.append(area)
        return areas

    @property
    def vertex(self):
        """
        Access vertex storage

        :return: Array of vertices
        """
        return self.__vertex

    @property
    def tri(self):
        """
        Access triangle storage

        :return: Array of triangle
        """
        return self.__tri

    @property
    def neighbour(self):
        """
        Return list of vertices index connected to each node

        :return:    Adjacent vertex for each vertices
        """
        return self.__neighbour

    @property
    def edge(self):
        """
        Return list of edges

        :return:    Edges
        """
        return self.__edge

    @vertex.setter
    def vertex(self, vertices):
        """
        Set a new vertices

        :param vertices: Array of vertices to overwrite with
        """
        self.__vertex = vertices

    @tri.setter
    def tri(self, trilist):
        """
        Set a new list of triangles

        :param trilist: Array of triangles to overwrite with
        """
        self.__tri = trilist

    @neighbour.setter
    def neighbour(self, trilist):
        """
        Initialize neighbour structure

        :param trilist: Triangle list
        """
        self.__neighbour = utils.gather_neighbour(self.tri) if self.tri is not None else None

    @edge.setter
    def edge(self, edge):
        """
        Initialize edge list

        :param edge: edge
        """
        self.__edge = self._compute_edges(self.tri) if self.tri is not None else None
