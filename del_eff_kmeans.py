import numpy as np

class Kmeans(object):
    '''
    In-house implementation of k-means via Lloyd-Max iterations
    This is a research prototype and is not necessarily well-optimized
    '''
    def __init__(self,
                    k,
                    termination='fixed',
                    iters=10,
                    tol=10**-3):
        '''
        Constructor
        INPUT:
        k - # of centroids/clusters
        iters - # of iterations to run
        termination - {'fixed', 'loss', 'centers'}
            if 'fixed' - runs for fixed # of iterations
            if 'loss' - runs until loss converges
            if 'centers' -runs until centers converge
        tol - numeric tolerance to determine convergence
        '''
        # set parameters
        self.k = k
        self.iters = iters
        self.tol = tol
        self.termination = termination
        # initialize placeholder values
        self._init_placeholders()

    def run(self, X):
        '''
        Run clustering algorithm
        INPUT:
        X - numpy matrix, n-by-d, each row is a data point
        OUTPUT: (3-tuple)
        centroids - k-by-d matrix of centroids
        assignments - Vector of length n, with datapoint to center assignments
        loss - The loss of the final partition
        '''
        self._set_data(X)
        self._lloyd_iterations()
        return self.centroids, self.assignments, self.loss

    def delete(self, del_idx):
        '''
        Delete point associated with key del_idx
        NOTE: del_idx must be int in {0,n-1}
            After deleting any key other than n-1,
            the (n-1)-th datapoint's key is automatically
            swapped with del_idx to
        '''
        self.data = np.delete(self.data, del_idx, 0)
        self.n = self.n-1
        self._init_placeholders()
        return self.run(self.data)

    def _init_placeholders(self):
        self.loss = np.Infinity
        self.empty_clusters = []
        self.kpp_inits = set()
        self.centroids = None
        self.assignments = None
        self.model = None

    def _set_data(self, X):
        self.data = X
        self.n, self.d = X.shape

    def _lloyd_iterations(self):
        self._init_centroids()
        for _ in range(self.iters):
            loss_prev = self.loss
            centers_prev = self.model
            self._assign_clusters()
            self._assign_centroids()
            prev = loss_prev if self.termination == 'loss' else centers_prev
            if self._check_termination(prev):
                break
            
    def _check_termination(self, prev):
        if self.termination == 'loss':
            return (1 - self.loss/prev) < self.tol
        elif self.termination == 'center':
            return np.linalg.norm(self.centroids - prev) < self.tol
        else:
            return False

    def _init_centroids(self):
        '''
        Kmeans++ initialization
        Returns vector of initial centroids
        '''
        first_idx = np.random.choice(self.n)
        self.centroids = self.data[first_idx,:]
        for kk in range(1,self.k):
            P = self._get_selection_prob()
            nxt_idx = np.random.choice(self.n,p=P)
            self.kpp_inits.add(nxt_idx)
            self.centroids = np.vstack([self.centroids,self.data[nxt_idx,:]])

    def _get_selection_prob(self):
        '''
        Outputs vector of selection probabilites
        Equal to Distance^2 to nearest centroid
        '''
        #handle edge case in centroids shape by unsqueezing
        if len(self.centroids.shape) == 1:
            self.centroids = np.expand_dims(self.centroids, axis=0)

        #probability is square distance to closest centroid
        D = np.zeros([self.n])
        for i in range(self.n):
            d = np.linalg.norm(self.data[i,:] - self.centroids, axis=1)
            D[i] = np.min(d)
        P = [dist**2 for dist in D]
        P = P / sum(P)
        return P  

    def _assign_centroids(self):
        '''
        Computes centroids in Lloyd iterations
        '''
        self.centroids = np.zeros([self.k,self.d])
        c = np.zeros([self.k])
        for i in range(self.n):
            a = self.assignments[i]
            c[a] += 1
            self.centroids[a,:] += self.data[i,:]
            
        for j in range(self.k):
            self.centroids[j,:] = self.centroids[j,:] / c[j]

        for j in self.empty_clusters: 
            self._reinit_cluster(j)
        self.empty_clusters = []
        
    def _assign_clusters(self):
        '''
        Computes clusters in Lloyd iterations
        '''
        assert (self.k, self.d) == self.centroids.shape, "Centers wrong shape"
        self.assignments = np.zeros([self.n]).astype(int)
        self.loss = 0
        for i in range(self.n):
            d = np.linalg.norm(self.data[i,:] - self.centroids, axis=1)
            d1 = np.linalg.norm(self.data[i,:] - self.centroids, axis=1,ord=1)
            self.assignments[i] = int(np.argmin(d))
            self.loss += np.min(d)**2
        self.loss = self.loss / self.n
        self.empty_clusters = self._check_4_empty_clusters()

    def _check_4_empty_clusters(self):
        empty_clusters = []
        for kappa in range(self.k):
            if len(np.where(self.assignments == kappa)[0]) == 0:
                empty_clusters.append(kappa)
        return empty_clusters

    def _reinit_cluster(self, j):
        '''
        Gets a failed centroid with idx j (empty cluster)
        Should replace with new k++ init centroid
        in:
            j is idx for centroid, 0 <= j <= n
        out:
            j_prime is idx for next centroid
        side-effects:
            centroids are update to reflect j -> j'
        '''
        P = self._get_selection_prob()
        j_prime = np.random.choice(self.n,p=P)
        self.kpp_inits.add(j_prime)
        self.centroids[j,:] = self.data[j_prime,:]
        return j_prime

class DCnode(Kmeans):
    '''
    A k-means subproblem for the divide-and-conquer tree
    in DC-k-means algorithm
    '''
    def __init__(self, k, iters):
        Kmeans.__init__(self, k, iters=iters)
        self.children = []
        self.parent = None
        self.time = 0
        self.loss = 0
        self.node_data = set()
        self.data_prop = set()

    def _run_node(self, X):
        self._set_node_data(X)
        self._lloyd_iterations()

    def _set_node_data(self, X):
        self.data = X[list(self.node_data)]
        self._set_data(self.data)

class DCKmeans():
    def __init__(self, ks, widths, iters=10):
        '''
        ks - list of k parameter for each layer of DC-tree
        widths - list of width parameter (number of buckets) for each layer
        iters - # of iterations to run 
            (at present, only supports fixed iteration termination)
        '''
        self.ks = ks
        self.widths = widths
        self.dc_tree = self._init_tree(ks,widths,iters)
        self.data_partition_table = dict()
        self.data = dict()
        self.dels = set()
        self.valid_ids = []
        self.d = 0
        self.n = 0
        self.h = len(self.dc_tree)
        for i in range(self.h):
            self.data[i] = None
            
    def run(self, X, assignments=False):
        '''
        X - numpy matrix, n-by-d, each row is a data point
        assignments (optional) - bool flag, computes assignments and loss
            NOTE: Without assignments flag, this only returns the centroids
        OUTPUT:
        centroids - k-by-d matrix of centroids
            IF assignments FLAG IS SET ALSO RETURNS:
        assignments - Vector of length n, with datapoint to center assignments
        loss - The loss of the final partition
        '''
        self._init_data(X)
        self._partition_data(X)
        self._run()
        if assignments:
            assignment_solver = Kmeans(self.ks[0])
            assignment_solver._set_data(X)
            assignment_solver.centroids = self.centroids
            assignment_solver._assign_clusters()
            self.assignments = assignment_solver.assignments
            self.loss = assignment_solver.loss
            return self.centroids, self.assignments, self.loss
        return self.centroids
    
    def delete(self, del_idx):
        idx = self.valid_ids[del_idx]
        self.valid_ids[del_idx] = self.valid_ids.pop()
        self.dels.add(idx)
        node = self.dc_tree[-1][self.data_partition_table[idx]]
        node.node_data.remove(idx)
        l = self.h-1
        self.n -= 1
        while True:
            node._run_node(self.data[l])
            if node.parent == None:
                self.centroids = node.centroids
                break
            data_prop = list(node.data_prop)
            for c_id in range(len(node.centroids)):
                idx = data_prop[c_id]
                self.data[l][idx] = node.centroids[c_id]
            node = node.parent
            l -= 1

    def _init_data(self,X):
        self.n = len(X)
        self.valid_ids = list(range(self.n))
        self.d = len(X[0])
        data_layer_size = self.n
        for i in range(self.h-1,-1,-1):
            self.data[i] = np.zeros((data_layer_size,self.d))
            data_layer_size = self.ks[i]*self.widths[i] 
        
    def _partition_data(self, X):
        self.d = len(X[0])
        num_leaves = len(self.dc_tree[-1])
        for i in range(len(X)):
            leaf_id = np.random.choice(num_leaves)
            leaf = self.dc_tree[-1][leaf_id]
            self.data_partition_table[i] = leaf_id
            leaf.node_data.add(i)
            self.data[self.h-1][i] = X[i]

    def _run(self):
        for l in range(self.h-1,-1,-1):
            c = 0
            for j in range(self.widths[l]):
                subproblem = self.dc_tree[l][j]
                subproblem._run_node(self.data[l])
                if subproblem.parent == None:
                    self.centroids = subproblem.centroids
                else:
                    for c_id in range(len(subproblem.centroids)):
                        subproblem.data_prop.add(c)
                        subproblem.parent.node_data.add(c)
                        self.data[l-1][c] = subproblem.centroids[c_id] 
                        c += 1

    def _init_tree(self, ks, widths, iters):
        tree = [[DCnode(ks[0],iters)]] # root node
        for i in range(1,len(widths)):
            k = ks[i]
            assert widths[i] % widths[i-1] == 0, "Inconsistent widths in tree"
            merge_factor = int(widths[i] / widths[i-1])
            level = []
            for j in range(widths[i-1]):
                parent = tree[i-1][j]
                for _ in range(merge_factor):
                    child = DCnode(k,iters=10)
                    child.parent = parent
                    parent.children.append(child)
                    level.append(child)
            tree.append(level)
        return tree
