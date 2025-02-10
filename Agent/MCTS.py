

class MCTS:

    def __init__(self, exp_const):
        self.exp_const = exp_const

    def select_node(self, root):
        current = root
        while len(current.children) != 0: # NB: !! WHILE !!
            current = max(current.children, key=lambda x: x.uct(self.exp_const))
        return current
    
    @staticmethod
    def expansion(node, A, B):
        nodes = []
        for action in node.n_actions: 
            nodes.append(node.predictive_step(action, A, B))
        return nodes
    
    @staticmethod
    def evaluation(nodes, likelihood):
        for node in nodes:
            node.cost = node.efe(likelihood)
    
    @staticmethod
    def propagation(nodes, likelihood):
        best_child = min(nodes, key=lambda x: x.efe(likelihood))

        cost = best_child.cost
        current = best_child.parent
        while current is not None:
            #print(f'current:{current.name} ')
            current.cost += cost
            current.visits += 1
            current = current.parent 