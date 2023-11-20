class MjNode:

    def __init__(self, id, name):
        self.id = id
        self.children = []
        self.parent = None
        self.name = name

    def addChild(self, child):
        self.children.append(child)

    def setParent(self, parent):
        self.parent = parent

    def __repr__(self):
        return f"Node({self.id})"
    
    def __str__(self):
        return f"Node({self.id})"
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __lt__(self, other):
        return self.id < other.id
    
    def __le__(self, other):
        return self.id <= other.id
    
    def __gt__(self, other):
        return self.id > other.id
    
    def __ge__(self, other):
        return self.id >= other.id
    
    def __cmp__(self, other):
        return self.id - other.id
    

def visualize_mj_tree(root: MjNode, depth=0):
    print("| "*depth + str(root.id) + " " + root.name)
    for child in root.children:
        visualize_mj_tree(child, depth+1)


def build_mj_node_children_map(model):
    node_children = {}
    # we start at 1 because 0 is the root node
    for i in range(1, model.nbody):
        body = model.body(i)
        parentId = body.parentid[0]
        if parentId not in node_children:
            node_children[parentId] = []
        node_children[parentId].append(i)
    return node_children


# also add hash map of node id to node
def build_mj_tree_from_node_children_map(model, root, node_children_map, node_map):
    node_map[root.id] = root
    if root.id not in node_children_map:
        return
    for child_id in node_children_map[root.id]:
        childNode = MjNode(child_id, model.body(child_id).name)
        childNode.setParent(root)
        root.addChild(childNode)
        build_mj_tree_from_node_children_map(model, childNode, node_children_map, node_map)


def build_mj_tree(model):
    root = MjNode(0, "world")
    node_children_map = build_mj_node_children_map(model)
    node_map = {}
    build_mj_tree_from_node_children_map(model, root, node_children_map, node_map)
    return root, node_map


def get_body_names(model):
   return [model.body(i).name for i in range(model.nbody)]


def get_site_names(model):
   return [model.site(i).name for i in range(model.nsite)]

def get_geom_ids(node: MjNode, model, geom_ids):
    mjbody = model.body(node.id)
    for geomadr in mjbody.geomadr:
        if geomadr != -1:
            geom_ids.append(geomadr)
    for child in node.children:
        get_geom_ids(child, geom_ids)


def toggle_visibility(model, node: MjNode, value):
    mjbody = model.body(node.id)
    geomadr = mjbody.geomadr[0]
    ngeom = mjbody.geomnum[0]
    # TODO sites
    if geomadr != -1:
        for i in range(ngeom):
            model.geom(geomadr + i).rgba[3] = value
    for child in node.children:
        toggle_visibility(model, child, value)


def toggle_sites_visibility(model, value):
    for i in range(model.nsite):
        model.site(i).rgba[3] = value


# def build_tree(root, env, depth=0):
#     for child in env._get_children(root.id):
#         childNode = Node(child)
#         childNode.setParent(root)
#         root.addChild(childNode)
#         build_tree(childNode, env, depth+1)