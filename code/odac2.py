from __future__ import annotations
import river.metrics
import collections
import itertools
import math
import typing

from river.metrics import Jaccard


from river import base , stats


class ODAC2(base.Clusterer):

    def __init__(self, confidence_level: float = 0.9, n_min: int = 1500, tau: float = 0.1): #n min 100
        if not (confidence_level > 0.0 and confidence_level < 1.0):
            raise ValueError("confidence_level must be between 0 and 1.")
        if not n_min > 0:
            raise ValueError("n_min must be greater than 1.")
        if not tau > 0.0:
            raise ValueError("tau must be greater than 0.")

        self._root_node = ODACCluster("ROOT")
        self.confidence_level = confidence_level
        self.n_min = n_min
        self.tau = tau

        self._update_timer: int = n_min
        self._n_observations: int = 0
        self._is_init = False

        self._structure_changed = False


        self._timeseries_data = collections.defaultdict(list)
        jaccard = river.metrics.Jaccard()
        self.previous_cluster_labels = None  # 

    @property
    def n_active_clusters(self):
        return self._count_active_clusters(self._root_node)

    @property
    def n_clusters(self):
        return self._count_clusters(self._root_node)

    @property
    def height(self) -> int:
        return self._calculate_height(self._root_node)

    @property
    def summary(self):
        summary = {
            "n_clusters": self.n_clusters,
            "n_active_clusters": self.n_active_clusters,
            "height": self.height,
        }
        return summary

    def _calculate_height(self, node: ODACCluster) -> int:
        if node.children is not None:
            child_heights = [
                self._calculate_height(child)
                for child in [node.children.first, node.children.second]
            ]
            return max(child_heights) + 1
        else:
            return 0

    def _count_clusters(self, node: ODACCluster) -> int:
        count = 1
        if node.children is not None:
            for child in [node.children.first, node.children.second]:
                count += self._count_clusters(child)
        return count

    def _count_active_clusters(self, node: ODACCluster) -> int:
        if node.active:
            return 1
        elif node.children is not None:
            return sum(
                self._count_active_clusters(child)
                for child in [node.children.first, node.children.second]
            )
        else:
            return 0


    def _find_all_active_clusters(self, node: ODACCluster,  path="root"):
        if node.active:
            node_path = node.name if hasattr(node, 'name') and node.name else path
            yield (node, node_path)
        elif node.children is not None:
            for i, child in enumerate([node.children.first, node.children.second]):
                child_path = f"{path}_{i}"
                yield from self._find_all_active_clusters(child, child_path)

        # if node.active:
        #     yield node
        # elif node.children is not None:
        #     for child in [node.children.first, node.children.second]:
        #         yield from self._find_all_active_clusters(child)


    def _modify_cluster_naming(self):
        """
        Updates cluster naming to include full path information.
        Should be called after tree structure changes.
        """
        def _update_names(node, path="ROOT", level=0):
            if node is None:
                return
                
            # Update the name to represent the path from the node to the root
            node.name = f"LVL_{level}_{path}"
            
            if node.children is not None:
                _update_names(node.children.first, f"{path}_0", level + 1)
                _update_names(node.children.second, f"{path}_1", level + 1)

        _update_names(self._root_node)
        #self._structure_changed = True


    def learn_one(self, x: dict):
        # If x is empty, do nothing
        if not x:
            return

        if self._structure_changed:
            self._structure_changed = False
            #self._modify_cluster_naming() # NEWWWW

        if not self._is_init:
            # Initialize the first cluster which is the ROOT cluster
            self._root_node(list(x.keys()))
            self._structure_changed = False
            #print("ROOT cluster initialized")
            self._is_init = True

        # Update the total observations received
        self._n_observations += 1

        # Split control
        self._update_timer -= 1

        # For each active cluster update the statistics and time to time verify if
        # the cluster needs to merge or expand
        for leaf_tuple in self._find_all_active_clusters(self._root_node):
            # Unpack the tuple to get the leaf node and its path
            leaf, leaf_path = leaf_tuple  # This is the key change
            
            if not leaf.active:
                continue

            # Update statistics
            leaf.update_statistics(x)

            if self._update_timer == 0:
                leaf.calculate_coefficients(confidence_level=self.confidence_level)

                if leaf.test_aggregate() or leaf.test_split(tau=self.tau):
                    # Put the flag change_detected to true to indicate to the user that the structure changed
                    self._structure_changed = True
                    self._modify_cluster_naming() # NEW
                    

        if self._structure_changed==True:
            self._print_cluster_labels() #HERE!
            
        if self._update_timer == 0:
            self._update_timer = self.n_min

            
    def _get_cluster_labels_string(self):
        cluster_info = []
        for cluster_tuple in self._find_all_active_clusters(self._root_node):
            cluster, cluster_path = cluster_tuple
            cluster_info.append(cluster.timeseries_names)
        
        return cluster_info

    
    def _print_cluster_labels(self):
        current_labels = {}
        for cluster_tuple in self._find_all_active_clusters(self._root_node):
            # Unpack the tuple
            cluster, cluster_path = cluster_tuple
            current_labels[cluster.name] = set(cluster.timeseries_names)
            
        # Only print if the structure has changed (i.e., the cluster labels are different from previous)
        if current_labels != self.previous_cluster_labels:
            for cluster_tuple in self._find_all_active_clusters(self._root_node):
                # Unpack the tuple here too
                cluster, cluster_path = cluster_tuple
                print(f"Cluster {cluster.name}: {cluster.timeseries_names}")
            self.previous_cluster_labels = current_labels



    def get_cluster_dict(self):
        """
        Creates a dictionary of clusters with their parent-child relationships based on the naming convention.
        
        Returns:
            dict: A dictionary with cluster information including members, parent, and siblings
        """
        cluster_dict = {}
        
        # First collect all active clusters
        for cluster_tuple in self._find_all_active_clusters(self._root_node):
            cluster, _ = cluster_tuple
            
            # Parse level and path from name
            parts = cluster.name.split('_')
            level = int(parts[1]) if len(parts) > 1 and parts[0] == 'LVL' else 0
            
            cluster_dict[cluster.name] = {
                'members': set(cluster.timeseries_names),
                'level': level,
                'parent': None,
                'siblings': []
            }
        
        # Establish parent-child relationships based on naming pattern
        for name in cluster_dict:
            # For clusters like LVL_2_ROOT_0_0, extract the path: ROOT_0_0
            if '_ROOT_' in name:
                path = name.split('_ROOT_')[1]
                parts = path.split('_')
                
                # If the path has multiple components (e.g., 0_0, 1_1)
                if len(parts) > 1:
                    # The parent path is everything except the last segment
                    parent_path = '_'.join(parts[:-1])
                    parent_level = cluster_dict[name]['level'] - 1
                    
                    # For LVL_2_ROOT_0_0, the parent would be LVL_1_ROOT_0
                    parent_name = f"LVL_{parent_level}_ROOT_{parent_path}"
                    
                    # Set the parent, even if it's not in our active clusters
                    cluster_dict[name]['parent'] = parent_name
        
        # Establish sibling relationships
        for name in cluster_dict:
            parent = cluster_dict[name]['parent']
            if parent:
                # Find all clusters with the same parent
                siblings = []
                for other_name in cluster_dict:
                    if other_name != name and cluster_dict[other_name]['parent'] == parent:
                        siblings.append(other_name)
                
                cluster_dict[name]['siblings'] = siblings
        
        return cluster_dict

    # Helper function to sort clusters by their level in the tree
    def sort_cluster_levels(cluster_dict):
        """
        Organizes clusters by their level in the tree.
        
        Args:
            cluster_dict: Dictionary of clusters from get_cluster_dict()
            
        Returns:
            dict: Dictionary where keys are levels and values are lists of cluster paths at that level
        """
        sorted_levels = {}
        for cluster_path, info in cluster_dict.items():
            level = info['level']
            if level not in sorted_levels:
                sorted_levels[level] = []
            sorted_levels[level].append(cluster_path)
        
        # Also sort clusters within each level
        for level in sorted_levels:
            sorted_levels[level].sort()
            
        return sorted_levels

    # This algorithm does not predict anything. It builds a hierarchical cluster's structure
    def predict_one(self, x: dict):
        """This algorithm does not predict anything. It builds a hierarchical cluster's structure.

        Parameters
        ----------
        x
            A dictionary of features.

        """
        raise NotImplementedError()

    def render_ascii(self, n_decimal_places: int = 2) -> str:
        """Render the structure of the clusters in ASCII format"""
        if not (n_decimal_places > 0 and n_decimal_places < 10):
            raise ValueError("n_decimal_places must be between 1 and 9.")

        return self._root_node.design_structure(n_decimal_places).rstrip("\n")




    @property
    def structure_changed(self) -> bool:
        return self._structure_changed
    

    def draw(self, show_clusters_info=None, cluster_colors=None):

        try:
            import graphviz
        except ImportError:
            raise ImportError("graphviz is required for visualization. Install it using 'pip install graphviz'.")
        
        dot = graphviz.Digraph(
            graph_attr={"splines": "ortho", "forcelabels": "true", "overlap": "false"},
            node_attr={
                "shape": "box",
                "penwidth": "1.2",
                "fontname": "trebuchet",
                "fontsize": "11",
                "margin": "0.1,0.0",
                "style": "filled",
            },
            edge_attr={"penwidth": "0.6", "center": "true", "fontsize": "7"},
        )
        
        def add_node(node, parent_name=None):
            if node is None:
                return
                
            # Default node color: blue for active clusters, gray for inactive
            if node.active:
                fill_color = "#76b5c5"  # Default blue for active clusters (from original code)
            else:
                fill_color = "#f2f2f2"  # Default gray for inactive clusters (from original code)
            
            node_label = node.name
            
            if show_clusters_info and node.active:
                info_parts = []
                
                if "timeseries_names" in show_clusters_info:
                    info_parts.append(f"{node.timeseries_names}")
                    
                    # Check if this cluster is highlighted (part of a merge)
                    members_tuple = tuple(sorted(node.timeseries_names))
                    if cluster_colors and members_tuple in cluster_colors:
                        fill_color = cluster_colors[members_tuple]
                
                if "level" in show_clusters_info:
                    level = node.name.split('_LVL_')[1].split('_')[0] if '_LVL_' in node.name else "N/A"
                    info_parts.append(f"Level: {level}")
                    
                if "path" in show_clusters_info:
                    path = node.name.split('_PATH_')[1] if '_PATH_' in node.name else node.name
                    info_parts.append(f"Path: {path}")
                    
                if info_parts:
                    node_label += '\n' + '\n'.join(info_parts)
            
            # Add node to graph with appropriate color
            dot.node(node.name, label=node_label, fillcolor=fill_color)
            
            # Add edge from parent if exists
            if parent_name:
                dot.edge(parent_name, node.name)
            
            # Recursively add children
            if node.children:
                add_node(node.children.first, node.name)
                add_node(node.children.second, node.name)
        
        # Start from root node
        add_node(self._root_node)
        
        return dot


class ODACCluster(base.Base):
    """Cluster class for representing individual clusters."""

    def __init__(self, name: str, parent: ODACCluster | None = None):
        self.name = name
        self.parent: ODACCluster | None = parent
        self.active = True
        self.children: ODACChildren | None = None

        self.timeseries_names: list[typing.Hashable] = []
      
        self._statistics = {
            (k1, k2): Jaccard() for k1, k2 in itertools.combinations(self.timeseries_names, 2)
        }

        self.d1: float | None = None
        self.d2: float | None = None
        self.e: float = 0
        self.d0: float | None = None
        self.avg: float | None = None

        self.pivot_0: tuple[typing.Hashable, typing.Hashable]
        self.pivot_1: tuple[typing.Hashable, typing.Hashable]
        self.pivot_2: tuple[typing.Hashable, typing.Hashable]

        self.n = 0
        self._timeseries_data = collections.defaultdict(list)
        
        

    # Method to design the structure of the cluster tree
    def design_structure(self, decimal_places: int = 2) -> str:
        pre_0 = "    "
        pre_1 = "│   "
        pre_2 = "├── "
        pre_3 = "└── "
        node = self
        prefix = (
            pre_2
            if node.parent is not None
            and (node.parent.children is None or id(node) != id(node.parent.children.second))  
            else pre_3
        )
        while node.parent is not None and node.parent.parent is not None:
            if node.parent.parent.children is None or id(node.parent) != id(
                node.parent.parent.children.second
            ):  
                prefix = pre_1 + prefix
            else:
                prefix = pre_0 + prefix
            node = node.parent

        if self.d1 is not None:
            r_d1 = f"{self.d1:.{decimal_places}f}"
        else:
            r_d1 = "<Not calculated>"

        if self.d2 is not None:
            r_d2 = f" d2={self.d2:.{decimal_places}f}"
        else:
            r_d2 = ""

        if self.parent is None:
            representation = f"{self.name} d1={r_d1}{r_d2}"
        else:
            representation = f"{prefix}{self.name} d1={r_d1}{r_d2}"

        if self.active is True:
            return representation + f" {self.timeseries_names}\n"
        else:
            representation += " [NOT ACTIVE]\n"
        if self.children is not None:
            for child in [self.children.first, self.children.second]:
                representation += child.design_structure(decimal_places)
        return representation

    def __str__(self) -> str:
        return self.design_structure()

    def __repr__(self) -> str:
        return self.design_structure()

    def _init_stats(
        self,
    ) -> dict[tuple[typing.Hashable, typing.Hashable], stats.PearsonCorr] | stats.Var:
        return (
            collections.defaultdict(
                stats.PearsonCorr,
                {
                    (k1, k2): stats.PearsonCorr()
                    for k1, k2 in itertools.combinations(self.timeseries_names, 2)
                },
            )
            if len(self.timeseries_names) > 1
            else stats.Var()
        )

    def __call__(self, ts_names: list[typing.Hashable]):
        """Method that associates the time-series into the cluster and initiates the statistics."""
        self.timeseries_names = sorted(ts_names)  # type: ignore
        self._statistics = self._init_stats()

    def update_statistics(self, x: dict) -> None:
        if len(self.timeseries_names) > 1:
         
            for (k1, k2), item in self._statistics.items():  # type: ignore
                if x.get(k1, None) is None or x.get(k2, None) is None:
                    continue
                # item.update(float(x[k1]), float(x[k2]))  # ---------
                item.update(x[k1], x[k2])
                
        else:
            self._statistics.update(float(x.get(self.timeseries_names[0])))  

        self.n += 1

 

    def _calculate_rnomc_dict(self) -> dict[tuple[typing.Hashable, typing.Hashable], float]:
        rnomc_dict = {}

        for k1, k2 in itertools.combinations(self.timeseries_names, 2):

            item = self._statistics[(k1, k2)] 
        
            # Get the Jaccard similarity score
            jaccard_similarity = item.get()

            rnomc_dict[(k1, k2)] = jaccard_similarity

        return rnomc_dict




    # Method to calculate coefficients for splitting or aggregation
    def calculate_coefficients(self, confidence_level: float) -> None:
        if len(self.timeseries_names) > 1:
            # Get the rnomc values
            rnomc_dict = self._calculate_rnomc_dict()

            # Get the average distance in the cluster
            self.avg = sum(rnomc_dict.values()) / self.n

            # Get the minimum distance and the pivot associated in the cluster
            self.pivot_0, self.d0 = min(rnomc_dict.items(), key=lambda x: x[1])
            # Get the maximum distance and the pivot associated in the cluster
            self.pivot_1, self.d1 = max(rnomc_dict.items(), key=lambda x: x[1])

            # Get the second maximum distance and the pivot associated in the cluster
            remaining = {k: v for k, v in rnomc_dict.items() if k != self.pivot_1}

            if len(remaining) > 0:
                self.pivot_2, self.d2 = max(remaining.items(), key=lambda x: x[1])
            else:
                self.pivot_2 = self.d2 = None  # type: ignore
        else:
            self.d1 = self._statistics.get()  # type: ignore
        # Calculate the Hoeffding bound in the cluster
        self.e = math.sqrt(math.log(1 / confidence_level) / (2 * self.n))

    # Method that gives the closest cluster where the current time series is located
    def _get_closest_cluster(self, pivot_1, pivot_2, current, rnormc_dict: dict) -> int:
        dist_1 = rnormc_dict.get((min(pivot_1, current), max(pivot_1, current)), 0)
        dist_2 = rnormc_dict.get((min(pivot_2, current), max(pivot_2, current)), 0)
        return 2 if dist_1 >= dist_2 else 1

    def _split_this_cluster(
        self,
        pivot_1: typing.Hashable,
        pivot_2: typing.Hashable,
        rnormc_dict: dict[tuple[typing.Hashable, typing.Hashable], float],
    ):
        """Expand into two clusters."""
        pivot_set = {pivot_1, pivot_2}
        pivot_1_list = [pivot_1]
        pivot_2_list = [pivot_2]

        # For each time-series in the cluster we need to find the closest pivot, to then associate with it
        for ts_name in self.timeseries_names:
            if ts_name not in pivot_set:
                cluster = self._get_closest_cluster(pivot_1, pivot_2, ts_name, rnormc_dict)
                if cluster == 1:
                    pivot_1_list.append(ts_name)
                else:
                    pivot_2_list.append(ts_name)
        # print(self.name)
        # new_name = "1" if self.name == "ROOT" else str(int(self.name.split("_")[-1]) + 1)
        new_name = "1" if self.name in ["ROOT", "LVL_0_ROOT"] else str(int(self.name.split("_")[-1]) + 1)

        # Create the two new clusters. The children of this cluster
        cluster_child_1 = ODACCluster("CH1_LVL_" + new_name, parent=self)
        cluster_child_1(pivot_1_list)

        cluster_child_2 = ODACCluster("CH2_LVL_" + new_name, parent=self)
        cluster_child_2(pivot_2_list)

        self.children = ODACChildren(cluster_child_1, cluster_child_2)

        # Set the active flag to false. Since this cluster is not an active cluster anymore.
        self.active = False

        # Reset some attributes
        self.avg = self.d0 = self.pivot_0 = self.pivot_1 = self.pivot_2 = self._statistics = None  

    # Method that proceeds to merge on this cluster
    def _aggregate_this_cluster(self):
        # Reset statistics
        self._statistics = self._init_stats()

        # Put the active flag to true. Since this cluster is an active cluster once again.
        self.active = True
        # Delete and disassociate the children.
        if self.children is not None:
            self.children.reset_parent()
            self.children = None
        # Reset the number of observations in this cluster
        self.n = 0

    # Method to test if the cluster should be split
    def test_split(self, tau: float):
        # Test if the cluster should be split based on specified tau
        if self.d2 is not None:
            if ((self.d1 - self.d2) > self.e) or (tau > self.e):  # type: ignore
                if ((self.d1 - self.d0) * abs(self.d1 + self.d0 - 2 * self.avg)) > self.e:  # type: ignore
                    # Split this cluster
                    self._split_this_cluster(
                        pivot_1=self.pivot_1[0],
                        pivot_2=self.pivot_1[1],
                        rnormc_dict=self._calculate_rnomc_dict(),
                    )
                    return True
        return False

    # Method to test if the cluster should be aggregated
    def test_aggregate(self):
        # Test if the cluster should be aggregated
        if self.parent is not None and self.d1 is not None:
            if self.d1 - self.parent.d1 > max(self.parent.e, self.e):
                self.parent._aggregate_this_cluster()
                return True
        return False


class ODACChildren(base.Base):
    """Children class representing child clusters."""

    def __init__(self, first: ODACCluster, second: ODACCluster):
        self.first = first
        self.second = second

    def reset_parent(self):
        self.first.parent = None
        self.second.parent = None