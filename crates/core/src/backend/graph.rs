/// In-memory import graph backed by petgraph.
///
/// Loaded from LanceDB `code_edges` table at `CompositeBackend::open` time.
/// Kept in sync by `upsert_edges` / `delete_edges_for_file`. All BFS and
/// PageRank operations run entirely in-memory — no I/O per hop.
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// Directed edge graph over file paths.
/// Nodes are file paths (Strings); edges carry the edge_type label.
pub struct ImportGraph {
    pub graph: DiGraph<String, String>,
    /// O(1) lookup: file_path → NodeIndex
    pub node_index: HashMap<String, NodeIndex>,
}

impl ImportGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_index: HashMap::new(),
        }
    }

    /// Get or create the NodeIndex for a file path.
    pub fn get_or_insert_node(&mut self, file_path: &str) -> NodeIndex {
        if let Some(&idx) = self.node_index.get(file_path) {
            return idx;
        }
        let idx = self.graph.add_node(file_path.to_string());
        self.node_index.insert(file_path.to_string(), idx);
        idx
    }

    /// Add a directed edge from_file → to_file with the given edge_type.
    pub fn add_edge(&mut self, from_file: &str, to_file: &str, edge_type: &str) {
        let from = self.get_or_insert_node(from_file);
        let to = self.get_or_insert_node(to_file);
        // Avoid duplicate edges (same from, to, type).
        if !self
            .graph
            .edges(from)
            .any(|e| e.target() == to && e.weight() == edge_type)
        {
            self.graph.add_edge(from, to, edge_type.to_string());
        }
    }

    /// Remove all edges where from_file or to_file matches `file_path`.
    pub fn remove_edges_for_file(&mut self, file_path: &str) {
        let Some(&node) = self.node_index.get(file_path) else {
            return;
        };
        // Collect edges to remove (petgraph doesn't support remove-during-iteration).
        let to_remove: Vec<_> = self
            .graph
            .edges(node)
            .chain(
                self.graph
                    .edges_directed(node, petgraph::Direction::Incoming),
            )
            .map(|e| e.id())
            .collect();
        for eid in to_remove {
            self.graph.remove_edge(eid);
        }
    }

    /// BFS forward (imports): returns `(file_path, depth)` pairs reachable from
    /// `start` up to `max_depth` hops, filtered by `edge_types` when provided.
    pub fn bfs_forward(
        &self,
        start: &str,
        max_depth: usize,
        edge_types: Option<&[&str]>,
    ) -> Vec<(String, usize)> {
        self.bfs_inner(start, max_depth, edge_types, petgraph::Direction::Outgoing)
    }

    /// BFS reverse (importers): files that (transitively) import `start`.
    pub fn bfs_reverse(
        &self,
        start: &str,
        max_depth: usize,
        edge_types: Option<&[&str]>,
    ) -> Vec<(String, usize)> {
        self.bfs_inner(start, max_depth, edge_types, petgraph::Direction::Incoming)
    }

    fn bfs_inner(
        &self,
        start: &str,
        max_depth: usize,
        edge_types: Option<&[&str]>,
        direction: petgraph::Direction,
    ) -> Vec<(String, usize)> {
        let Some(&start_node) = self.node_index.get(start) else {
            return vec![];
        };
        let mut visited = std::collections::HashSet::new();
        visited.insert(start_node);
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((start_node, 0usize));
        let mut results = Vec::new();

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            for edge in self.graph.edges_directed(node, direction) {
                let neighbor = match direction {
                    petgraph::Direction::Outgoing => edge.target(),
                    petgraph::Direction::Incoming => edge.source(),
                };
                // Filter by edge type if requested.
                if let Some(types) = edge_types {
                    if !types.contains(&edge.weight().as_str()) {
                        continue;
                    }
                }
                if visited.insert(neighbor) {
                    let path = self.graph[neighbor].clone();
                    results.push((path, depth + 1));
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }
        results
    }
}

impl Default for ImportGraph {
    fn default() -> Self {
        Self::new()
    }
}
