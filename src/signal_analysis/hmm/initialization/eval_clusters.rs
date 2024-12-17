use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::fmt;


use super::kmeans::k_means_1_d;



#[derive(Debug, Clone, PartialEq)]
pub enum ClusterEvaluationMethod {
    Silhouette,
    SimplifiedSilhouette,
}

impl ClusterEvaluationMethod {
    pub fn default() -> Self {
        Self::Silhouette
    }
}

// Implement Display for ClusterEvaluationMethod
impl fmt::Display for ClusterEvaluationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClusterEvaluationMethod::Silhouette => write!(f, "Silhouette"),
            ClusterEvaluationMethod::SimplifiedSilhouette => write!(f, "Simplified Silhouette"),
        }
    }
}

// Helper struct to represent an unordered pair
#[derive(Eq, PartialEq, Debug, Clone)]
struct UnorderedPair((usize, usize), (usize, usize));

impl UnorderedPair {
    fn new(cluster_point_pair1: (usize, usize), cluster_point_pair2: (usize, usize)) -> Self {
        if cluster_point_pair1.0 < cluster_point_pair2.0 {
            return UnorderedPair(cluster_point_pair1, cluster_point_pair2);
        }

        if cluster_point_pair1.0 > cluster_point_pair2.0 {
            return UnorderedPair(cluster_point_pair1, cluster_point_pair2);
        }

        // Else, fist idx is equal
        if cluster_point_pair1.1 > cluster_point_pair2.1 {
            return UnorderedPair(cluster_point_pair1, cluster_point_pair2);
        }

        UnorderedPair(cluster_point_pair1, cluster_point_pair2)
    }
}

impl Hash for UnorderedPair {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.0.hash(state);
        self.0.1.hash(state);
        self.1.0.hash(state);
        self.1.1.hash(state);
    }
}

// Function to calculate the silhouette score
pub fn silhouette_score_1_d(cluster_means: &Vec<f64>, assignments: &Vec<usize>, data: &[f64]) -> f64 {
    let num_clusters = cluster_means.len();

    // Divide data points into their clusters
    let mut divided_clusters: Vec<Vec<f64>> = vec![Vec::new(); num_clusters];
    data.iter()
        .zip(assignments)
        .for_each(|(value, &assignment)| divided_clusters[assignment].push(*value));

    // If any of the clusters is empty return -inf
    if divided_clusters.iter().any(|cluster| cluster.is_empty()) {return f64::NEG_INFINITY;}

    

    // Compute and store intra-cluster and inter-cluster distances
    let mut distances = HashMap::<UnorderedPair, f64>::new();
    for (cluster1_id, cluster1_values) in divided_clusters.iter().enumerate() {
        for (cluster2_id, cluster2_values) in divided_clusters.iter().enumerate() {
            for (value1_id, value1) in cluster1_values.iter().enumerate() {
                for (value2_id, value2) in cluster2_values.iter().enumerate() {
                    if cluster1_id == cluster2_id && value1_id == value2_id {continue}

                    let key = UnorderedPair((cluster1_id, value1_id), (cluster2_id, value2_id));
                    let abs_dist = (value1 - value2).abs();
                    distances.insert(key, abs_dist);
                }
            }
        }
    }

    // Compute silhouette scores for each point
    let mut total_silhouette_score = 0.0;
    for (cluster_id, cluster_values) in divided_clusters.iter().enumerate() {
        let curr_cluster_len = cluster_values.len();
        
        for point_idx in 0..curr_cluster_len {
            // Calculate intra-cluster distance (a(i))
            let mut intra_cluster_distance = 0.0;
            
            for other_idx in 0..curr_cluster_len {
                if point_idx == other_idx {
                    continue;
                }
                intra_cluster_distance += distances[&UnorderedPair::new((cluster_id, point_idx), (cluster_id, other_idx))];
            }
            let a_i = if curr_cluster_len > 0 {
                intra_cluster_distance / curr_cluster_len as f64
            } else {
                0.0
            };

            // Calculate inter-cluster distance (b(i)) as the minimum distance to another cluster
            let mut b_i = f64::MAX;
            for (other_cluster_id, other_cluster_values) in divided_clusters.iter().enumerate() {
                if cluster_id == other_cluster_id {
                    continue;
                }
                let mut inter_cluster_distance = 0.0;
                let other_cluster_len = other_cluster_values.len();
                for other_point_idx in 0..other_cluster_len {
                    inter_cluster_distance += distances[&UnorderedPair::new((cluster_id, point_idx), (other_cluster_id, other_point_idx))];
                }
                let avg_inter_cluster_distance = inter_cluster_distance / other_cluster_values.len() as f64;
                if avg_inter_cluster_distance < b_i {
                    b_i = avg_inter_cluster_distance;
                }
            }

            // Calculate silhouette score for this point
            let s_i = (b_i - a_i) / a_i.max(b_i);
            total_silhouette_score += s_i;
        }
    }

    // Average silhouette score across all points
    total_silhouette_score / data.len() as f64

}

// Compute the simplidied silhouette score. A lot faster.
pub fn simplified_silhouette_score_1_d(cluster_means: &Vec<f64>, assignments: &Vec<usize>, data: &[f64]) -> f64 {
    // Divide data points into their clusters
    let num_clusters = cluster_means.len();
    let mut divided_clusters: Vec<Vec<f64>> = vec![Vec::new(); num_clusters];
    data.iter()
        .zip(assignments)
        .for_each(|(value, &assignment)| divided_clusters[assignment].push(*value));

    // If any of the clusters is empty return -inf
    if divided_clusters.iter().any(|cluster| cluster.is_empty()) {return f64::NEG_INFINITY;}
    

    // Initialize total silhouette score
    let mut total_silhouette_score = 0.0;

    // Iterate over each point to calculate s'(i)
    for (i, &point) in data.iter().enumerate() {
        let cluster_id = assignments[i];
        let a_i = (point - cluster_means[cluster_id]).abs();

        // Calculate b'(i) as the minimum distance to any other cluster center
        let b_i = cluster_means
            .iter()
            .enumerate()
            .filter(|&(other_cluster_id, _)| other_cluster_id != cluster_id)
            .map(|(_, &other_mean)| (point - other_mean).abs())
            .fold(f64::MAX, f64::min);

        // Calculate simplified silhouette score for this point
        let s_i = (b_i - a_i) / a_i.max(b_i);
        total_silhouette_score += s_i;
    }

    // Average silhouette score across all points
    total_silhouette_score / data.len() as f64
}

pub fn silhouette_analysis(
    sequence: &[f64], 
    min_k: usize, 
    max_k: usize,
    max_iterations: usize,
    tolerance: f64,
    simplified: bool, 
) -> Result<(usize, f64), ClusterEvalError> {
    let mut left = min_k;
    let mut right = max_k;
    let mut best_k = min_k;
    let mut best_score = f64::NEG_INFINITY;

    while left <= right {
        let mid = (left + right) / 2;

        // Calculate silhouette score for mid and mid + 1 clusters
        let (cluster_means_mid, cluster_assignments_mid) = k_means_1_d(sequence, mid, max_iterations, tolerance);
        let score_mid = if simplified {
            silhouette_score_1_d(&cluster_means_mid, &cluster_assignments_mid, sequence)
        } else {
            simplified_silhouette_score_1_d(&cluster_means_mid, &cluster_assignments_mid, sequence)
        };

        let (cluster_means_next, cluster_assignments_next) = k_means_1_d(sequence, mid + 1, max_iterations, tolerance);
        let score_next = if simplified {
            simplified_silhouette_score_1_d(&cluster_means_next, &cluster_assignments_next, sequence)
        } else {
            simplified_silhouette_score_1_d(&cluster_means_next, &cluster_assignments_next, sequence)
        };

        // Track the best score and corresponding k
        if score_mid > best_score {
            best_score = score_mid;
            best_k = mid;
        }

        // Adjust search range
        if score_mid < score_next {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    if best_score == f64::NEG_INFINITY {return Err(ClusterEvalError::AllTriesFailed)}
    Ok((best_k, best_score))
}

#[derive(Debug, Clone)]
pub enum ClusterEvalError{
    AllTriesFailed,
}