use rand::seq::SliceRandom;
use rand::thread_rng;

pub fn k_means_1D(data: &[f64], k: usize, max_iterations: usize, tolerance: f64) -> (Vec<f64>, Vec<usize>) {
    // Step 1: Initialize random cluster centers
    let mut rng = thread_rng();
    let mut centers: Vec<f64> = data.choose_multiple(&mut rng, k).cloned().collect();

    let mut assignments = vec![0; data.len()]; // Cluster assignments for each data point
    let mut iter = 0;

    // Step 2: Iteratively update clusters
    loop {
        let mut changes = 0;

        // Step 3: Assign points to the nearest cluster center
        for (i, &point) in data.iter().enumerate() {
            let closest_center = centers
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (*a - point).abs().partial_cmp(&(*b - point).abs()).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            if assignments[i] != closest_center {
                changes += 1;
                assignments[i] = closest_center;
            }
        }

        // Step 4: Recompute cluster centers as the mean of assigned points
        let mut new_centers = vec![0.0; k];
        let mut counts = vec![0; k];

        for (assignment, &point) in assignments.iter().zip(data.iter()) {
            new_centers[*assignment] += point;
            counts[*assignment] += 1;
        }

        for i in 0..k {
            if counts[i] > 0 {
                new_centers[i] /= counts[i] as f64;
            }
        }

        // Step 5: Check for convergence
        let max_shift = centers
            .iter()
            .zip(&new_centers)
            .map(|(old, new)| (old - new).abs())
            .fold(0.0, f64::max);

        centers = new_centers;

        iter += 1;
        if max_shift < tolerance || changes == 0 || iter >= max_iterations {
            break;
        }
    }

    (centers, assignments)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_means_1D() {
        let data = vec![1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2];
        let k = 3;
        let max_iterations = 100;
        let tolerance = 1e-4;

        // Run the K-means algorithm on the test data
        let (centers, assignments) = k_means_1D(&data, k, max_iterations, tolerance);

        // Verify the correct number of cluster centers
        assert_eq!(centers.len(), k, "Number of centers should be equal to k");

        // Verify that the assignments make sense
        for (i, &point) in data.iter().enumerate() {
            let assigned_center = centers[assignments[i]];
            let distance_to_assigned_center = (point - assigned_center).abs();

            // Make sure that the point is closer to its assigned center than to any other center
            for &center in &centers {
                if center != assigned_center {
                    let distance_to_other_center = (point - center).abs();
                    assert!(
                        distance_to_assigned_center <= distance_to_other_center,
                        "Point {} is not closest to its assigned center",
                        point
                    );
                }
            }
        }

        println!("Cluster centers: {:?}", centers);
        println!("Cluster assignments: {:?}", assignments);
    }
}