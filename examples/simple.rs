use std::sync::Arc;

use balancer::Balancer;
use mpi::{environment::Universe, traits::Communicator};

fn main() {
    let universe = Arc::new(mpi::initialize().unwrap());
    experiment(universe.clone());
    experiment(universe.clone());
    if universe.world().rank() == 0 {
        println!("done!");
    }
}

fn experiment(universe: Arc<Universe>) {
    // Get all data on all nodes
    let data: Vec<f64> = (0..100_000).map(|x| x as f64 / 100_000.0).collect();

    // Define task
    let work = |x: &f64| x * x;

    // Initialize balancer
    let verbose = false;
    let balancer = Balancer::new(universe, verbose);
    
    // Work on our subset and collect result on root node
    balancer.work_subset(&data, work);
    let output = balancer.collect();

    // That's it!
    // Let's do some verification
    if balancer.rank == 0 {
        for (expected, actual) in data.iter().map(work).zip(output.as_ref().unwrap()) {
            assert_eq!(expected, *actual);
        }
    }
}
