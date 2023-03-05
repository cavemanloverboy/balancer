use std::sync::Arc;

use balancer::Balancer;
use mpi::{environment::Universe, traits::Communicator};

fn main() {
    let universe = Arc::new(mpi::initialize().unwrap());
    let world = universe.world();
    for _ in 0..1000 {
        if world.rank() == 0 {
            experiment(universe.clone());
        } else {
            helper(universe.clone());
        }
    }
    if world.rank() == 0 {
        println!("done!");
    }
}

fn experiment(universe: Arc<Universe>) {
    // Get relevant portion of data on this node
    let data: Vec<f64> = (0..100_000).map(|x| x as f64 / 100_000.0).collect();

    // Define task
    let work = |x: &f64| x * x;

    // Initialize balancer, work and collect
    let verbose = false;
    let balancer = Balancer::new(universe, verbose);
    let ours = balancer.distribute(Some(data.clone())).unwrap();
    balancer.work(&ours, work);
    let output = balancer.collect();

    // That's it!
    // Let's do some verification
    if balancer.rank == 0 {
        for (expected, actual) in data.iter().map(work).zip(output.as_ref().unwrap()) {
            assert_eq!(expected, *actual);
        }
    }
}

fn helper(universe: Arc<Universe>) {
    // Initialize balancer
    let verbose = false;
    let balancer = Balancer::new(universe, verbose);

    // Get relevant portion of data on this node
    // rank !=0 passes in none
    let data: Vec<f64> = balancer.distribute(None).unwrap();

    // Define task
    let work = |x: &f64| x * x;

    balancer.work(&data, work);
    balancer.collect();
}
