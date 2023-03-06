use std::sync::Arc;

use balancer::Balancer;

fn main() {
    // Initialize balancer
    let universe = Arc::new(mpi::initialize().unwrap());
    let verbose = false;
    let balancer = Balancer::new(universe, verbose);

    // Get relevant portion of data on this node
    let data: Vec<f64> = {
        if balancer.rank == 0 {
            let data = (0..=10).map(|x| x as f64 / 10.0).collect();
            balancer.distribute(Some(data)).unwrap()
        } else {
            balancer.distribute(None).unwrap()
        }
    };

    for i in 1..5 {
        let mut multiple = if balancer.rank == 0 { i as f64 } else { 0.0 };
        balancer.synchronize_value(&mut multiple);
        let task = |x: &f64| multiple * x;
        balancer.work(&data, task);
        let output = balancer.collect();
        if balancer.rank == 0 {
            println!("rank 0 got {output:?}");
        }
    }
}
