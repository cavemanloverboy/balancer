# balancer: simple multinode parallelism

With the power of `rayon` and `mpi`, internode and intranode parallelism is achieved for simple tasks.

## Example Usage
See `examples/simple.rs`.
```rust
use std::sync::Arc;

use balancer::Balancer;
use mpi::environment::Universe;

fn main() {
    let universe = Arc::new(mpi::initialize().unwrap());
    experiment(universe.clone());
    experiment(universe.clone());
}

fn experiment(universe: Arc<Universe>) {
    // Get relevant portion of data on this node
    let data: Vec<f64> = (0..100_000).map(|x| x as f64 / 100_000.0).collect();

    // Define task
    let work = |x: &f64| x * x;

    // Initialize balancer, work and collect
    let verbose = false;
    let balancer = Balancer::new(universe, verbose);
    balancer.work_local(&data, work);
    let output = balancer.collect();

    // That's it!
    // Let's do some verification
    if balancer.rank == 0 {
        for (expected, actual) in data.iter().map(work).zip(output.as_ref().unwrap()) {
            assert_eq!(expected, *actual);
        }
    }
}
```