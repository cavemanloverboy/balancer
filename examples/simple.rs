use balancer::Balancer;

fn main() {
    // Get relevant portion of data on this node
    let data: Vec<f64> = (0..100_000).map(|x| x as f64 / 100_000.).collect();

    // Define task
    let work = |x: &f64| x * x;

    // Initialize balancer, work and collect
    let balancer = Balancer::new();
    balancer.work_local(&data, work);
    let output = balancer.collect();
    if balancer.rank == 0 {
        let output = output.unwrap();
        println!(
            "rank {} has output [{}, {}, {}, ..] with length {}",
            balancer.rank,
            output[0],
            output[1],
            output[2],
            output.len(),
        );
    } else {
        println!("rank {} has output {output:?}", balancer.rank);
    }
}
