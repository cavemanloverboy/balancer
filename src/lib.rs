use std::cell::Cell;

use mpi::environment::Universe;
use mpi::topology::{Communicator, SystemCommunicator};
use mpi::traits::*;
use rayon::prelude::*;

/// This struct helps manage compute on a given node and across nodes
pub struct Balancer<O> {
    // This has a custom drop impl which calls MPI_FINALIZE so it needs to hang around
    #[allow(unused)]
    universe: Universe,
    world: SystemCommunicator,
    pub workers: usize,
    pub rank: usize,
    pub size: usize,
    work: Cell<Option<LocalWork<O>>>,
}

impl<O> Balancer<O>
where
    O: Send + Equivalence,
{
    /// Constructs a new `Balancer` from an `mpi::SystemCommunicator` a.k.a. `world`.
    pub fn new() -> Self {
        // Initialize mpi
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        // This is the maximum number of `JoinHandle`s allowed.
        // Set equal to available_parallelism minus reduce (user input)
        let workers: usize = std::thread::available_parallelism().unwrap().get();

        // This is the node id and total number of nodes
        let rank: usize = world.rank() as usize;
        let size: usize = world.size() as usize;

        if rank == 0 {
            println!("--------- Balancer Activated ---------");
            println!("            Nodes : {size}");
            println!(" Workers (rank 0) : {workers} ");
            println!("--------------------------------------");
        }
        Balancer {
            universe,
            world,
            workers,
            rank,
            size,
            work: Cell::new(None),
        }
    }

    /// Calculates local set of items on which to work on.
    pub fn work_local<'b, I, F>(&self, items: &'b [I], work: F)
    where
        I: Send + Sync,
        F: Fn(&'b I) -> O + Send + Sync,
        O: Send,
    {
        // Gather and return local set of items
        let total = items.len();
        let chunk_size = div_ceil(items.len(), self.size);
        let (l, r) = (
            self.rank * chunk_size,
            ((self.rank + 1) * chunk_size).min(items.len()),
        );
        let our_items: &'b [I] = &items[l..r];

        // Carry out work on local node threads
        let output = our_items.into_par_iter().map(|i| work(i)).collect();

        // Save work
        self.work.set(Some(LocalWork { output, total }));
    }

    pub fn collect(&self) -> Option<Vec<O>> {
        // Get rank output
        let work = self.work.replace(None)?;
        let mut output: Vec<O> = work.output;

        if self.rank == 0 {
            // Allocate for all output
            output.reserve_exact(work.total - output.len());

            // Collect outputs from all other ranks
            for rank in 1..self.size {
                let (mut rank_output, _status) =
                    self.world.process_at_rank(rank as i32).receive_vec::<O>();
                output.append(&mut rank_output);
            }

            // If rank 0 return output
            Some(output)
        } else {
            self.world.process_at_rank(0).send(&output);
            // If not rank 0 return None
            None
        }
    }

    /// Waits for all threads to finish across all ranks.
    pub fn barrier(&mut self) {
        self.world.barrier();
    }
}

fn div_ceil(a: usize, b: usize) -> usize {
    // Note to self:
    // If a is zero this will be zero.
    // If b is zero this will panic.
    (a + b - 1) / b
}

pub struct LocalWork<O> {
    output: Vec<O>,
    total: usize,
}
