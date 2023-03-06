use std::cell::Cell;
use std::sync::Arc;

use mpi::collective::SystemOperation;
use mpi::environment::Universe;
use mpi::topology::{Communicator, SystemCommunicator};
use mpi::traits::*;
use rayon::prelude::*;

pub use mpi::traits::Equivalence;

/// This struct helps manage compute on a given node and across nodes
pub struct Balancer<O> {
    // This has a custom drop impl which calls MPI_FINALIZE so it needs to hang around
    #[allow(unused)]
    universe: Arc<Universe>,
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
    pub fn new(universe: Arc<Universe>, verbose: bool) -> Self {
        // Initialize mpi
        let world = universe.world();

        // This is the maximum number of `JoinHandle`s allowed.
        // Set equal to available_parallelism minus reduce (user input)
        let workers: usize = std::thread::available_parallelism().unwrap().get();

        // This is the node id and total number of nodes
        let rank: usize = world.rank() as usize;
        let size: usize = world.size() as usize;

        if rank == 0 && verbose {
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
    pub fn get_subset<'b, I>(&self, items: &'b [I]) -> &'b [I]
    where
        I: Send + Sync,
    {
        // Gather and return local set of items
        let chunk_size = div_ceil(items.len(), self.size);
        let (l, r) = (
            self.rank * chunk_size,
            ((self.rank + 1) * chunk_size).min(items.len()),
        );
        &items[l..r]
    }

    /// Calculates local set of items on which to work on, and then works on it.
    pub fn work_subset<'b, I, F>(&self, items: &'b [I], work: F)
    where
        I: Send + Sync,
        F: Fn(&'b I) -> O + Send + Sync,
        O: Send,
    {
        // Gather and return local set of items
        let chunk_size = div_ceil(items.len(), self.size);
        let (l, r) = (
            self.rank * chunk_size,
            ((self.rank + 1) * chunk_size).min(items.len()),
        );
        let our_items: &'b [I] = &items[l..r];

        // Carry out work on local node threads
        let output = our_items.into_par_iter().map(|i| work(i)).collect();

        // Save work
        self.work.set(Some(LocalWork { output }));
    }

    /// Works on the entire set provided
    pub fn work<'b, I, F>(&self, items: &'b [I], work: F)
    where
        I: Send + Sync,
        F: Fn(&'b I) -> O + Send + Sync,
        O: Send,
    {
        // Carry out work on local node threads
        let output = items.into_par_iter().map(|i| work(i)).collect();

        // Save work
        self.work.set(Some(LocalWork { output }));
    }

    /// Distributes items for work
    pub fn distribute<'b, I>(&self, items: Option<Vec<I>>) -> Option<Vec<I>>
    where
        I: Send + Sync + Equivalence,
    {
        // Gather and return local set of items
        if self.rank == 0 && self.size > 1 {
            let mut items = items.unwrap();
            let chunk_size = div_ceil(items.len(), self.size);
            let mut rank = 1;
            let ours: Vec<I> = items.drain(..chunk_size).collect();
            while !items.is_empty() {
                let theirs: Vec<I> = items.drain(..chunk_size.min(items.len())).collect();
                self.world.process_at_rank(rank).send(&theirs);
                rank += 1
            }
            self.world.barrier();
            Some(ours)
        } else {
            let (ours, _status) = self.world.process_at_rank(0).receive_vec();
            self.world.barrier();
            Some(ours)
        }
    }

    pub fn collect(&self) -> Option<Vec<O>> {
        // Get rank output
        let work = self.work.replace(None)?;
        let mut output: Vec<O> = work.output;

        if self.rank == 0 {
            // Collect outputs from all other ranks
            for rank in 1..self.size {
                let (mut rank_output, _status) =
                    self.world.process_at_rank(rank as i32).receive_vec::<O>();
                output.append(&mut rank_output);
            }

            // If rank 0 return output
            self.world.barrier();
            Some(output)
        } else {
            self.world.process_at_rank(0).send(&output);
            // If not rank 0 return None
            self.world.barrier();
            None
        }
    }

    /// Waits for all threads to finish across all ranks.
    pub fn barrier(&self) {
        self.world.barrier();
    }

    pub fn world(&self) -> &SystemCommunicator {
        &self.world
    }

    pub fn synchronize_value<T: Equivalence>(&self, value: &mut T) {
        self.world.process_at_rank(0).broadcast_into(value);
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
}
