use crate::{
    constants::{ELITE_COUNT, MAX_SCORE, POPULATION_SIZE},
    game::GameState,
    net::Net,
    paddle::Paddle,
};
use rand::{Rng, thread_rng};
use std::fs;
use std::io;

const BEST_NET_FILE: &str = "best_net.json";

/// Runs the main genetic algorithm evolution loop.
pub fn run_evolution_stack(generations: u32) -> io::Result<()> {
    println!("Running for {} generations...", generations);

    // Initialize population on the stack
    let mut population = [Net::default(); POPULATION_SIZE];
    for net in &mut population {
        *net = Net::new();
    }

    for gen in 0..generations {
        // --- Fitness Evaluation ---
        // Each network's fitness is determined by (plays, wins)
        // plays = returns + shots
        let mut fitness_scores = [(0, 0); POPULATION_SIZE];

        for i in 0..POPULATION_SIZE {
            for j in (i + 1)..POPULATION_SIZE {
                let game = run_training_game(&population[i], &population[j]);

                // Update plays (returns + shots)
                fitness_scores[i].0 += (game.returns[0] + game.shots[0]) as i32;
                fitness_scores[j].0 += (game.returns[1] + game.shots[1]) as i32;

                // Update wins
                if game.score[0] > game.score[1] {
                    fitness_scores[i].1 += 1;
                } else {
                    fitness_scores[j].1 += 1;
                }
            }
        }

        // --- Ranking ---
        let mut ranked_indices = [0; POPULATION_SIZE];
        for (i, v) in ranked_indices.iter_mut().enumerate() {
            *v = i;
        }

        // Sort indices based on fitness scores (plays descending, then wins descending)
        ranked_indices.sort_unstable_by(|&a, &b| {
            fitness_scores[b]
                .0
                .cmp(&fitness_scores[a].0)
                .then(fitness_scores[b].1.cmp(&fitness_scores[a].1))
        });

        let best_fitness = fitness_scores[ranked_indices[0]];
        println!(
            "\n--- Generation {} --- Best fitness: (plays: {}, wins: {})",
            gen, best_fitness.0, best_fitness.1
        );

        // --- Breeding (in-place) ---
        let elite_indices = &ranked_indices[..ELITE_COUNT];
        let non_elite_indices = &ranked_indices[ELITE_COUNT..];

        // The elites are preserved. We will overwrite the non-elites.
        let mut non_elite_iter = non_elite_indices.iter().copied();

        // 1. Crossover: Overwrite some non-elites with children of elites.
        'crossover: for i in 0..ELITE_COUNT {
            for j in (i + 1)..ELITE_COUNT {
                if let Some(dest_index) = non_elite_iter.next() {
                    let p1 = population[elite_indices[i]]; // It's a copy
                    let p2 = population[elite_indices[j]]; // It's a copy
                    let child = &mut population[dest_index];
                    Net::crossover(&p1, &p2, child);
                } else {
                    break 'crossover; // No more non-elite slots to fill.
                }
            }
        }

        // 2. Mutation: Overwrite the rest of the non-elites with mutated elites.
        let mut rng = thread_rng();
        for dest_index in non_elite_iter {
            let parent_index = elite_indices[rng.gen_range(0..ELITE_COUNT)];
            let mut new_net = population[parent_index]; // It's a copy
            new_net.mutate();
            population[dest_index] = new_net;
        }
        // The `population` vector has now been updated in-place.
    }

    // After all generations, re-evaluate the final population to find the true best network.
    println!("\n--- Final Evaluation ---");
    let mut final_fitness_scores = [(0, 0); POPULATION_SIZE];
    for i in 0..POPULATION_SIZE {
        for j in (i + 1)..POPULATION_SIZE {
            let game = run_training_game(&population[i], &population[j]);
            final_fitness_scores[i].0 += (game.returns[0] + game.shots[0]) as i32;
            final_fitness_scores[j].0 += (game.returns[1] + game.shots[1]) as i32;
            if game.score[0] > game.score[1] {
                final_fitness_scores[i].1 += 1;
            } else {
                final_fitness_scores[j].1 += 1;
            }
        }
    }

    let mut final_ranked_indices = [0; POPULATION_SIZE];
    for (i, v) in final_ranked_indices.iter_mut().enumerate() {
        *v = i;
    }
    final_ranked_indices.sort_unstable_by(|&a, &b| {
        final_fitness_scores[b]
            .0
            .cmp(&final_fitness_scores[a].0)
            .then(final_fitness_scores[b].1.cmp(&final_fitness_scores[a].1))
    });

    let best_net = &population[final_ranked_indices[0]];
    let best_fitness = final_fitness_scores[final_ranked_indices[0]];
    println!(
        "Best network found with fitness: (plays: {}, wins: {})",
        best_fitness.0, best_fitness.1
    );
    println!("Saving best network...");
    save_best_net(best_net, BEST_NET_FILE)?;

    Ok(())
}

/// Runs a game simulation between two nets and returns the final game state.
fn run_training_game(net1: &Net, net2: &Net) -> GameState {
    let paddle1 = Paddle::new_with_net(net1.clone());
    let paddle2 = Paddle::new_with_net(net2.clone());
    let mut game = GameState::new_with_paddles(paddle1, paddle2);

    // Simulate until a winner is decided
    while game.score[0] < MAX_SCORE && game.score[1] < MAX_SCORE {
        game.tick();
    }

    game
}

/// Saves a neural network to a file.
pub fn save_best_net(net: &Net, path: &str) -> io::Result<()> {
    let json_data = serde_json::to_string_pretty(net).expect("Failed to serialize net");
    fs::write(path, json_data)
}
