use crate::{
    constants::{ELITE_COUNT, MAX_SCORE, POPULATION_SIZE},
    game::GameState,
    net::Net,
    paddle::Paddle,
};
use rand::Rng;
use std::fs;
use std::io;

const BEST_NET_FILE: &str = "best_net.json";

/// Runs the main genetic algorithm evolution loop.
pub fn run_evolution(generations: u32) -> io::Result<()> {
    println!("Running for {} generations...", generations);

    // Initialize population with random networks
    let mut population: Vec<Net> = (0..POPULATION_SIZE).map(|_| Net::new()).collect();

    for gen in 0..generations {
        // --- Fitness Evaluation ---
        // Each network's fitness is determined by (plays, wins)
        // plays = returns + shots
        let mut fitness_scores: Vec<(i32, i32)> = vec![(0, 0); POPULATION_SIZE];

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
        let mut ranked_indices: Vec<usize> = (0..POPULATION_SIZE).collect();

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

        // --- Breeding ---
        let mut next_generation = Vec::with_capacity(POPULATION_SIZE);

        // 1. Elitism: Keep the best-performing networks (the elites)
        let elites: Vec<Net> = ranked_indices
            .iter()
            .take(ELITE_COUNT)
            .map(|&index| population[index].clone())
            .collect();

        next_generation.extend(elites.iter().cloned());

        // 2. Crossover: Breed new networks from the elites
        for i in 0..ELITE_COUNT {
            for j in (i + 1)..ELITE_COUNT {
                if next_generation.len() < POPULATION_SIZE {
                    let child = Net::crossover(&elites[i], &elites[j]);
                    next_generation.push(child);
                }
            }
        }

        // 3. Mutation: Fill the remaining spots with mutations of the elites
        let mut rng = rand::thread_rng();
        while next_generation.len() < POPULATION_SIZE {
            let mut parent = elites[rng.gen_range(0..ELITE_COUNT)].clone();
            parent.mutate();
            next_generation.push(parent);
        }

        population = next_generation;
    }

    // After all generations, re-evaluate the final population to find the true best network.
    println!("\n--- Final Evaluation ---");
    let mut final_fitness_scores: Vec<(i32, i32)> = vec![(0, 0); POPULATION_SIZE];
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

    let mut final_ranked_indices: Vec<usize> = (0..POPULATION_SIZE).collect();
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
