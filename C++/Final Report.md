# CITS4404 : Artificial Intelligence and Adaptive Algorithms

# Final Report

## Authors redacted

## Submission Date: 03/11/2015

**Table of Contents**

[1 	Introduction](#1-introduction)

[1.1 	Problem Context](#1.1-problem-context)

[2 	Implementation Details and Strategies](#2-implementation-details-and-strategies)

[3 	Structure and Design of Algorithm](#3-structure-and-design-of-algorithm)

[3.1 	Representation](#3.1-representation)

[3.2 	Evaluation](#3.2-evaluation)

[3.3 	Selection](#3.3-selection)

[3.4 	Variation](#heading=h.3szyghb49fyl)

[3.4.1 	Crossover](#3.4.1-crossover)

[3.4.2 	Combination](#3.4.2-mutation)

[4 	Theoretical and Experimental Analysis](#heading=h.x7ft5o302ime)

[5 	Appendices](#heading=h.ll6ona8zp8ez)

[5.1 	Appendix A: Name](#heading=h.6zeur2h0j2ub)

[5.2 	Appendix B: Name](#heading=h.enilg3z3qu5c)

[5.3 	Appendix C: Name](#heading=h.aim3n6o9ich)

[6 	References](#6-references)

# **1 	Introduction** {#1-introduction}

## **1.1 	Problem Context** {#1.1-problem-context}

The problem chosen for this project is the game of Pong and the goal is to evolve a superior pong player through the technique of Evolutionary Algorithms. More specifically this project aims to use Evolutionary Algorithms to evolve an optimal set of weightings for a Neural Net which allows the pong paddle to learn from previous games and improve its performance. The implementation of this project will result in a final generation of pong paddles with, ideally, superior capabilities to the initial pong paddles.

The game of pong became prevalent in the mid 1970s after its development as an arcade game by Atari (Lowood 2009). The game is similar to a tennis match,  with two paddles, one on either side of the court, and a ball that must be hit between the two. One point is awarded to the opponent each time the paddle misses the ball. The game commences with the ball starting in the middle of the court and moving towards one of the paddles. In different versions of the game, the paddles can be controlled by two human players, one human and one computer player or two computer players. In this project, the game of pong will consist of two computer players.

Of important consideration is the fact that pong paddles are limited to movement along the y axis, whilst the ball can move along both x and y axes. The variables to be considered are the position and velocity of the ball with respect to x and y, and the position and velocity of each paddle with respect to y. The solution of this project will produce the optimal velocity, in terms of y, for each paddle, in order to hit the ball.

# **2 	Implementation Details and Strategies** {#2-implementation-details-and-strategies}

In order to implement this solution, three pieces of code were required \- the Neural Net, the Evolutionary Algorithm and the Pong simulator. Whilst not the focus of the project, the Pong simulator was required to demonstrate the results in a more contextually relevant manner. Each piece of code was written in C++ and was shared between team members using GitHub.

A neural net was created which took 8 inputs and produced one output. The inputs to the net were the position of each paddle (sy1, sy2 ), the velocity of each paddle (vy1, vy2), the position of the ball (sxb, syb) and the velocity of the ball (vxb, vyb). The output of the neural net was the ideal velocity of the paddle that was being evolved.

The structure of the neural net consisted of an input layer of eight neurons comprised of the eight variables mentioned above. The second layer consisted of 16 neurons, with a weighting linking each neuron to each of the previous 8 neurons plus a ‘unity weight’. The third layer consisted of 4 neurons each neuron connected to the previous 16 neurons and their own ‘unity weight’. Finally, the output layer consisted of a single neuron whose weightings were linked to each of the four previous neurons and a ‘unity weight’. The value of the output neuron was the value then used directly in the Pong simulator where an output value of 1 indicated maximum velocity ‘forwards’ (positive y direction) and an output value of \-1 indicated maximum velocity ‘backwards’ (negative y direction). This resulted in a total of 217 weightings between neurons which needed to be evolved by the Evolutionary Algorithm.

# **3 	Structure and Design of Algorithm** {#3-structure-and-design-of-algorithm}

## **3.1 	Representation** {#3.1-representation}

The evolutionary algorithm employed in this solution utilised a simple genome \- an array of doubles representing the weightings of the neural network. This strategy is known as neuroevolution, a technique which is inspired by the evolution of biological nervous systems (Lehman & Risto 2013). The motivating benefit for this design decision was that evolution time is drastically reduced by comparison to other genetic programming techniques. 

## **3.2 	Evaluation** {#3.2-evaluation}

The method by which an evolutionary algorithm evaluates individuals in its population is its fitness function. In this solution, the fitness function was a sum of three measures: ‘returns’, ‘shots’, and the number of games won against the other genomes in its tournament set.

A return is, intuitively, added to the score of an individual when their paddle makes contact with the ball. ‘Shot’ is a term which has special meaning in the context of this solution \- any game tick in which the ball is moving towards the opponent’s goal adds a shot to the player’s score. There are 60 ticks per second in the implementation submitted for this project.

It is important to recognise that this fitness function is time-dependent. A genome will likely have a different fitness function in each tournament that it enters because of the fact that it is likely to face different opponents. This means that the algorithm attempts to optimize a player which is good at beating other computer players, rather than some objective standard such as a wall or a simple ball-following paddle. The motivation behind this design decision is that players who learn to beat other evolved players will hopefully develop strategies that make them better at beating human players as well. The reasoning behind this is that evolved players ‘look’ at the movement of their opponent’s paddles. It is therefore possible that an evolved player might try to trick their opponent, as that opponent takes their movement into consideration. These types of strategies were observed in a number of test runs \- paddles would deliberately flinch, and opponents would flinch as a result \- demonstrating far more sophisticated play than a simple ball-following algorithm.

## **3.3 	Selection** {#3.3-selection}

Selection is the process by which an evolutionary algorithm decides which individuals should be allowed to reproduce, or proceed forward into the next generation, based on their evaluated fitness. The selection method employed in this solution is proportional truncation selection. Truncation selection is a type of selection in which the individuals who exceed some cut-off fitness are allowed to reproduce or move forward. The method employed is referred to as proportional because the cut-off fitness was relative to the fitness of the population \- the best square root of the population was kept. More exotic selection functions were tested, but ultimately proved to offer negligible benefits considering the added computation time, or none at all.

The reasoning behind this proportion is that if a child is produced by each possible pair of kept individuals, the total population size of the next generation will be smaller by a proportional amount. This ‘room’ in the population is made up by mutants, which are generated by the mutation of a kept individual or one of their children.

Mathematically this relationship can be represented as follows:

K=P

C=K(K-1)2

N=K+C

\=P+P(P-1)2

\=P+P2

M=P-N

\=P-P+P2

\=P-P2

Where:

P is the number of individuals in the original population,  
	N is the number of individuals in the next generation of the population,  
	K is the number of individuals who are kept in the population,  
	C is the number of children added to the population (the progeny of K), and  
	M is the number of mutants added to the population (mutated from copies of C and K).

## **3.4 	Variation**

As in biological evolution \- the inspiration for evolutionary algorithms \- progress is dependent not just improvement mechanisms of the type outlined in the previous few subsections. The force of change is also required. The entropy of the system must be increased in order for improvements to be skimmed from the top, progressing the population forward through the generations.

Consequently, the methods of introducing entropy into the system are of vital importance to evolutionary algorithms, and must be carefully chosen to introduce change in a way which promotes progress.

Two primary modes of variation are used in this solution. Crossover is the process of creating a child genome from parent genomes, and mutation is the process of mutating a copy of another genome.

### **3.4.1 	Crossover** {#3.4.1-crossover}

In this solution a child was always the offspring of two parents, as in the biological world. It is important to note that it is possible to generate any arbitrary number of children from an arbitrary number of parents in evolutionary algorithms. The design decision of two parents to one child was made because of the mathematical relationship outlined in the previous section which greatly simplifies the process.

In this solution, crossover is a generative process which creates a child gene by gene corresponding to the same gene in its two parents, where a gene is a given weighting in the neural net. 

Each gene of a given child is taken as a point, normally distributed, on the line between the values of that gene in its two parents.

### **3.4.2 	Mutation** {#3.4.2-mutation}

A mutant, in this solution, is a copy of another genome in the population with a mutation on a single one of its genes. The cloned genome is randomly selected from the population, and the gene to be mutated is also randomly selected. Finally, that gene is set to a new value based on a normal distribution with mean set to the previous value of the gene.

# **4 	Theoretical and Experimental Analysis**

## **4.1 	Research on Methods of Evaluation**

Prior to the project being ready for testing, research was undertaken to determine what methods of evaluation may be most suitable for the problem context. Initially, a ‘round-robin’ style tournament was considered, in which every evolved pong paddle of a single generation would play against every other paddle of that generation (Byl 2006). In this tournament, each pair of paddles would play two games, with the ball being ‘sent’ initially to a different paddle in each game. The games would end when a single point was scored. This ‘round-robin’ style, whilst longer than an elimination tournament, ensures that every paddle is compared against all others in its generation. This avoids the risk of stronger generation members being removed early in testing purely due to being paired, by chance, against another strong member. The disadvantage of a ‘round-robin’ tournament is the increased amount of time required to play every paddle against each other in a single generation. This disadvantage was viewed to be significant due to the time constraints of the project and therefore, this method of evaluation was discarded.

In addition to the time requirements of a tournament style form of evaluation, Nakashima et al 2006 presented a useful perspective on the reliability of results taken from matches, in the context of RoboCup Soccer. This study found variation in results when two teams were played repeatedly against each other, highlighting how strong teams could still be eliminated during the evolution of generations due to limited evaluation of their performance. This suggested that the comparison of two Pong paddles over just a few games may not be sufficient, and ultimately the paper proposed a system of taking the average result of 20 games between the same two paddles. Once again, this method was deemed to be implausible under the time constraints. 

Further research revealed several papers such as Kuo and Ou 2009 and Jong-Hwan et al. 2009 that had used fitness functions to evaluate the performance of evolved soccer players and teams in the context of the RoboCup Soccer game. Unfortunately, these fitness functions relied on metrics such as how far a ball was kicked and the distance the ball was from the opponent’s goal after being kicked, neither of which were relevant to the game of Pong. This finding further highlighted the limitations the game of Pong presented in terms of evaluation. 

Finally, it was decided that pong paddles would be evaluated within the context of their generation aspects of the fitness function discussed above which considered different metrics to those proposed by the RoboCup soccer papers. More specifically, the results below use the number of plays and the number of wins as methods of evaluation. 

## **4.2 	Evaluation of Chosen Truncation Selection Method**

During evaluation, a variety of selection methods were tested. Overall it was found that the originally decided upon truncation selection, which chooses √(population size) number of players with the highest fitness, was the best for our purposes. Other trialled methods of selection were found to be unsatisfactory, as they did not learn at the same speed as truncation selection.  
The above graph shows the maximum and average number of plays per generation for the players that have been selected for reproduction. One ‘play’ is counted every time a paddle hits the ball. In this case the plays have been calculated across all paddles and all games for the given generation. After approximately 80 generations the system maintains a high level of plays, before a drop at approximately 700 generations. This steep increase in the number of plays between generations 0 and 80 indicate the rapid learning ability of the paddles, as the number of plays reflects the ability of the paddle. 

The number of wins that a player can have in a single generation is, in the case of 24 players, capped at 46 wins. No players win every single game, however the average number of wins for the top players is almost always over half, as indicated in the graph below.

## **4.3 	Variations to Selection Method \- Worst Players Truncation**

To provide a contrast to the Best-Players Truncation method, tests were conducted with a flipped truncation method – the worst √(population size) number of players were selected for reproduction at every generation. The system quickly converged to the point where the players would run away from the ball at the very beginning of play – the only way a paddle could win a game was for the ball to start on a heading towards its opponent. When compared to the default truncation selection, the players selected by the Worst Truncation Selection have an extremely low rate of contiguous plays, at a level of approximately 200 plays on average. The number of wins experienced by the worst players has a steady average and maximum of 23 wins. This seems high for players that are supposedly the worst at their game, however it is an expected result due to the way games are scored \- the left hand player will score even if the ball has not entered its side of the court, if the right hand side player misses the first shot of the game. Results are indicated in Appendix A,

## **4.4 	Variations to Selection Method \- Random Selection with Uniform Probability**

The system was tested with a non-proportional random selection method to see how well it could learn without any selection pressure. While the population size of 24 is small compared to that used with the default truncation selection test, we can see that the system begins to learn some rudimentary tennis skills. However it is slow to learn as those players with a lower fitness have no selective pressure to prevent their breeding. Some spikes in plays and wins are observed, however this particular iteration of the uniform probability random selection is little better than the Worst-Player Truncation Selection above. It is possible that given more generations and a larger population, the random selection will eventually learn to play a decent game of tennis, but there are better methods of selection available, with a more predictable learning curve. Results are indicated in Appendix B. 

## **4.5 	Variations to Selection Method \- Roulette-Wheel Selection**

A simple roulette-wheel (proportional) selection method was tested. This selection method used the size of the population as an indicator for probability – a player with the highest fitness function had a selection probability equal to the population size. The next fittest player's probability was the size of the population minus one, and so on until the weakest player had a probability of 1/total\_probability. Random numbers in the range of 0 – total\_probability were generated and matched to a corresponding portion of the roulette wheel. 

The results of the Roulette-Wheel Selection are more promising than the Random Selection, but also considerably more volatile than the default truncation selection. There are larger spreads of average and maximum wins with the Roulette Wheel. The average and maximum values of the plays for the selected players do not have the same consistency that the default truncation selection did. However, the Roulette Wheel selection allows the possibility for one individual to be chosen for selection more than once unlike the truncation selection. Results are indicated in Appendix C. 

# **5 	Appendices**

## **5.1 	Appendix A**

![Worst Selection Plays.png][image1]  
![Worst Selection Wins.png][image2]

## **5.2 	Appendix B: Name**

![Random Plays.png][image3]  
![Random Wins.png][image4]

## **5.3 	Appendix C: Name**

![Roulette Plays.png][image5]  
![Roulette Wins.png][image6]

# **6 	References** {#6-references}

Byl, J 2006, *Organizing Successful Tournaments,* Human Kinetics.

Kuo, J, Ou, Y 2009, ‘An Evolutionary Fuzzy Behaviour Controller Using Genetic Algorithm in RoboCup Soccer Game’, *Ninth International Conference on Hybrid Intelligent Systems*, vol.1, pp.281-286. Available from: IEEE Xplore.

Jong-Hwan, K, Ye-Hoon, K, Seung-Hwan, C 2009, ‘Evolutionary multi-objective optimization in robot soccer system for education’, *In-Won Park IEEE Computational Intelligence Magazine*, vol.4, no.1, pp.31-41. Available from: IEEE XPlore.

Lehman, J, Miikkulainen, R 2013, *‘Neuroevolution’* Scholarpedia, 8(6):30977.

Lowood, H 2009, ‘Videogames in Computer Space: The Complex History of Pong’, *Annals of the History of Computing, IEEE*, vol.21, no.3, pp.5-19. Available from: IEEE Xplore.

Nakashima, T, Takatani, M, Namikawa, N, Ishibuchi, H, Nii, M 2006 ‘Robust Evaluation of RoboCup Soccer Strategies by Using Match History’*, IEEE International Conference on Evolutionary Computation*’, pp.1195-1201. Available from: IEEE XPlore.
