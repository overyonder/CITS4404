# CITS4404_Project

## Notes
- Using C++11

## Sections
- Evolutionary Algorithm
- Pong Simulation
  - Done
  - Tested
- Neural Net Evaluator
  - Done
  - Partly tested

## Evolutionary Algorithm
- Should handle real-valued genomes represented as vector<double>
- Genome length must not change
- Recommend giving each individual a unique integer ID
  - Allows representation of population as map<int, vector<double>> (map from id to genome)
- Be aware of future possibility of adding logging (so we can reconstruct the evolution)
- Beware of growing NN weights to be too large. Investigate appropriate bounds on gene values.

## Pong Simulation
- Arbitrary distance units, velocities in units/tick
- Recommend mental image of 60 ticks/second to make animation (if we do it) easy later
- Construct a game instance
- Register player controller objects (one per player)
  - These will later contain the Neural Net evaluation
  - Must have a .tick() method that will be run after each simulation tick
    - On execution, reads game state, runs NN and returns desired velocity
  - Must normalize input values to between -1.0 and 1.0 for NN, and convert desired velocity produced by NN to units/tick
    - Recommend normalize such that -1.0 = far left of rectangular field and 1.0 = far right
    - Hence, vertical range is -0.75 to 0.75 or something
  - Rotate field for second player
    - Recommend coordinates from field centre, then you can just negate all coordinate/velocity values
- Remember we want to add paddle velocity to ball velocity on bounce
- First to 11 or something like that

## Neural Net
- Weights represented as single vector<double>
  - For each layer, for each neuron, for each edge to previous layer, all concatenated
  - Last weight for each neuron is to a "virtual" neuron with value 1.0
- Layers fixed at 8-8-4-2-1, given as vector<int>
- Constructor will likely take vectors representing layers and weights and then have an .eval(vector<double> inputs) function
