In this assignment we take a look at the KTH fishing game. There is one player in this game who is a diver out at sea (see figure 1). You are in command of the diver and your goal is to get the highest score possible, obtained through catching some fishes by avoiding others. The game is over whenever you catch the king fish.

The diver can move (UP, DOWN, LEFT, RIGHT). There are two types of fish in the game, jelly fish and one single gold fish. There are different (possibly negative) rewards associated with catching each type of fish. The fishes do not move. A fish is caught once its position coincides with the diver’s position. An episode of the game finishes when the king fish is caught.

The position of the fish is not known to the diver. At each time instance, the diver knows it’s own position and when they perform an action, they can observe the immediate reward. The diver goes through multiple episodes of the game with fixed fish positions and the goal for the student is to implement the code for the diver agent that allows it to find a policy that maximizes the reward in the game.



**Input:**
The interface with the judge is the player file player.py. A sample scenario is provided in the code skeleton (the settings.yml file).

**Output:**
Messages printed on the standard output. 

