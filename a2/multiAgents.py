# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        remainingCapsules = successorGameState.getCapsules()
        remainingFoods = currentGameState.getFood().asList()
        currPosition = currentGameState.getPacmanPosition()

        score = 0
        for i in range(len(newGhostStates)):
            currGhostPosition = newGhostStates[i].getPosition()
            distance = util.manhattanDistance(currGhostPosition, newPos) + 0.0000001
            if newScaredTimes[i] > distance:
                score += 1000 / (distance * 1.0)
            else:
                if distance < 3:
                    score -= 100 / (distance * 1.0)

        for capsulePos in remainingCapsules:
            distance = util.manhattanDistance(capsulePos, newPos) + 0.0000001
            score += 20 / (distance * 1.0)

        for foodPos in remainingFoods:
            distance = util.manhattanDistance(foodPos, newPos) + 0.0000001
            score += 10 / (distance * 1.0)

        if newPos == currPosition:
            score -= 20

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        agentNum = gameState.getNumAgents()
        action, val = self.DFMiniMax(agentNum, 0, gameState)
        return action

    def DFMiniMax(self, agentNum, currDepth, gameState):
        if (currDepth == self.depth * agentNum) or (gameState.isLose()) or (gameState.isWin()):
            return None, self.evaluationFunction(gameState)
        else:
            actions = gameState.getLegalActions(currDepth % agentNum)
            if len(actions) != 0:
                bestAction = None
                if currDepth % agentNum == 0:
                    val = -float("inf")
                else:
                    val = float("inf")
                for action in actions:
                    successor = gameState.generateSuccessor(currDepth % agentNum, action)
                    cost = self.DFMiniMax(agentNum, currDepth + 1, successor)
                    if currDepth % agentNum == 0:
                        if cost[1] > val:
                            val = cost[1]
                            bestAction = action
                    else:
                        if cost[1] < val:
                            val = cost[1]
                            bestAction = action
                return bestAction, val
            else:
                return None, self.evaluationFunction(gameState)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        agentNum = gameState.getNumAgents()
        action, val = self.DFMiniMax(agentNum, 0, gameState, -float("inf"), float("inf"))
        return action

    def DFMiniMax(self, agentNum, currDepth, gameState, alpha, beta):
        if (currDepth == self.depth * agentNum) or (gameState.isLose()) or (gameState.isWin()):
            return None, self.evaluationFunction(gameState)
        else:
            actions = gameState.getLegalActions(currDepth % agentNum)
            if len(actions) != 0:
                bestAction = None
                if currDepth % agentNum == 0:
                    val = -float("inf")
                else:
                    val = float("inf")
                for action in actions:
                    successor = gameState.generateSuccessor(currDepth % agentNum, action)
                    cost = self.DFMiniMax(agentNum, currDepth + 1, successor, alpha, beta)
                    if currDepth % agentNum == 0:
                        if cost[1] > val:
                            val = cost[1]
                            bestAction = action
                        if val >= beta:
                            return bestAction, val
                        alpha = max(alpha, val)
                    else:
                        if cost[1] < val:
                            val = cost[1]
                            bestAction = action
                        if val <= alpha:
                            return bestAction, val
                        beta = min(beta, val)
                return bestAction, val
            else:
                return None, self.evaluationFunction(gameState)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        agentNum = gameState.getNumAgents()
        action, val = self.DFMiniMax(agentNum, 0, gameState)
        return action

    def DFMiniMax(self, agentNum, currDepth, gameState):
        if (currDepth == self.depth * agentNum) or (gameState.isLose()) or (gameState.isWin()):
            return None, self.evaluationFunction(gameState)
        else:
            actions = gameState.getLegalActions(currDepth % agentNum)
            if len(actions) != 0:
                bestAction = None
                if currDepth % agentNum == 0:
                    val = -float("inf")
                else:
                    val = 0
                prob = 1 / (len(actions) * 1.0)
                for action in actions:
                    successor = gameState.generateSuccessor(currDepth % agentNum, action)
                    cost = self.DFMiniMax(agentNum, currDepth + 1, successor)
                    if currDepth % agentNum == 0:
                        if cost[1] > val:
                            val = cost[1]
                            bestAction = action
                    else:
                        val += cost[1] * prob
                return bestAction, val
            else:
                return None, self.evaluationFunction(gameState)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Encourage agent to chase ghost when ghost is scared and remaining
      scared time is less than distance, otherwise run away from ghost if ghost is
      too close. Encourage agent to move close to food and capsule, the closer it gets,
      the higher the score. Agent also gets higher score for less food and capsule
      remaining. Total ghost score has highest weight as it strongly encourages agent to
      chase ghost when ghost is scared, and keeps the agent alive by penalizing when ghost
      is too close and not scared.
    """

    currPos = currentGameState.getPacmanPosition()
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]

    remainingCapsules = currentGameState.getCapsules()
    remainingFoods = currentGameState.getFood().asList()

    score = 0
    ghostScore = 0
    for i in range(len(currGhostStates)):
        currGhostPosition = currGhostStates[i].getPosition()
        distance = util.manhattanDistance(currGhostPosition, currPos) + 0.0000001
        if currScaredTimes[i] > distance:
            ghostScore += pow(10000000 / (distance * 1.0), 2)
        else:
            if distance < 3:
                ghostScore -= 6000 / (distance * 1.0)

    capsuleScore = 0
    for capsulePos in remainingCapsules:
        distance = util.manhattanDistance(capsulePos, currPos) + 0.0000001
        capsuleScore += 50 / (distance * 1.0)

    if len(remainingCapsules) != 0:
        capsuleScore += 1000 / (len(remainingCapsules) * 1.0)
    else:
        capsuleScore += 1000000

    for foodPos in remainingFoods:
        distance = util.manhattanDistance(foodPos, currPos) + 0.0000001
        score += 1 / (distance * 1.0)

    if len(remainingFoods) != 0:
        score += 10000 / (len(remainingFoods) * 1.0)
    else:
        score += 1000000

    return score + 8 * ghostScore + 5 * capsuleScore


# Abbreviation
better = betterEvaluationFunction
