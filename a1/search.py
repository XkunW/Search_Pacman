# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import copy


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    OPEN = util.Stack()
    start_state = problem.getStartState()
    start_path = [start_state]
    start = Node(start_state, start_path)
    OPEN.push(start)
    while not OPEN.isEmpty():
        n = OPEN.pop()
        if problem.isGoalState(n.state):
            return n.actions
        successors = problem.getSuccessors(n.state)
        for successor in successors:
            if successor[0] not in n.path:
                succ = Node(n.state, n.path, n.actions, n.cost)
                succ.successor(successor[0], successor[1], successor[2])
                OPEN.push(succ)
    return False


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    OPEN = util.Queue()
    start_state = problem.getStartState()
    start_path = [start_state]
    start = Node(start_state, start_path)
    OPEN.push(start)
    seen = {
        start_state: start.cost
    }
    while not OPEN.isEmpty():
        n = OPEN.pop()
        if n.cost <= seen[n.state]:
            if problem.isGoalState(n.state):
                return n.actions
            successors = problem.getSuccessors(n.state)
            for successor in successors:
                if (successor[0] not in seen) or \
                        ((successor[2] + n.cost) < seen[successor[0]]):
                    succ = Node(n.state, n.path, n.actions, n.cost)
                    succ.successor(successor[0], successor[1], successor[2])
                    OPEN.push(succ)
                    seen[successor[0]] = succ.cost
    return False


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    OPEN = util.PriorityQueue()
    start_state = problem.getStartState()
    start_path = [start_state]
    start = Node(start_state, start_path)
    OPEN.push(start, start.cost)
    seen = {
        start_state: start.cost
    }
    while not OPEN.isEmpty():
        n = OPEN.pop()
        if problem.isGoalState(n.state):
            return n.actions
        successors = problem.getSuccessors(n.state)
        for successor in successors:
            if (successor[0] not in seen) or \
                    ((successor[2] + n.cost) < seen[successor[0]]):
                succ = Node(n.state, n.path, n.actions, n.cost)
                succ.successor(successor[0], successor[1], successor[2])
                OPEN.push(succ, succ.cost)
                seen[successor[0]] = succ.cost

    return False


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    OPEN = util.PriorityQueue()
    start_state = problem.getStartState()
    start_path = [start_state]
    start_heuristic = heuristic(start_state, problem)
    start = Node(start_state, start_path)
    OPEN.push(start, start.cost)
    seen = {
        start_state: start.cost + start_heuristic
    }
    while not OPEN.isEmpty():
        n = OPEN.pop()
        if problem.isGoalState(n.state):
            return n.actions
        successors = problem.getSuccessors(n.state)
        for successor in successors:
            succ_heuristic = heuristic(successor[0], problem)
            if (successor[0] not in seen) or \
                    ((successor[2] + n.cost + succ_heuristic) < seen[successor[0]]):
                succ = Node(n.state, n.path, n.actions, n.cost)
                succ.successor(successor[0], successor[1], successor[2])
                OPEN.push(succ, succ.cost + succ_heuristic)
                seen[successor[0]] = succ.cost + succ_heuristic

    return False


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


class Node:
    """
    Class for nodes in OPEN
    """

    def __init__(self, state=(-1, -1), path=[], actions=[], cost=0):
        self.state = state
        self.path = copy.deepcopy(path)
        self.actions = copy.deepcopy(actions)
        self.cost = cost

    def successor(self, new_state, new_action, new_cost):
        self.state = new_state
        self.path.append(new_state)
        self.actions.append(new_action)
        self.cost += new_cost
