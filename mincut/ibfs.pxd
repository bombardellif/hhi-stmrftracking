# distutils: language=c++

from libcpp cimport bool

cdef extern from "ibfs/ibfs.h":
    ctypedef enum IBFSInitMode "IBFSGraph::IBFSInitMode":
        IB_INIT_FAST "IBFSGraph::IB_INIT_FAST",
        IB_INIT_COMPACT "IBFSGraph::IB_INIT_COMPACT",
    cdef cppclass IBFSGraph:
        # Constructor
        IBFSGraph(IBFSInitMode initMode) except +

        void initSize(int numNodes, int numEdges)

        # Functions for creating the graph
        void addEdge(int nodeIndexFrom,
            int nodeIndexTo,
            int capacity,
            int reverseCapacity)
        void addNode(int nodeIndex, int capFromSource, int capToSink)

        # Functions for changing graph's weights
        void incEdge(int nodeIndexFrom,
            int nodeIndexTo,
            int capacity,
            int reverseCapacity)
        void incNode(int nodeIndex, int deltaCapFromSource, int deltaCapToSink)

        # Initialize inner data structures
        void initGraph()
        # Run optimization algorithm and return optimum function's value
        int computeMaxFlow(bool allowIncrements)
        # Return 1 if node is in subset S, 0 if it is in subset T
        int isNodeOnSrcSide(int nodeIndex, int freeNodeValue)
