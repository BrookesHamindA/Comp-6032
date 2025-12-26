import math
import numpy
import heapq


# a data container for all pertinent information related to fares. (Should we
# add an underway flag and require taxis to acknowledge collection to the dispatcher?)
class FareEntry:
    def __init__(self, origin, dest, time, price=0, taxiIndex=-1):
        self.origin = origin
        self.destination = dest
        self.calltime = time
        self.price = price
        # the taxi allocated to service this fare. -1 if none has been allocated
        self.taxi = taxiIndex
        # a list of indices of taxis that have bid on the fare.
        self.bidders = []


"""
A Dispatcher is a static agent whose job is to allocate fares amongst available taxis. Like the taxis, all
the relevant functionality happens in ClockTick. The Dispatcher has a list of taxis, a map of the service area,
and a dictionary of active fares (ones which have called for a ride) that it can use to manage the allocations.
Taxis bid after receiving the price, which should be decided by the Dispatcher, and once a 'satisfactory' number
of bids are in, the dispatcher should run allocateFare in its world (parent) to inform the winning bidder that they
now have the fare.
"""


class Dispatcher:
    # constructor only needs to know the world it lives in, although you can also populate its knowledge base
    # with taxi and map information.
    def __init__(self, parent, taxis=None, serviceMap=None):
        self._parent = parent
        # our incoming account
        self._revenue = 0
        # the list of taxis
        self._taxis = taxis
        if self._taxis is None:
            self._taxis = []
        # fareBoard will be a nested dictionary indexed by origin, then destination, then call time.
        # Its values are FareEntries. The nesting structure provides for reasonably fast lookup; it's
        # more or less a multi-level hash.
        self._fareBoard = {}
        # serviceMap gives the dispatcher its service area
        self._map = serviceMap
        # track when each taxi last received a fare to balance allocation fairness
        self._lastAllocationTime = {}
        self._maxPickupWindow = 90

    # _________________________________________________________________________________________________________
    # methods to add objects to the Dispatcher's knowledge base

    # make a new taxi known.
    def addTaxi(self, taxi):
        if taxi not in self._taxis:
            self._taxis.append(taxi)

    # incrementally add to the map. This can be useful if, e.g. the world itself has a set of
    # nodes incrementally added. It can then call this function on the dispatcher to add to
    # its map
    def addMapNode(self, coords, neighbours):
        if self._parent is None:
            return AttributeError("This Dispatcher does not exist in any world")
        node = self._parent.getNode(coords[0], coords[1])
        if node is None:
            return KeyError(
                "No such node: {0} in this Dispatcher's service area".format(coords)
            )
        # build up the neighbour dictionary incrementally so we can check for invalid nodes.
        neighbourDict = {}
        for neighbour in neighbours:
            neighbourCoords = (neighbour[1], neighbour[2])
            neighbourNode = self._parent.getNode(neighbour[1], neighbour[2])
            if neighbourNode is None:
                return KeyError(
                    "Node {0} expects neighbour {1} which is not in this Dispatcher's service area".format(
                        coords, neighbour
                    )
                )
            neighbourDict[neighbourCoords] = (
                neighbour[0],
                self._parent.distance2Node(node, neighbourNode),
            )
        self._map[coords] = neighbourDict

    # importMap gets the service area map, and can be brought in incrementally as well as
    # in one wodge.
    def importMap(self, newMap):
        # a fresh map can just be inserted
        if self._map is None:
            self._map = newMap
        # but importing a new map where one exists implies adding to the
        # existing one. (Check that this puts in the right values!)
        else:
            for node in newMap.items():
                neighbours = [
                    (neighbour[1][0], neighbour[0][0], neighbour[0][1])
                    for neighbour in node[1].items()
                ]
                self.addMapNode(node[0], neighbours)

    # any legacy fares or taxis from a previous dispatcher can be imported here - future functionality,
    # for the most part
    def handover(self, parent, origin, destination, time, taxi, price):
        if self._parent == parent:
            # handover implies taxis definitely known to a previous dispatcher. The current
            # dispatcher should thus be made aware of them
            if taxi not in self._taxis:
                self._taxis.append(taxi)
            # add any fares found along with their allocations
            self.newFare(parent, origin, destination, time)
            self._fareBoard[origin][destination][time].taxi = self._taxis.index(taxi)
            self._fareBoard[origin][destination][time].price = price

    # --------------------------------------------------------------------------------------------------------------
    # runtime methods used to inform the Dispatcher of real-time events

    # fares will call this when they appear to signal a request for service.
    def newFare(self, parent, origin, destination, time):
        # only add new fares coming from the same world
        if parent == self._parent:
            fare = FareEntry(origin, destination, time)
            if origin in self._fareBoard:
                if destination not in self._fareBoard[origin]:
                    self._fareBoard[origin][destination] = {}
            else:
                self._fareBoard[origin] = {destination: {}}
            # overwrites any existing fare with the same (origin, destination, calltime) triplet, but
            # this would be equivalent to saying it was the same fare, at least in this world where
            # a given Node only has one fare at a time.
            self._fareBoard[origin][destination][time] = fare

    # abandoning fares will call this to cancel their request
    def cancelFare(self, parent, origin, destination, calltime):
        # if the fare exists in our world,
        if parent == self._parent and origin in self._fareBoard:
            if destination in self._fareBoard[origin]:
                if calltime in self._fareBoard[origin][destination]:
                    # get rid of it
                    print("Fare ({0},{1}) cancelled".format(origin[0], origin[1]))
                    # inform taxis that the fare abandoned
                    self._parent.cancelFare(
                        origin,
                        self._taxis[
                            self._fareBoard[origin][destination][calltime].taxi
                        ],
                    )
                    del self._fareBoard[origin][destination][calltime]
                if len(self._fareBoard[origin][destination]) == 0:
                    del self._fareBoard[origin][destination]
                if len(self._fareBoard[origin]) == 0:
                    del self._fareBoard[origin]

    # taxis register their bids for a fare using this mechanism
    def fareBid(self, origin, taxi):
        # rogue taxis (not known to the dispatcher) can't bid on fares
        if taxi in self._taxis:
            # everyone else bids on fares available
            if origin in self._fareBoard:
                for destination in self._fareBoard[origin].keys():
                    for time in self._fareBoard[origin][destination].keys():
                        # as long as they haven't already been allocated
                        if self._fareBoard[origin][destination][time].taxi == -1:
                            self._fareBoard[origin][destination][time].bidders.append(
                                self._taxis.index(taxi)
                            )
                            # only one fare per origin can be actively open for bid, so
                            # immediately return once we[ve found it
                            return

    # fares call this (through the parent world) when they have reached their destination
    def recvPayment(self, parent, amount):
        # don't take payments from dodgy alternative universes
        if self._parent == parent:
            self._revenue += amount

    # ________________________________________________________________________________________________________________

    # clockTick is called by the world and drives the simulation for the Dispatcher. It must, at minimum, handle the
    # 2 main functions the dispatcher needs to run in the world: broadcastFare(origin, destination, price) and
    # allocateFare(origin, taxi).
    def clockTick(self, parent):
        if self._parent == parent:
            for origin in self._fareBoard.keys():
                for destination in self._fareBoard[origin].keys():
                    # not super-efficient here: need times in order, dictionary view objects are not
                    # sortable because they are an iterator, so we need to turn the times into a
                    # sorted list. Hopefully fareBoard will never be too big
                    for time in sorted(
                        list(self._fareBoard[origin][destination].keys())
                    ):
                        if self._fareBoard[origin][destination][time].price == 0:
                            self._fareBoard[origin][destination][
                                time
                            ].price = self._costFare(
                                self._fareBoard[origin][destination][time]
                            )
                            # broadcastFare actually returns the number of taxis that got the info, if you
                            # wish to use that information in the decision over when to allocate
                            self._parent.broadcastFare(
                                origin,
                                destination,
                                self._fareBoard[origin][destination][time].price,
                            )
                        elif (
                            self._fareBoard[origin][destination][time].taxi < 0
                            and len(self._fareBoard[origin][destination][time].bidders)
                            > 0
                        ):
                            self._allocateFare(origin, destination, time)

    # ----------------------------------------------------------------------------------------------------------------

    """ HERE IS THE PART THAT YOU NEED TO MODIFY
      """

    """this internal method decides a dynamic cost for the fare, taking into account estimated
         travel time, expected pickup delay, and local congestion. The goal is to balance
         profitability with a high probability that the fare will accept the price and not cancel.
      """
    def _costFare(self, fare):
        origin_node = self._parent.getNode(fare.origin[0], fare.origin[1])
        destination_node = self._parent.getNode(
            fare.destination[0], fare.destination[1]
        )
        timeToDestination = self._parent.travelTime(origin_node, destination_node)
        if timeToDestination < 0:
            return 150

        avg_pickup = self._estimatePickupTime(origin_node)
        wait_discount = max(0.85, 1 - (avg_pickup / 200))
        congestion_factor = 1.0
        if origin_node is not None and origin_node.maxTraffic > 0:
            congestion_factor += min(
                0.3, origin_node.traffic / float(origin_node.maxTraffic)
            )

        base_price = 5 + timeToDestination * 1.2
        price = base_price * wait_discount * congestion_factor
        upper_bound = max(20, timeToDestination * 6)
        lower_bound = max(5, timeToDestination * 0.8)
        return max(lower_bound, min(upper_bound, price))

    # this method decides which taxi to allocate to a given fare.
    # Task 2: Advanced dispatcher optimization with multi-factor scoring:
    # Considers proximity, workload, capital, traffic conditions, and waiting time.
    def _allocateFare(self, origin, destination, time):
        fare_entry = self._fareBoard[origin][destination][time]
        bidders = fare_entry.bidders
        if len(bidders) == 0:
            return

        fareNode = self._parent.getNode(origin[0], origin[1])
        destNode = self._parent.getNode(destination[0], destination[1])
        if fareNode is None or destNode is None:
            return

        allocatedTaxi = -1
        bestScore = float("-inf")
        waiting_time = self._parent.simTime - time

        for taxiIdx in bidders:
            if taxiIdx >= len(self._taxis):
                continue
            taxi = self._taxis[taxiIdx]
            bidderLoc = taxi.currentLocation
            if bidderLoc[0] < 0 or bidderLoc[1] < 0:
                continue

            bidderNode = self._parent.getNode(bidderLoc[0], bidderLoc[1])
            if bidderNode is None:
                continue

            travel_time = self._parent.travelTime(bidderNode, fareNode)
            if travel_time < 0:
                travel_time = self._parent.distance2Node(bidderNode, fareNode)
            pickup_with_wait = waiting_time + travel_time
            if pickup_with_wait > self._maxPickupWindow:
                continue
            proximityScore = max(0, 100 - travel_time * 2)

            # workload considers current passenger and queued allocations
            workloadScore = 100
            if getattr(taxi, "_passenger", None) is not None:
                workloadScore -= 60
            allocated_fares = [
                fare
                for fare in getattr(taxi, "_availableFares", {}).values()
                if getattr(fare, "allocated", False)
            ]
            workloadScore -= min(40, len(allocated_fares) * 20)
            workloadScore = max(0, workloadScore)

            # capital health keeps taxis with low balance active
            daily_loss = max(1, getattr(taxi, "_dailyLoss", 1))
            capitalScore = max(
                10, min(100, (getattr(taxi, "_account", 0) / daily_loss) * 100)
            )

            # fairness penalty discourages repeatedly allocating to same taxi
            last_alloc = self._lastAllocationTime.get(taxiIdx, None)
            fairnessPenalty = 0
            if last_alloc is not None:
                ticks_since = self._parent.simTime - last_alloc
                fairnessPenalty = max(0, 40 - ticks_since)

            urgencyBoost = max(0, 100 - pickup_with_wait * 2)

            totalScore = (
                proximityScore * 0.35
                + workloadScore * 0.2
                + capitalScore * 0.2
                + urgencyBoost * 0.15
                - fairnessPenalty * 0.1
            )

            if totalScore > bestScore:
                bestScore = totalScore
                allocatedTaxi = taxiIdx

        if allocatedTaxi >= 0:
            fare_entry.taxi = allocatedTaxi
            self._lastAllocationTime[allocatedTaxi] = self._parent.simTime
            self._parent.allocateFare(origin, self._taxis[allocatedTaxi])

    def _estimatePickupTime(self, origin_node):
        if origin_node is None:
            return 30
        travel_times = []
        for taxi in self._taxis:
            loc = taxi.currentLocation
            if loc[0] < 0 or loc[1] < 0:
                continue
            taxi_node = self._parent.getNode(loc[0], loc[1])
            if taxi_node is None:
                continue
            travel_time = self._parent.travelTime(taxi_node, origin_node)
            if travel_time >= 0:
                travel_times.append(travel_time)
        if len(travel_times) == 0:
            return 30
        return sum(travel_times) / len(travel_times)
