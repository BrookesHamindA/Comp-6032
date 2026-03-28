import pygame
import threading
import time
import math
import sys
import json
import copy

import networld
import taxi
import dispatcher
import metrics

from ruparams import *


def runRoboUber(
    worldX,
    worldY,
    runTime,
    stop,
    junctions=None,
    streets=None,
    interpolate=False,
    outputValues=None,
    oLock=None,
    **args,
):
    # Create metrics collector
    run_id = args.get("run_id", 1)
    algorithm = args.get("algorithm", "astar")
    metrics_collector = metrics.MetricsCollector(
        run_id=run_id, algorithm_type=algorithm, simulation_minutes=runTime
    )

    if "fareProb" not in args:
        args["fareProb"] = 0.001

    if "fareFile" not in args:
        args["fareFile"] = None

    print("Creating world...")
    svcArea = networld.NetWorld(
        x=worldX,
        y=worldY,
        runtime=runTime,
        fareprob=args["fareProb"],
        jctNodes=junctions,
        edges=streets,
        interpolateNodes=interpolate,
        farefile=args["fareFile"],
        metrics=metrics_collector,
    )
    print("Exporting map...")
    svcMap = svcArea.exportMap()
    if "serviceMap" in args:
        args["serviceMap"] = svcMap

    print("Creating taxis")
    try:
        configured_taxis = taxi_configs
    except NameError:
        configured_taxis = None

    taxis = []
    if configured_taxis is None or len(configured_taxis) == 0:
        configured_taxis = [
            {"taxi_num": 100, "start_point": (20, 0)},
            {"taxi_num": 101, "start_point": (49, 15)},
            {"taxi_num": 102, "start_point": (15, 49)},
            {"taxi_num": 103, "start_point": (0, 35)},
        ]

    for idx, cfg in enumerate(configured_taxis):
        taxi_num = cfg.get("taxi_num", 100 + idx)
        idle_loss = cfg.get("idle_loss", 1024)
        start_point = cfg.get("start_point")
        taxis.append(
            taxi.Taxi(
                world=svcArea,
                taxi_num=taxi_num,
                idle_loss=idle_loss,
                service_area=svcMap,
                start_point=start_point,
            )
        )

    print("Adding a dispatcher")
    dispatcher0 = dispatcher.Dispatcher(parent=svcArea, taxis=taxis)

    svcArea.addDispatcher(dispatcher0)

    print("Bringing taxis on duty")
    for onDutyTaxi in taxis:
        onDutyTaxi.comeOnDuty()

    threadRunTime = runTime
    threadTime = 0
    print("Starting world")
    while threadTime < threadRunTime:
        args["ackStop"].wait()
        if stop.is_set():
            threadRunTime = 0
        else:
            svcArea.runWorld(ticks=1, outputs=outputValues, outlock=oLock)
            if threadTime != svcArea.simTime:
                threadTime += 1
            time.sleep(0.01)

    print("\nSimulation complete - collecting metrics...")
    metrics_collector.finalize()
    metrics_collector.print_summary()
    metrics_collector.save_to_csv("simulation_results.csv")
    print("")


if recordFares:
    fareFile = open("./faretypes.csv", "a")
    print(
        '"{0}"'.format("FareType"),
        '"{0}"'.format("originX"),
        '"{0}"'.format("originY"),
        '"{0}"'.format("destX"),
        '"{0}"'.format("destY"),
        '"{0}"'.format("MaxWait"),
        '"{0}"'.format("MaxCost"),
        sep=",",
        file=fareFile,
    )
else:
    fareFile = None

userExit = threading.Event()
userConfirmExit = threading.Event()
userConfirmExit.set()

pygame.init()
displaySurface = pygame.display.set_mode(size=displaySize, flags=pygame.RESIZABLE)
backgroundRect = None
aspectRatio = worldX / worldY
if aspectRatio > 4 / 3:
    activeSize = (displaySize[0] - 100, (displaySize[0] - 100) / aspectRatio)
else:
    activeSize = (aspectRatio * (displaySize[1] - 100), displaySize[1] - 100)
displayedBackground = pygame.Surface(activeSize)
displayedBackground.fill(pygame.Color(255, 255, 255))
activeRect = pygame.Rect(
    round((displaySize[0] - activeSize[0]) / 2),
    round((displaySize[1] - activeSize[1]) / 2),
    activeSize[0],
    activeSize[1],
)

meshSize = ((activeSize[0] / worldX), round(activeSize[1] / worldY))

positions = [
    [
        pygame.Rect(
            round(x * meshSize[0]),
            round(y * meshSize[1]),
            round(meshSize[0]),
            round(meshSize[1]),
        )
        for y in range(worldY)
    ]
    for x in range(worldX)
]
drawPositions = [
    [displayedBackground.subsurface(positions[x][y]) for y in range(worldY)]
    for x in range(worldX)
]

jctRect = pygame.Rect(
    round(meshSize[0] / 4),
    round(meshSize[1] / 4),
    round(meshSize[0] / 2),
    round(meshSize[1] / 2),
)
jctSquares = [drawPositions[jct[0]][jct[1]].subsurface(jctRect) for jct in junctionIdxs]

for street in streets:
    pygame.draw.aaline(
        displayedBackground,
        pygame.Color(128, 128, 128),
        (
            round(street.nodeA[0] * meshSize[0] + meshSize[0] / 2),
            round(street.nodeA[1] * meshSize[1] + meshSize[1] / 2),
        ),
        (
            round(street.nodeB[0] * meshSize[0] + meshSize[0] / 2),
            round(street.nodeB[1] * meshSize[1] + meshSize[1] / 2),
        ),
    )

for jct in range(len(junctionIdxs)):
    jctSquares[jct].fill(pygame.Color(192, 192, 192))
    pygame.draw.rect(
        jctSquares[jct],
        pygame.Color(128, 128, 128),
        pygame.Rect(0, 0, round(meshSize[0] / 2), round(meshSize[1] / 2)),
        5,
    )

displaySurface.blit(displayedBackground, activeRect)
pygame.display.flip()

taxiColours = {}
taxiPalette = [
    pygame.Color(0, 0, 0),
    pygame.Color(0, 0, 255),
    pygame.Color(0, 255, 0),
    pygame.Color(255, 0, 0),
    pygame.Color(255, 0, 255),
    pygame.Color(0, 255, 255),
    pygame.Color(255, 255, 0),
    pygame.Color(255, 255, 255),
]

taxiRect = pygame.Rect(
    round(meshSize[0] / 3),
    round(meshSize[1] / 3),
    round(meshSize[0] / 3),
    round(meshSize[1] / 3),
)

fareRect = pygame.Rect(
    round(3 * meshSize[0] / 8),
    round(3 * meshSize[1] / 8),
    round(meshSize[0] / 4),
    round(meshSize[1] / 4),
)

for run in range(numDays):
    outputValues = {"time": [], "fares": {}, "taxis": {}}
    outputLock = threading.Lock()

    roboUber = threading.Thread(
        target=runRoboUber,
        name="RoboUberThread",
        kwargs={
            "worldX": worldX,
            "worldY": worldY,
            "runTime": runTime,
            "stop": userExit,
            "ackStop": userConfirmExit,
            "junctions": junctions,
            "streets": streets,
            "interpolate": True,
            "outputValues": outputValues,
            "oLock": outputLock,
            "fareProb": fGenDefault,
            "fareFile": fareFile,
        },
    )

    curTime = 0
    roboUber.start()

    while curTime < runTime - 1:
        events = pygame.event.get()
        for evt in events:
            if evt.type == pygame.QUIT:
                print("Window close requested. Exiting...")
                userExit.set()
                userConfirmExit.set()
                roboUber.join()
                if fareFile is not None:
                    try:
                        fareFile.close()
                    except Exception:
                        pass
                pygame.quit()
                sys.exit()

        try:
            quitevent = next(evt for evt in events if evt.type == pygame.KEYDOWN)
            if quitevent.key == pygame.K_q:
                userConfirmExit.clear()
                print("Really quit? Press Y to quit, any other key to ignore")
                while not userConfirmExit.is_set():
                    try:
                        events2 = pygame.event.get()
                        quitevent = next(
                            evt for evt in events2 if evt.type == pygame.KEYDOWN
                        )
                        if quitevent.key == pygame.K_y:
                            userExit.set()
                            userConfirmExit.set()
                            roboUber.join()
                            if fareFile is not None:
                                try:
                                    fareFile.close()
                                except Exception:
                                    pass
                            pygame.quit()
                            sys.exit()
                        userConfirmExit.set()
                    except StopIteration:
                        continue
        except StopIteration:
            if (
                "time" in outputValues
                and len(outputValues["time"]) > 0
                and curTime != outputValues["time"][-1]
            ):
                print("curTime: {0}, world.time: {1}".format(
                    curTime, outputValues["time"][-1]
                ))

                displayedBackground.fill(pygame.Color(255, 255, 255))

                for street in streets:
                    pygame.draw.aaline(
                        displayedBackground,
                        pygame.Color(128, 128, 128),
                        (
                            round(street.nodeA[0] * meshSize[0] + meshSize[0] / 2),
                            round(street.nodeA[1] * meshSize[1] + meshSize[1] / 2),
                        ),
                        (
                            round(street.nodeB[0] * meshSize[0] + meshSize[0] / 2),
                            round(street.nodeB[1] * meshSize[1] + meshSize[1] / 2),
                        ),
                    )

                for jct in range(len(junctionIdxs)):
                    jctSquares[jct].fill(pygame.Color(192, 192, 192))
                    pygame.draw.rect(
                        jctSquares[jct],
                        pygame.Color(128, 128, 128),
                        pygame.Rect(0, 0, round(meshSize[0] / 2), round(meshSize[1] / 2)),
                        5,
                    )

                outputLock.acquire()
                faresToRedraw = dict(
                    [
                        (
                            rfare[0],
                            dict(
                                [
                                    (time[0], time[1])
                                    for time in rfare[1].items()
                                    if time[0] > curTime
                                ]
                            ),
                        )
                        for rfare in outputValues["fares"].items()
                        if max(rfare[1].keys()) > curTime
                    ]
                )
                outputLock.release()
                outputLock.acquire()
                taxisToRedraw = dict(
                    [
                        (
                            rtaxi[0],
                            dict(
                                [
                                    (taxiPos[0], taxiPos[1])
                                    for taxiPos in rtaxi[1].items()
                                    if taxiPos[0] > curTime
                                ]
                            ),
                        )
                        for rtaxi in outputValues["taxis"].items()
                        if max(rtaxi[1].keys()) > curTime
                    ]
                )
                outputLock.release()

                if len(taxisToRedraw) > 0:
                    for rtaxi in taxisToRedraw.items():
                        if rtaxi[0] not in taxiColours and len(taxiPalette) > 0:
                            taxiColours[rtaxi[0]] = taxiPalette.pop(0)
                        if rtaxi[0] in taxiColours:
                            newestTime = max(rtaxi[1].keys())
                            pygame.draw.circle(
                                drawPositions[rtaxi[1][newestTime][0]][
                                    rtaxi[1][newestTime][1]
                                ],
                                taxiColours[rtaxi[0]],
                                (round(meshSize[0] / 2), round(meshSize[1] / 2)),
                                round(meshSize[0] / 3),
                            )

                if len(faresToRedraw) > 0:
                    for rfare in faresToRedraw.items():
                        newestFareTime = max(rfare[1].keys())
                        pygame.draw.polygon(
                            drawPositions[rfare[0][0]][rfare[0][1]],
                            pygame.Color(255, 128, 0),
                            [
                                (meshSize[0] / 2, meshSize[1] / 4),
                                (
                                    meshSize[0] / 2 - math.cos(math.pi / 6) * meshSize[1] / 4,
                                    meshSize[1] / 2 + math.sin(math.pi / 6) * meshSize[1] / 4,
                                ),
                                (
                                    meshSize[0] / 2 + math.cos(math.pi / 6) * meshSize[1] / 4,
                                    meshSize[1] / 2 + math.sin(math.pi / 6) * meshSize[1] / 4,
                                ),
                            ],
                        )

                displaySurface.blit(displayedBackground, activeRect)
                pygame.display.flip()
                curTime += 1

    roboUber.join()
    print("end of day: {0}".format(run))

if fareFile is not None:
    fareFile.close()
pygame.quit()
sys.exit()