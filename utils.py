import os


def getjsons(path):
    type = ""
    root = []
    if "/nadir" in path:
        root = path.split("/nadir")
        #print(path.split("/nadir"))
        type = "nadir"
    elif "/oblique" in path:
        root = path.split("/obliques")
        #print(path.split("/obliques"))
        type = "obliques"
    parts = root[0].split('/');
    geojson = root[0]+"/"+parts[len(parts)-1]+".geojson"
    json = path[:-3]+"json"
    return geojson, json, type




print(getjsons("/local/2020_hackathon/2020_hackathon/19704962/nadirs/19704962_TXZLUF017018NeighOrtho1352X_190711.jpg"));
print(getjsons("/local/2020_hackathon/2020_hackathon/19704962/obliques/19704962_E_TXZLUF017018NeighObliq1345E_190711.jpg"));



