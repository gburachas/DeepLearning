#!/usr/bin/env python
import os, sys
import json
from string import Template

print("n,rbm,usec,measure,value")
t = Template('$n,$rbm,$usec')

for fname in sys.argv[1:]:
    f = open(fname)
    
    for l in f:
        try:
            d = json.loads(l)

            lp = t.substitute(d)
            
            if "error" in d:
                print("%s,error,%s" % (lp, d["error"]["data"]))
                if "holdout" in d["error"]:
                    print("%s,holdout_error,%s" % (lp, d["error"]["holdout"]))
            elif "data_entropy" in d:
                print("%s,data_entropy,%s" % (lp, d["data_entropy"]))
                print("%s,recon_entropy,%s" % (lp, d["recon_entropy"]))
            elif "effect" in d:
                print("%s,effect,%s" % (lp, d["effect"]))
            elif "histograms" in d:
                # { "usec":5098798,"histograms":true,"visible":{n:19434,"min":0,"max":1,"mean":0.114387,"stdev":0.318253,"histogram": [17211,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2223] },"visible_recon":{n:19434,"min":0.0033862,"max":0.999989,"mean":0.111802,"stdev":0.189436,"histogram": [6605,1509,988,860,733,610,621,581,491,461,437,399,383,323,322,287,214,175,196,206,194,180,138,195,138,103,76,91,127,103,91,104,78,65,46,46,40,48,57,35,35,32,28,49,42,24,7,6,9,7,11,30,3,12,21,9,6,3,0,5,7,6,3,5,9,9,4,3,4,8,5,4,0,4,4,9,8,14,24,20,12,16,9,5,5,6,6,14,18,2,18,17,9,12,15,14,28,26,48,309] },"hidden":0.839222,"hidden_recon":0.868445,"visible_biases":{n:6478,"min":-0.0433971,"max":0.0133737,"mean":-0.0321659,"stdev":0.00728715,"histogram": [3,13,48,129,200,271,292,249,207,167,162,155,138,115,156,159,196,180,200,212,248,259,264,274,284,269,254,239,199,165,142,124,103,76,50,31,34,26,16,19,13,6,10,7,2,2,2,3,4,1,2,0,1,2,1,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,2,1,2,1,0,2,2,0,2,1,1,2,1,3,4,0,0,2,0,2,3,2,1,0,6,23,27] },"hidden_biases":{n:256,"min":0.0358982,"max":0.0448507,"mean":0.0409448,"stdev":0.00142999,"histogram": [1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,1,0,1,0,1,1,3,4,0,4,6,3,2,2,7,3,7,3,4,2,5,6,9,7,3,2,1,6,1,7,10,10,8,1,4,4,6,9,3,4,5,3,5,11,7,2,4,3,5,5,9,5,4,6,1,4,1,1,5,2,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1] },"weights":{n:1658368,"min":-0.0718117,"max":0.0890961,"mean":-0.0135081,"stdev":0.0147807,"histogram": [2,2,3,2,10,14,27,56,94,156,271,473,766,1195,1718,2578,3836,5344,7348,9887,13173,17171,21303,26571,32537,38327,44318,51071,57535,63300,68559,72961,76578,79240,80462,80927,80223,77672,74221,70731,65922,60198,54612,48487,42490,36920,31123,26556,21817,17807,14512,11529,9313,7220,5516,4447,3546,2806,2337,1875,1653,1429,1225,1132,1095,1161,1202,1309,1275,1312,1246,1330,1423,1356,1285,1288,1254,1127,1016,846,773,646,527,443,358,252,221,162,102,79,45,41,27,12,9,8,2,1,0,1] },"momentum":{n:1658368,"min":-0.0032651,"max":0.00433099,"mean":-1.6896e-05,"stdev":0.000660728,"histogram": [15,61,55,55,45,67,154,352,410,440,366,902,1438,1372,1224,1181,1331,1617,2331,2897,3349,3183,3221,3802,4715,5621,6119,6547,9585,11355,12018,11898,16715,21911,31198,38063,43245,54837,73159,86174,104723,167184,588568,23363,10948,9023,8222,8331,10466,12468,14312,16845,16557,17903,19064,20296,21146,21883,19622,16571,12766,7242,2991,2143,2183,2279,2491,2635,2898,3427,3775,3816,3367,2731,2177,1696,1388,1124,936,764,575,476,487,399,417,369,363,362,368,297,277,175,140,112,71,50,28,30,17,3] },"n":200,"batch":200,"rbm":"dmr"}
                print("%s,p_hidden,%s" % (lp, d["hidden"]))
                
            
        except ValueError as e:
            sys.stderr.write("---> ERROR: %s" % l)
            sys.stderr.write(str(e))
            pass
        
    f.close()