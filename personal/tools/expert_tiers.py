"""Download Baalorlord's and Jorbs' beta tier lists from sts2.untapped.gg
(public pages) into expert_tiers.json: {expert: {character: {CARD_ID:
[tier, comment, comment_version, good_upgrade]}}}. Local data, not shipped."""
import json, re, sys, urllib.request

IDS = {"baalorlord": {"ironclad": "004de170-026a-4dd4-a280-3b904be0b5d6", "silent": "6d61ea21-0552-4c49-8bb5-a5c15530fc00",
                      "defect": "5a512e04-4583-4a16-9271-d46864c6cb4c", "necrobinder": "43d0b41f-7d6d-4ce9-928e-c1310a413983",
                      "regent": "0e6c1e23-bec6-4887-a9e0-dbf49ede974d", "colorless": "0a626c22-dc49-433e-ac6e-76cc9abf5684"},
       "jorbs": {"ironclad": "6b3390b7-bcbc-4ebc-ae78-ae9a400c66dc", "silent": "2cd72d4b-eeb4-42ab-89ca-4ac682137672",
                 "defect": "5f8d3e04-1f50-4b56-b51a-d8650b000ce3", "necrobinder": "3ae661e6-6190-4a45-b00e-63e8400fff94",
                 "regent": "c47dbfda-4983-4107-8c9c-c33d5ae32b23", "colorless": "81684acf-26a7-4e44-a649-84f4c2824ec0"}}
HEAD = re.compile(r'"uuid":"[0-9a-f-]{36}","name":"([^"]{1,40})","order":\d+,"color":"[^"]*","cards":\[')
ITEM = re.compile(r'"commentary":(null|"(?:[^"\\]|\\.)*"),"commentary_version":(null|"[^"]*")'
                  r'((?:,"[a-z_]+":(?:null|true|false|\d+|"[^"]*"))*),"item_id":"([A-Z0-9_]+)"')


def parse(uid):
    req = urllib.request.Request(f"https://sts2.untapped.gg/en/tier-list/{uid}/beta", headers={"User-Agent": "Mozilla/5.0"})
    t = re.sub(r'\\+"', '"', urllib.request.urlopen(req, timeout=60).read().decode("utf-8"))
    heads = [(m.group(1), m.end()) for m in HEAD.finditer(t)]
    out = {}
    for i, (name, start) in enumerate(heads):
        end = heads[i + 1][1] if i + 1 < len(heads) else start + 400000
        for c in ITEM.finditer(t, start, end):
            out.setdefault(c.group(4), [name, "" if c.group(1) == "null" else c.group(1)[1:-1],
                                        "" if c.group(2) == "null" else c.group(2)[1:-1], '"good_upgrade":true' in c.group(3)])
    return out


res = {e: {ch: parse(u) for ch, u in d.items()} for e, d in IDS.items()}
json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
for e, d in res.items():
    print(e, {ch: len(v) for ch, v in d.items()})
