import pandas as pd
from tqdm import tqdm
from pathsnap.io import find_trips, load_trip
from pathsnap.features import build_table
from pathsnap.config import PROC, RES

stats = []
for part in ('train', 'student'):  # evaluator is never processed here
    infos = [i for i in find_trips() if i['part'] == part]
    df = pd.concat([build_table(load_trip(i)) for i in tqdm(infos, desc=part)], ignore_index=True)
    df.to_parquet(PROC / (part + '.parquet'))
    has_pos = df.groupby(['tid', 'obs'])['label'].max()
    stats.append(dict(
        part=part,
        trips=len(infos),
        observations=len(has_pos),
        candidate_rows=len(df),
        candidate_miss_rate=float((has_pos < 1).mean()),
        positive_rate=float((df.label == 1).mean())
    ))
    print(stats[-1])
pd.DataFrame(stats).to_csv(RES / 'candidate_stats.csv', index=False)
