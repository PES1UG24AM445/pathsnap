import pandas as pd
from pathsnap.io import find_trips, _read

df = pd.DataFrame(find_trips())
print(df.groupby('part').size())
print(df.head(3).to_string())
print('missing files:', df[['nodes', 'arcs', 'route']].isna().sum().to_dict())
if len(df) > 0:
    print('first track row (lon, lat, sec):', _read(df.iloc[0]['track'])[0])
