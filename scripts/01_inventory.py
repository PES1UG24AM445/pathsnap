
import pandas as pd
from pathsnap.io import find_trips
df = pd.DataFrame(find_trips())
print(df.groupby('part').size())
print(df.head(3).to_string())
print('missing files:', df[['nodes', 'arcs', 'route']].isna().sum().to_dict())