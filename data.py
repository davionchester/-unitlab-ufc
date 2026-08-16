import io, os, requests, pandas as pd, numpy as np

DEFAULT_URL='https://raw.githubusercontent.com/larissapavan/ufc-historical-fight-dataset/main/fights.csv'

def download_dataset(url=DEFAULT_URL,path='data/fights.csv'):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    r=requests.get(url,timeout=60); r.raise_for_status()
    df=pd.read_csv(io.BytesIO(r.content)); df.to_csv(path,index=False); return df

def load(path='data/fights.csv'): return pd.read_csv(path)

def clean(df):
    d=df.copy()
    # Current source uses event_date + fighter_1/fighter_2 and orders fighter_1 as the winner.
    date_col=next((c for c in ['date','Date','EVENT_DATE','event_date'] if c in d.columns),None)
    d['_date']=pd.to_datetime(d[date_col],errors='coerce') if date_col else pd.NaT
    pairs=[('R_fighter','B_fighter'),('red_fighter','blue_fighter'),('fighter1','fighter2'),('f1','f2'),('fighter_1','fighter_2')]
    for a,b in pairs:
        if a in d.columns and b in d.columns:
            d['_A']=d[a].astype(str).str.strip(); d['_B']=d[b].astype(str).str.strip(); break
    if '_A' not in d: raise ValueError('Could not identify fighter-name columns.')
    # Explicit winner columns take precedence. The current larissapavan dataset documents fighter_1 as the winner.
    if 'winner' in d.columns: d['_winner']=d['winner'].astype(str).str.strip()
    elif 'Winner' in d.columns: d['_winner']=d['Winner'].astype(str).str.strip()
    elif 'result' in d.columns: d['_winner']=d['result'].astype(str).str.strip()
    elif 'outcome' in d.columns:
        d['_winner']=np.where(d['outcome'].astype(str).str.lower().isin(['fighter','fighter1']),d['_A'],np.where(d['outcome'].astype(str).str.lower().isin(['fighter2','fighter_2']),d['_B'],''))
    elif 'f1_result' in d.columns:
        d['_winner']=np.where(d['f1_result'].astype(str).str.upper().eq('W'),d['_A'],np.where(d['f2_result'].astype(str).str.upper().eq('W'),d['_B'],''))
    elif 'fighter_1' in d.columns and 'fighter_2' in d.columns:
        d['_winner']=d['_A']
    else: raise ValueError('Could not identify winner/outcome columns.')
    d['_y']=np.where(d['_winner'].str.lower()==d['_A'].str.lower(),1,np.where(d['_winner'].str.lower()==d['_B'].str.lower(),0,np.nan))
    d=d.dropna(subset=['_y']).copy(); d['_y']=d['_y'].astype(int)
    if d['_date'].notna().sum()>0: d=d.sort_values('_date').reset_index(drop=True)
    return d

def find_col(d,side,keys):
    prefixes=['R_','red_'] if side=='A' else ['B_','blue_']
    for k in keys:
        for p in prefixes:
            if p+k in d.columns:return p+k
    return None

def num(v):
    try: return float(str(v).replace('%','').strip()) if not pd.isna(v) else 0.0
    except: return 0.0

def fight_observation(row,side):
    mapping={'kd':['kd','KD'],'sig_str':['sig_str','SIG_STR'],'td':['td','TD'],'sub':['sub_att','SUB_ATT']}
    out={}
    for k,keys in mapping.items():
        c=find_col(row,side,keys); out[k]=num(row[c]) if c else 0
    return out
